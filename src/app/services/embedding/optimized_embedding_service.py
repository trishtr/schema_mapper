"""
Optimized embedding service with performance improvements.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import json
import logging
import asyncio
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import torch
from dataclasses import dataclass

from src.app.utils.error_handling import EmbeddingError
from .config.business_rules import (
    get_field_description,
    get_compatible_types,
    get_field_constraints
)
from .config.contextual_descriptions import (
    generate_contextual_description,
    get_semantic_relationships,
    enrich_field_context
)

@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    embedding_generation_time: float
    similarity_search_time: float
    memory_usage: float
    cache_hit_rate: float
    batch_size: int
    total_columns: int

class ModelCache:
    """Singleton model cache for efficient model loading."""
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def get_model(cls, model_name: str) -> SentenceTransformer:
        """Get or create model instance."""
        if model_name not in cls._instances:
            cls._instances[model_name] = SentenceTransformer(model_name)
            
            # Optimize model if GPU is available
            if torch.cuda.is_available():
                cls._instances[model_name] = cls._instances[model_name].to('cuda')
                
                # Apply quantization for faster inference
                try:
                    cls._instances[model_name] = torch.quantization.quantize_dynamic(
                        cls._instances[model_name],
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                except Exception as e:
                    logging.warning(f"Quantization failed: {e}")
        
        return cls._instances[model_name]

class OptimizedEmbeddingService:
    """Optimized service for generating and managing embeddings."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_workers: int = 4,
        enable_caching: bool = True,
        cache_ttl: int = 3600
    ):
        # Initialize model with caching
        self.model = ModelCache.get_model(model_name)
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Initialize ChromaDB with optimized settings
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create optimized collection
        self.collection = self.chroma_client.create_collection(
            name="optimized_schema_embeddings",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
        
        # Initialize caches and thread pool
        self.embedding_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
        
        # Performance monitoring
        self.metrics: List[PerformanceMetrics] = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """Get embedding from memory cache."""
        return self.embedding_cache.get(cache_key)
    
    async def generate_column_embedding(
        self,
        column_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate embedding for a database column with caching."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.enable_caching:
                cache_key = self._generate_cache_key(column_info)
                cached_embedding = self._get_cached_embedding(cache_key)
                
                if cached_embedding is not None:
                    self.cache_hits += 1
                    return {
                        "embedding": cached_embedding,
                        "metadata": self._prepare_metadata(column_info),
                        "text": self._build_column_text(
                            column_info["name"],
                            column_info["data_type"],
                            column_info.get("sample_values", [])
                        ),
                        "cached": True
                    }
                
                self.cache_misses += 1
            
            # Extract column information
            column_name = column_info["name"]
            data_type = column_info["data_type"]
            sample_values = column_info.get("sample_values", [])
            
            # Generate rich column description
            column_text = self._build_column_text(
                column_name,
                data_type,
                sample_values
            )
            
            # Generate embedding using thread pool for CPU-intensive work
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                lambda: self.model.encode(
                    column_text,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
            )
            
            # Prepare metadata
            metadata = self._prepare_metadata(column_info)
            
            # Cache the result
            if self.enable_caching:
                cache_key = self._generate_cache_key(column_info)
                self.embedding_cache[cache_key] = embedding
                
                # Limit cache size
                if len(self.embedding_cache) > 1000:
                    # Remove oldest entries
                    oldest_keys = list(self.embedding_cache.keys())[:100]
                    for key in oldest_keys:
                        del self.embedding_cache[key]
            
            generation_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(generation_time, 0, 1)
            
            return {
                "embedding": embedding,
                "metadata": metadata,
                "text": column_text,
                "cached": False
            }
            
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(
                f"Failed to generate embedding for {column_info.get('name', 'unknown')}",
                details={"error": str(e)}
            )
    
    async def batch_generate_embeddings(
        self,
        columns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate embeddings in batches for better performance."""
        start_time = time.time()
        results = []
        
        try:
            for i in range(0, len(columns), self.batch_size):
                batch = columns[i:i + self.batch_size]
                
                # Process batch with semaphore for concurrency control
                async with self.semaphore:
                    batch_results = await self._process_batch(batch)
                    results.extend(batch_results)
            
            generation_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(generation_time, 0, len(columns))
            
            return results
            
        except Exception as e:
            logging.error(f"Batch embedding generation failed: {str(e)}")
            raise EmbeddingError(
                "Failed to generate batch embeddings",
                details={"error": str(e)}
            )
    
    async def _process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a single batch of columns."""
        # Prepare batch texts
        texts = [
            self._build_column_text(
                col["name"],
                col["data_type"],
                col.get("sample_values", [])
            )
            for col in batch
        ]
        
        # Generate embeddings in batch using thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=False
            )
        )
        
        # Process results
        batch_results = []
        for idx, (column, embedding) in enumerate(zip(batch, embeddings)):
            result = {
                "embedding": embedding,
                "metadata": self._prepare_metadata(column),
                "text": texts[idx],
                "cached": False
            }
            batch_results.append(result)
        
        return batch_results
    
    async def build_vector_index(
        self,
        columns: List[Dict[str, Any]]
    ) -> None:
        """Build vector index for multiple columns with optimization."""
        try:
            # Clear existing collection
            self.chroma_client.delete_collection("optimized_schema_embeddings")
            self.collection = self.chroma_client.create_collection(
                name="optimized_schema_embeddings",
                metadata={"hnsw:space": "cosine"},
                embedding_function=None
            )
            
            # Generate embeddings in batches
            embeddings_data = await self.batch_generate_embeddings(columns)
            
            # Prepare data for ChromaDB
            embeddings = [data["embedding"] for data in embeddings_data]
            metadatas = [data["metadata"] for data in embeddings_data]
            documents = [data["text"] for data in embeddings_data]
            ids = [f"col_{idx}" for idx in range(len(embeddings_data))]
            
            # Add to collection in optimized batches
            await self._optimized_batch_add(
                embeddings, metadatas, documents, ids
            )
            
        except Exception as e:
            logging.error(f"Vector index build failed: {str(e)}")
            raise EmbeddingError(
                "Failed to build vector index",
                details={"error": str(e)}
            )
    
    async def _optimized_batch_add(
        self,
        embeddings: List[np.ndarray],
        metadatas: List[Dict],
        documents: List[str],
        ids: List[str],
        batch_size: int = 100
    ):
        """Add embeddings to ChromaDB in optimized batches."""
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            self.collection.add(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents,
                ids=batch_ids
            )
    
    async def find_similar_columns(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        score_threshold: float = 0.7,
        use_approximate: bool = True
    ) -> List[Dict[str, Any]]:
        """Find similar columns using optimized vector similarity search."""
        start_time = time.time()
        
        try:
            # Use approximate search for better performance
            if use_approximate:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results * 2,  # Get more results for filtering
                    include=["metadatas", "distances"]
                )
            else:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["metadatas", "distances"]
                )
            
            # Filter and process results
            matches = []
            for idx, (distance, metadata) in enumerate(
                zip(results["distances"][0], results["metadatas"][0])
            ):
                similarity = 1 - (distance / 2)
                
                if similarity >= score_threshold:
                    matches.append({
                        "column_name": metadata["name"],
                        "similarity_score": similarity,
                        "data_type": metadata["data_type"],
                        "compatible_types": json.loads(
                            metadata["compatible_types"]
                        ),
                        "constraints": json.loads(metadata["constraints"]),
                        "sample_values": json.loads(metadata["sample_values"])
                    })
            
            search_time = time.time() - start_time
            
            # Record metrics
            self._record_metrics(0, search_time, 1)
            
            # Return top results
            return matches[:n_results]
            
        except Exception as e:
            logging.error(f"Similar columns search failed: {str(e)}")
            raise EmbeddingError(
                "Failed to find similar columns",
                details={"error": str(e)}
            )
    
    async def map_schemas_with_embeddings(
        self,
        source_columns: List[Dict[str, Any]],
        target_columns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Map schemas using optimized embeddings and similarity search."""
        try:
            # Build index for target columns
            await self.build_vector_index(target_columns)
            
            mappings = []
            
            # Process source columns in parallel
            tasks = []
            for source_column in source_columns:
                task = asyncio.create_task(
                    self._map_single_column(source_column)
                )
                tasks.append(task)
            
            # Wait for all mappings to complete
            mapping_results = await asyncio.gather(*tasks)
            mappings.extend(mapping_results)
            
            return mappings
            
        except Exception as e:
            logging.error(f"Schema mapping failed: {str(e)}")
            raise EmbeddingError(
                "Failed to map schemas",
                details={"error": str(e)}
            )
    
    async def _map_single_column(
        self,
        source_column: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map a single source column to target columns."""
        # Generate embedding for source column
        source_result = await self.generate_column_embedding(source_column)
        
        # Find similar target columns
        matches = await self.find_similar_columns(
            source_result["embedding"]
        )
        
        if matches:
            # Add semantic relationships
            relationships = get_semantic_relationships(source_column["name"])
            
            return {
                "source_column": source_column["name"],
                "matches": matches,
                "semantic_relationships": relationships
            }
        
        return {
            "source_column": source_column["name"],
            "matches": [],
            "semantic_relationships": []
        }
    
    def _prepare_metadata(self, column_info: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for embedding storage."""
        return {
            "name": column_info["name"],
            "data_type": column_info["data_type"],
            "compatible_types": json.dumps(
                get_compatible_types(column_info["data_type"])
            ),
            "constraints": json.dumps(
                get_field_constraints(
                    column_info["name"],
                    column_info["data_type"]
                )
            ),
            "sample_values": json.dumps(
                column_info.get("sample_values", [])[:5]
            ),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _build_column_text(
        self,
        column_name: str,
        data_type: str,
        sample_values: List[str]
    ) -> str:
        """Build rich text description for a column."""
        components = []
        
        # Add field description
        description = get_field_description(column_name)
        if description:
            components.append(description)
        
        # Add contextual description
        context = generate_contextual_description(
            column_name,
            data_type,
            sample_values
        )
        components.append(context)
        
        # Add data type information
        compatible_types = get_compatible_types(data_type)
        components.append(
            f"Data type {data_type} compatible with: {', '.join(compatible_types)}"
        )
        
        # Add constraints
        constraints = get_field_constraints(column_name, data_type)
        if constraints:
            constraint_str = ", ".join(
                f"{k}: {v}" for k, v in constraints.items()
            )
            components.append(f"Constraints: {constraint_str}")
        
        return " | ".join(components)
    
    def _record_metrics(
        self,
        embedding_time: float,
        search_time: float,
        columns_processed: int
    ):
        """Record performance metrics."""
        import psutil
        
        memory_usage = psutil.virtual_memory().percent / 100.0
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0.0
        )
        
        metrics = PerformanceMetrics(
            embedding_generation_time=embedding_time,
            similarity_search_time=search_time,
            memory_usage=memory_usage,
            cache_hit_rate=cache_hit_rate,
            batch_size=self.batch_size,
            total_columns=columns_processed
        )
        
        self.metrics.append(metrics)
        
        # Keep only last 100 metrics
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if not self.metrics:
            return {}
        
        return {
            "avg_embedding_time": sum(m.embedding_generation_time for m in self.metrics) / len(self.metrics),
            "avg_search_time": sum(m.similarity_search_time for m in self.metrics) / len(self.metrics),
            "avg_memory_usage": sum(m.memory_usage for m in self.metrics) / len(self.metrics),
            "avg_cache_hit_rate": sum(m.cache_hit_rate for m in self.metrics) / len(self.metrics),
            "total_columns_processed": sum(m.total_columns for m in self.metrics),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True) 