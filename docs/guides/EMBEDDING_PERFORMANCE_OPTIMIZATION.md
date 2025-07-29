# Embedding Service Performance Optimization Guide

## Overview

This guide provides comprehensive strategies to optimize the performance of the embedding service for schema mapping. Performance optimization is crucial for handling large-scale schema mapping tasks efficiently.

## Current Performance Bottlenecks

### 1. Sequential Processing

- **Issue**: Embeddings are generated one by one
- **Impact**: Linear time complexity O(n)
- **Solution**: Implement batch processing

### 2. Model Loading Overhead

- **Issue**: Model is loaded on every service instantiation
- **Impact**: Slow startup times
- **Solution**: Implement model caching and lazy loading

### 3. Memory Inefficiency

- **Issue**: Large embeddings stored in memory
- **Impact**: High memory usage
- **Solution**: Implement memory-efficient storage

### 4. ChromaDB Query Optimization

- **Issue**: Inefficient similarity search
- **Impact**: Slow query response times
- **Solution**: Optimize index and query parameters

## Performance Optimization Strategies

### 1. Batch Processing

#### Implementation

```python
async def batch_generate_embeddings(
    self,
    columns: List[Dict[str, Any]],
    batch_size: int = 32
) -> List[Dict[str, Any]]:
    """Generate embeddings in batches for better performance."""
    results = []

    for i in range(0, len(columns), batch_size):
        batch = columns[i:i + batch_size]

        # Prepare batch texts
        texts = [
            self._build_column_text(
                col["name"],
                col["data_type"],
                col.get("sample_values", [])
            )
            for col in batch
        ]

        # Generate embeddings in batch
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=False
        )

        # Process results
        for idx, (column, embedding) in enumerate(zip(batch, embeddings)):
            result = {
                "embedding": embedding,
                "metadata": self._prepare_metadata(column),
                "text": texts[idx]
            }
            results.append(result)

    return results
```

#### Benefits

- **Speed**: 3-5x faster than sequential processing
- **Memory**: Better memory utilization
- **GPU**: Efficient GPU utilization if available

### 2. Model Optimization

#### Quantization

```python
from sentence_transformers import SentenceTransformer
import torch

class OptimizedEmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load model with quantization
        self.model = SentenceTransformer(model_name)

        # Quantize model for faster inference
        if torch.cuda.is_available():
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
```

#### Model Caching

```python
import functools
from typing import Dict, Any

class ModelCache:
    _instances: Dict[str, Any] = {}

    @classmethod
    def get_model(cls, model_name: str) -> SentenceTransformer:
        if model_name not in cls._instances:
            cls._instances[model_name] = SentenceTransformer(model_name)
        return cls._instances[model_name]
```

### 3. Memory Optimization

#### Efficient Storage

```python
import numpy as np
from typing import List, Dict, Any

class MemoryOptimizedEmbeddingService:
    def __init__(self):
        self.embeddings_cache = {}
        self.max_cache_size = 1000

    def _optimize_embedding_storage(
        self,
        embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """Optimize embedding storage using float16 precision."""
        # Convert to float16 for memory efficiency
        optimized = np.array(embeddings, dtype=np.float16)

        # Apply compression if needed
        if len(optimized) > 1000:
            # Use PCA for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=384)  # Reduce from 512 to 384
            optimized = pca.fit_transform(optimized)

        return optimized
```

#### Lazy Loading

```python
class LazyEmbeddingService:
    def __init__(self):
        self._model = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._init_collection()
        return self._collection
```

### 4. ChromaDB Optimization

#### Index Optimization

```python
class OptimizedChromaDBService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create optimized collection
        self.collection = self.client.create_collection(
            name="optimized_schema_embeddings",
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            embedding_function=None  # Use custom embeddings
        )

    async def optimized_batch_add(
        self,
        embeddings: List[np.ndarray],
        metadatas: List[Dict],
        documents: List[str],
        ids: List[str],
        batch_size: int = 100
    ):
        """Add embeddings in optimized batches."""
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
```

#### Query Optimization

```python
async def optimized_similarity_search(
    self,
    query_embedding: np.ndarray,
    n_results: int = 5,
    score_threshold: float = 0.7,
    use_approximate: bool = True
) -> List[Dict[str, Any]]:
    """Optimized similarity search with better parameters."""

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
                "compatible_types": json.loads(metadata["compatible_types"]),
                "constraints": json.loads(metadata["constraints"]),
                "sample_values": json.loads(metadata["sample_values"])
            })

    # Return top results
    return matches[:n_results]
```

### 5. Caching Strategies

#### Multi-Level Caching

```python
import asyncio
from functools import lru_cache
import hashlib

class MultiLevelCache:
    def __init__(self):
        self.memory_cache = {}
        self.redis_cache = None  # Redis client
        self.cache_ttl = 3600  # 1 hour

    def _generate_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()

    @lru_cache(maxsize=1000)
    def get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """Get embedding from memory cache."""
        return self.memory_cache.get(cache_key)

    async def get_embedding_with_cache(
        self,
        column_info: Dict[str, Any]
    ) -> np.ndarray:
        """Get embedding with multi-level caching."""
        cache_key = self._generate_cache_key(column_info)

        # Check memory cache first
        cached = self.get_cached_embedding(cache_key)
        if cached is not None:
            return cached

        # Check Redis cache
        if self.redis_cache:
            cached = await self.redis_cache.get(cache_key)
            if cached:
                embedding = np.frombuffer(cached, dtype=np.float32)
                self.memory_cache[cache_key] = embedding
                return embedding

        # Generate new embedding
        embedding = await self._generate_embedding(column_info)

        # Store in caches
        self.memory_cache[cache_key] = embedding
        if self.redis_cache:
            await self.redis_cache.setex(
                cache_key,
                self.cache_ttl,
                embedding.tobytes()
            )

        return embedding
```

### 6. Parallel Processing

#### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelEmbeddingService:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

    async def parallel_batch_process(
        self,
        columns: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Process embeddings in parallel batches."""
        tasks = []

        for i in range(0, len(columns), batch_size):
            batch = columns[i:i + batch_size]
            task = asyncio.create_task(
                self._process_batch_with_semaphore(batch)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return [item for batch_result in results for item in batch_result]

    async def _process_batch_with_semaphore(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process batch with semaphore for concurrency control."""
        async with self.semaphore:
            return await self._process_batch(batch)

    async def _process_batch(
        self,
        batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a single batch."""
        loop = asyncio.get_event_loop()

        # Run CPU-intensive embedding generation in thread pool
        texts = [
            self._build_column_text(
                col["name"],
                col["data_type"],
                col.get("sample_values", [])
            )
            for col in batch
        ]

        embeddings = await loop.run_in_executor(
            self.executor,
            lambda: self.model.encode(texts, convert_to_tensor=False)
        )

        return [
            {
                "embedding": embedding,
                "metadata": self._prepare_metadata(column),
                "text": text
            }
            for embedding, column, text in zip(embeddings, batch, texts)
        ]
```

## Performance Monitoring

### Metrics Collection

```python
import time
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PerformanceMetrics:
    embedding_generation_time: float
    similarity_search_time: float
    memory_usage: float
    cache_hit_rate: float
    batch_size: int
    total_columns: int

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []

    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics.append(metrics)

    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics."""
        if not self.metrics:
            return {}

        return {
            "avg_embedding_time": sum(m.embedding_generation_time for m in self.metrics) / len(self.metrics),
            "avg_search_time": sum(m.similarity_search_time for m in self.metrics) / len(self.metrics),
            "avg_memory_usage": sum(m.memory_usage for m in self.metrics) / len(self.metrics),
            "avg_cache_hit_rate": sum(m.cache_hit_rate for m in self.metrics) / len(self.metrics)
        }
```

### Benchmarking

```python
async def benchmark_embedding_service(
    service: EmbeddingService,
    test_columns: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Benchmark embedding service performance."""
    start_time = time.time()

    # Test embedding generation
    embedding_start = time.time()
    embeddings = await service.batch_generate_embeddings(test_columns)
    embedding_time = time.time() - embedding_start

    # Test similarity search
    search_start = time.time()
    for embedding in embeddings[:10]:  # Test with first 10
        await service.find_similar_columns(embedding["embedding"])
    search_time = time.time() - search_start

    total_time = time.time() - start_time

    return {
        "total_time": total_time,
        "embedding_time": embedding_time,
        "search_time": search_time,
        "columns_per_second": len(test_columns) / embedding_time,
        "searches_per_second": 10 / search_time
    }
```

## Configuration Optimization

### Environment Variables

```bash
# Performance configuration
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING_USE_GPU=true
EMBEDDING_QUANTIZATION=true
EMBEDDING_CACHE_TTL=3600
EMBEDDING_MAX_WORKERS=4
EMBEDDING_USE_APPROXIMATE_SEARCH=true
```

### Configuration Class

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class EmbeddingConfig:
    batch_size: int = 32
    model_name: str = "all-MiniLM-L6-v2"
    use_gpu: bool = True
    quantization: bool = True
    cache_ttl: int = 3600
    max_workers: int = 4
    use_approximate_search: bool = True
    memory_limit_mb: int = 1024
    enable_caching: bool = True
    enable_parallel_processing: bool = True
```

## Best Practices

### 1. Batch Size Optimization

- **Small datasets (< 1000 columns)**: batch_size = 16
- **Medium datasets (1000-10000 columns)**: batch_size = 32
- **Large datasets (> 10000 columns)**: batch_size = 64

### 2. Memory Management

- Monitor memory usage during processing
- Use garbage collection for large datasets
- Implement memory pooling for embeddings

### 3. Caching Strategy

- Cache frequently accessed embeddings
- Use TTL-based cache expiration
- Implement cache warming for common queries

### 4. Error Handling

- Implement retry logic for failed operations
- Use circuit breaker pattern for external services
- Graceful degradation for performance issues

## Expected Performance Improvements

### Speed Improvements

- **Batch Processing**: 3-5x faster embedding generation
- **GPU Acceleration**: 2-3x faster with CUDA
- **Quantization**: 1.5-2x faster inference
- **Parallel Processing**: 2-4x faster for large datasets

### Memory Improvements

- **Float16 Precision**: 50% memory reduction
- **Lazy Loading**: 30% memory reduction
- **Efficient Caching**: 40% memory optimization

### Scalability Improvements

- **Horizontal Scaling**: Support for multiple instances
- **Load Balancing**: Distribute processing across nodes
- **Database Optimization**: Efficient ChromaDB usage

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Embedding Generation Time**: Target < 100ms per column
2. **Similarity Search Time**: Target < 50ms per query
3. **Memory Usage**: Target < 80% of available memory
4. **Cache Hit Rate**: Target > 70%
5. **Error Rate**: Target < 1%

### Alerting Rules

```python
def check_performance_alerts(metrics: PerformanceMetrics):
    """Check for performance alerts."""
    alerts = []

    if metrics.embedding_generation_time > 0.1:  # 100ms
        alerts.append("Embedding generation time exceeded threshold")

    if metrics.similarity_search_time > 0.05:  # 50ms
        alerts.append("Similarity search time exceeded threshold")

    if metrics.memory_usage > 0.8:  # 80%
        alerts.append("Memory usage exceeded threshold")

    if metrics.cache_hit_rate < 0.7:  # 70%
        alerts.append("Cache hit rate below threshold")

    return alerts
```

## Conclusion

Implementing these performance optimizations can significantly improve the embedding service's efficiency and scalability. The key is to:

1. **Start with batch processing** for immediate improvements
2. **Add caching** for frequently accessed data
3. **Implement parallel processing** for large datasets
4. **Monitor performance** continuously
5. **Optimize incrementally** based on usage patterns

These optimizations will enable the embedding service to handle large-scale schema mapping tasks efficiently while maintaining high accuracy and reliability.
