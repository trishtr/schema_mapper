"""
Embedding service for generating and managing schema embeddings.
"""
from typing import List, Dict, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
import logging
from datetime import datetime

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

class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.create_collection(
            name="schema_embeddings",
            metadata={"description": "Schema column embeddings"}
        )
        
    async def generate_column_embedding(
        self,
        column_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate embedding for a database column."""
        try:
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
            
            # Generate embedding
            embedding = self.model.encode(
                column_text,
                convert_to_tensor=False
            )
            
            # Prepare metadata
            metadata = {
                "name": column_name,
                "data_type": data_type,
                "compatible_types": json.dumps(
                    get_compatible_types(data_type)
                ),
                "constraints": json.dumps(
                    get_field_constraints(column_name, data_type)
                ),
                "sample_values": json.dumps(sample_values[:5]),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return {
                "embedding": embedding,
                "metadata": metadata,
                "text": column_text
            }
            
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            raise EmbeddingError(
                f"Failed to generate embedding for {column_name}",
                details={"error": str(e)}
            )
    
    async def build_vector_index(
        self,
        columns: List[Dict[str, Any]]
    ) -> None:
        """Build vector index for multiple columns."""
        try:
            # Clear existing collection
            self.chroma_client.delete_collection("schema_embeddings")
            self.collection = self.chroma_client.create_collection(
                name="schema_embeddings",
                metadata={"description": "Schema column embeddings"}
            )
            
            embeddings = []
            metadatas = []
            documents = []
            ids = []
            
            # Generate embeddings for all columns
            for idx, column in enumerate(columns):
                result = await self.generate_column_embedding(column)
                
                embeddings.append(result["embedding"])
                metadatas.append(result["metadata"])
                documents.append(result["text"])
                ids.append(f"col_{idx}")
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
                ids=ids
            )
            
        except Exception as e:
            logging.error(f"Vector index build failed: {str(e)}")
            raise EmbeddingError(
                "Failed to build vector index",
                details={"error": str(e)}
            )
    
    async def find_similar_columns(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar columns using vector similarity."""
        try:
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            matches = []
            for idx, (distance, metadata) in enumerate(
                zip(results["distances"][0], results["metadatas"][0])
            ):
                # Convert distance to similarity score
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
            
            return matches
            
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
        """Map schemas using embeddings and similarity search."""
        try:
            # Build index for target columns
            await self.build_vector_index(target_columns)
            
            mappings = []
            
            # Find matches for each source column
            for source_column in source_columns:
                # Generate embedding for source column
                source_result = await self.generate_column_embedding(
                    source_column
                )
                
                # Find similar target columns
                matches = await self.find_similar_columns(
                    source_result["embedding"]
                )
                
                if matches:
                    # Add semantic relationships
                    relationships = get_semantic_relationships(
                        source_column["name"]
                    )
                    
                    mappings.append({
                        "source_column": source_column["name"],
                        "matches": matches,
                        "semantic_relationships": relationships
                    })
            
            return mappings
            
        except Exception as e:
            logging.error(f"Schema mapping failed: {str(e)}")
            raise EmbeddingError(
                "Failed to map schemas",
                details={"error": str(e)}
            )
    
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