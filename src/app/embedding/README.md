# Embedding Service

## Overview

The embedding service is responsible for generating semantic representations of database columns using sentence transformers and managing vector similarity search.

## Components

### 1. Embedding Generation

```python
async def generate_column_embedding(
    self,
    column_info: Dict[str, Any]
) -> Dict[str, Any]:
    # Generate rich text description
    # Create embedding
    # Add metadata
```

Key features:

- Uses SentenceTransformer model (`all-MiniLM-L6-v2`)
- Incorporates business rules and context
- Handles metadata generation

### 2. Vector Search

```python
async def find_similar_columns(
    self,
    query_embedding: np.ndarray,
    n_results: int = 5,
    score_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    # Search vector store
    # Process matches
    # Filter by threshold
```

Features:

- ChromaDB vector store
- Cosine similarity search
- Configurable thresholds
- Metadata filtering

### 3. Schema Mapping

```python
async def map_schemas_with_embeddings(
    self,
    source_columns: List[Dict[str, Any]],
    target_columns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    # Build vector index
    # Find matches
    # Apply rules
```

Features:

- Batch processing
- Rule application
- Confidence scoring
- Semantic relationships

## Configuration

### 1. Model Settings

```python
model_config = {
    "name": "all-MiniLM-L6-v2",
    "max_seq_length": 128,
    "do_lower_case": True
}
```

### 2. Vector Store

```python
vector_store_config = {
    "path": "./chroma_db",
    "collection_name": "schema_embeddings",
    "distance_metric": "cosine"
}
```

### 3. Mapping Parameters

```python
mapping_config = {
    "similarity_threshold": 0.7,
    "max_matches": 5,
    "min_confidence": 0.8
}
```

## Usage

### 1. Initialize Service

```python
from src.app.services.embedding.embedding_service import EmbeddingService

service = EmbeddingService()
```

### 2. Generate Embeddings

```python
embedding = await service.generate_column_embedding({
    "name": "patient_id",
    "data_type": "VARCHAR",
    "sample_values": ["P123", "P456"]
})
```

### 3. Map Schemas

```python
mappings = await service.map_schemas_with_embeddings(
    source_columns,
    target_columns
)
```

## Best Practices

### 1. Text Generation

- Include domain-specific context
- Add data type information
- Include sample values
- Consider constraints

### 2. Vector Search

- Use appropriate thresholds
- Consider multiple matches
- Handle edge cases
- Validate results

### 3. Performance

- Batch process embeddings
- Cache common queries
- Monitor vector store size
- Optimize search parameters

## Error Handling

### 1. Common Issues

```python
try:
    embedding = await service.generate_column_embedding(column)
except EmbeddingError as e:
    logging.error(f"Embedding generation failed: {str(e)}")
    # Handle error
```

### 2. Recovery Strategies

- Use cache if available
- Apply fallback rules
- Log errors for monitoring
- Implement retries

## Testing

### 1. Unit Tests

```python
def test_embedding_generation():
    embedding = await service.generate_column_embedding(column)
    assert embedding is not None
    assert len(embedding) == 384  # Expected dimension
```

### 2. Integration Tests

```python
def test_schema_mapping():
    result = await service.map_schemas_with_embeddings(
        source_schema,
        target_schema
    )
    assert result["mappings"]
    assert result["stats"]["total_mappings"] > 0
```

## Dependencies

- sentence-transformers
- chromadb
- numpy
- pandas

## Contributing

1. Fork the repository
2. Create your feature branch
3. Add tests for new features
4. Submit a pull request
