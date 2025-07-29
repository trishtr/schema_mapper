# Embedding-Based Schema Mapping

## Overview

The schema mapping system uses embedding models to generate semantic representations of database columns and perform intelligent mapping based on similarity search. This document explains the embedding strategy and mapping process.

## Embedding Generation

### Column Text Generation

The system generates rich text descriptions for columns by combining:

1. **Field Description**:

   ```python
   description = get_field_description(column_name)
   # Example: "Unique identifier for patient records"
   ```

2. **Contextual Description**:

   ```python
   context = generate_contextual_description(
       column_name,
       data_type,
       sample_values
   )
   # Example: "Domain: healthcare | Entity Type: identifier"
   ```

3. **Data Type Information**:
   ```python
   compatible_types = get_compatible_types(data_type)
   # Example: "Data type VARCHAR compatible with: TEXT, CHAR"
   ```

### Business Rules Integration

The system incorporates domain-specific business rules:

```python
# Healthcare field descriptions
HEALTHCARE_FIELDS = {
    "patient_id": "Unique identifier for patient records",
    "npi": "National Provider Identifier",
    "icd_code": "International Classification of Diseases code"
}

# Data type compatibility
DATA_TYPE_COMPATIBILITY = {
    "VARCHAR": ["VARCHAR", "TEXT", "CHAR"],
    "INTEGER": ["INTEGER", "BIGINT", "SMALLINT"]
}
```

### Vector Store Management

The system uses ChromaDB for efficient vector storage and similarity search:

```python
# Initialize vector store
collection = chromadb.PersistentClient(path="./chroma_db")

# Store embeddings
collection.add(
    embeddings=[embedding_vector],
    documents=[column_text],
    metadatas=[metadata],
    ids=[unique_id]
)
```

## Mapping Process

### 1. Generate Embeddings

```python
async def generate_column_embedding(
    self,
    column_info: Dict[str, Any]
) -> Dict[str, Any]:
    # Generate rich text description
    column_text = self._build_column_text(
        column_info["name"],
        column_info["data_type"],
        column_info.get("sample_values", [])
    )

    # Generate embedding
    embedding = self.model.encode(column_text)

    return {
        "embedding": embedding,
        "metadata": metadata,
        "text": column_text
    }
```

### 2. Find Similar Columns

```python
async def find_similar_columns(
    self,
    query_embedding: np.ndarray,
    n_results: int = 5,
    score_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    # Search vector store
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    # Process matches
    matches = []
    for distance, metadata in zip(
        results["distances"][0],
        results["metadatas"][0]
    ):
        similarity = 1 - (distance / 2)
        if similarity >= score_threshold:
            matches.append({
                "column_name": metadata["name"],
                "similarity_score": similarity,
                "data_type": metadata["data_type"]
            })

    return matches
```

### 3. Map Schemas

```python
async def map_schemas_with_embeddings(
    self,
    source_columns: List[Dict[str, Any]],
    target_columns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    # Build index for target columns
    await self.build_vector_index(target_columns)

    mappings = []
    for source_column in source_columns:
        # Generate embedding
        source_result = await self.generate_column_embedding(
            source_column
        )

        # Find matches
        matches = await self.find_similar_columns(
            source_result["embedding"]
        )

        if matches:
            mappings.append({
                "source_column": source_column["name"],
                "matches": matches
            })

    return mappings
```

## Configuration

### 1. Embedding Model

```python
# Initialize model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Model parameters
model_config = {
    "max_seq_length": 128,
    "do_lower_case": True
}
```

### 2. Vector Store

```python
# ChromaDB configuration
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

## Examples

### 1. Simple Mapping

```python
# Source column
source = {
    "name": "patient_id",
    "data_type": "VARCHAR",
    "sample_values": ["P123", "P456"]
}

# Target column
target = {
    "name": "id",
    "data_type": "VARCHAR",
    "sample_values": ["PAT123"]
}

# Map columns
result = await mapper.map_schemas_with_embeddings(
    [source],
    [target]
)
```

### 2. Complex Mapping

```python
# Source columns
source_columns = [
    {
        "name": "email_address",
        "data_type": "VARCHAR",
        "sample_values": ["user@example.com"]
    },
    {
        "name": "birth_date",
        "data_type": "DATE",
        "sample_values": ["1990-01-01"]
    }
]

# Target columns
target_columns = [
    {
        "name": "email",
        "data_type": "VARCHAR",
        "sample_values": ["contact@example.com"]
    },
    {
        "name": "dob",
        "data_type": "DATE",
        "sample_values": ["1985-12-31"]
    }
]

# Map schemas
result = await mapper.map_schemas_with_embeddings(
    source_columns,
    target_columns
)
```

## Troubleshooting

### Common Issues

1. **Low Similarity Scores**

   - Check text generation
   - Verify sample values
   - Adjust thresholds
   - Consider domain context

2. **Performance Issues**

   - Monitor vector store size
   - Check batch sizes
   - Optimize search parameters
   - Consider caching

3. **Quality Issues**
   - Review business rules
   - Check data types
   - Validate constraints
   - Add more context

## Next Steps

1. **Enhancements**

   - Add more domain rules
   - Improve text generation
   - Optimize search
   - Add caching

2. **Integration**
   - Add API endpoints
   - Implement monitoring
   - Add validation
   - Create UI
