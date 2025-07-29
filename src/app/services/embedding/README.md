# Embedding Service

The Embedding Service is a core component of the schema mapping system that provides semantic understanding of database columns through vector embeddings.

## Overview

The embedding service uses state-of-the-art language models to convert column metadata (names, descriptions, data types, sample values) into high-dimensional vector representations. These embeddings enable semantic similarity search for intelligent schema mapping.

## Features

- **Semantic Embeddings**: Generate embeddings for column metadata using SentenceTransformers
- **Vector Search**: Efficient similarity search using ChromaDB
- **Contextual Descriptions**: Enhanced embeddings with business rules and domain knowledge
- **Batch Processing**: Process multiple columns efficiently
- **Caching**: Intelligent caching for improved performance
- **Configurable**: Flexible configuration for different use cases

## Architecture

```
EmbeddingService
├── Embedding Generation
│   ├── Column Metadata Processing
│   ├── Contextual Description Enhancement
│   └── Vector Generation
├── Vector Storage
│   ├── ChromaDB Integration
│   ├── Collection Management
│   └── Metadata Storage
├── Similarity Search
│   ├── k-NN Search
│   ├── Confidence Scoring
│   └── Result Ranking
└── Configuration
    ├── Business Rules
    ├── Contextual Descriptions
    └── Model Settings
```

## Components

### Core Service (`embedding_service.py`)

The main service class that orchestrates embedding generation and similarity search.

**Key Methods:**

- `generate_embedding()`: Create embeddings for column metadata
- `search_similar_columns()`: Find similar columns using vector search
- `batch_process()`: Process multiple columns efficiently
- `clear_collection()`: Reset the vector store

### Business Rules (`config/business_rules.py`)

Defines domain-specific rules and data type compatibility matrices.

**Features:**

- Healthcare field mappings
- Data type compatibility rules
- Field constraint definitions
- Semantic relationship mappings

### Contextual Descriptions (`config/contextual_descriptions.py`)

Enhances column descriptions with business context and domain knowledge.

**Features:**

- Healthcare domain contexts
- Semantic relationship detection
- Field context enrichment
- Business rule integration

## Usage

### Basic Usage

```python
from src.app.services.embedding.embedding_service import EmbeddingService

# Initialize service
embedding_service = EmbeddingService()

# Generate embedding for a column
column_info = {
    "name": "patient_id",
    "data_type": "VARCHAR",
    "description": "Unique patient identifier",
    "sample_values": ["P001", "P002", "P003"]
}

embedding = await embedding_service.generate_embedding(column_info)

# Search for similar columns
similar_columns = await embedding_service.search_similar_columns(
    target_embedding=embedding,
    top_k=5
)
```

### Advanced Usage with Context

```python
# Enhanced column info with business context
enhanced_info = {
    "name": "patient_id",
    "data_type": "VARCHAR",
    "description": "Unique patient identifier",
    "sample_values": ["P001", "P002", "P003"],
    "business_context": "healthcare_patient_management",
    "domain": "healthcare",
    "constraints": ["unique", "not_null"]
}

embedding = await embedding_service.generate_embedding(enhanced_info)
```

## Configuration

### Model Settings

The service uses the `all-MiniLM-L6-v2` model by default, which provides a good balance between performance and accuracy.

### ChromaDB Configuration

- **Persistent Storage**: Embeddings are stored in `./chroma_db`
- **Collection Management**: Automatic collection creation and management
- **Metadata Storage**: Rich metadata for each embedding

### Business Rules Configuration

Business rules are defined in `config/business_rules.py` and include:

- Field name patterns
- Data type compatibility matrices
- Domain-specific mappings
- Constraint definitions

## Performance Considerations

### Caching Strategy

- **Embedding Cache**: Caches generated embeddings to avoid recomputation
- **Search Cache**: Caches similarity search results
- **TTL-based**: Automatic cache expiration

### Optimization Tips

1. **Batch Processing**: Use `batch_process()` for multiple columns
2. **Collection Management**: Clear collections when starting fresh
3. **Metadata Optimization**: Include relevant metadata for better search
4. **Model Selection**: Choose appropriate model for your use case

## Error Handling

The service includes comprehensive error handling:

- **Model Loading Errors**: Graceful fallback to default model
- **ChromaDB Errors**: Automatic retry with exponential backoff
- **Input Validation**: Robust validation of column metadata
- **Memory Management**: Automatic cleanup of large datasets

## Integration

### With Schema Mapper

The embedding service integrates seamlessly with the schema mapper:

```python
from src.app.services.schema.schema_mapper import SchemaMapper

schema_mapper = SchemaMapper()
mappings = await schema_mapper.map_schemas(
    source_schema=source_columns,
    target_schema=target_columns
)
```

### With LLM Service

For low-confidence mappings, the embedding service works with the LLM service:

```python
# High confidence: Direct mapping
if confidence > 0.8:
    return mapping

# Low confidence: LLM refinement
else:
    llm_result = await llm_service.refine_mapping(mapping)
    return llm_result
```

## Testing

### Unit Tests

```bash
# Run embedding service tests
pytest tests/test_embedding_service.py
```

### Integration Tests

```bash
# Run full integration tests
pytest tests/test_integration.py
```

### Performance Tests

```bash
# Run performance benchmarks
python scripts/benchmark_embedding.py
```

## Monitoring

### Metrics

The service provides various metrics for monitoring:

- **Embedding Generation Time**: Time to generate embeddings
- **Search Performance**: Query response times
- **Cache Hit Rate**: Cache effectiveness
- **Memory Usage**: Memory consumption patterns

### Logging

Comprehensive logging for debugging and monitoring:

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**

   - Check internet connection
   - Verify model cache directory permissions
   - Try alternative model

2. **ChromaDB Errors**

   - Check disk space
   - Verify database directory permissions
   - Clear and recreate collection

3. **Memory Issues**
   - Reduce batch size
   - Clear cache periodically
   - Use smaller model

### Debug Mode

Enable debug mode for detailed logging:

```python
embedding_service = EmbeddingService(debug=True)
```

## Future Enhancements

### Planned Features

1. **Multi-Model Support**: Support for multiple embedding models
2. **Dynamic Context**: Real-time context updates
3. **Federated Learning**: Collaborative model improvement
4. **Advanced Caching**: Redis-based distributed caching
5. **Custom Models**: Support for domain-specific models

### Performance Improvements

1. **GPU Acceleration**: CUDA support for faster processing
2. **Quantization**: Model quantization for reduced memory usage
3. **Streaming**: Real-time embedding generation
4. **Parallel Processing**: Multi-threaded batch processing

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up development environment
4. Run tests: `pytest tests/`

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add comprehensive docstrings
- Write unit tests for new features

### Pull Request Process

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
