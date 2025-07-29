# Quick Start Guide

## Installation

1. Clone the repository:

```bash
git clone https://github.com/trishtr/schema_mapper.git
cd schema_mapper
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment:

```bash
cp docs/config.env.example .env
# Edit .env with your settings
```

## Basic Usage

### 1. Map Schemas

```python
from schema_mapper import SchemaMapper

# Initialize mapper
mapper = SchemaMapper()

# Define schemas
source_schema = [
    {
        "name": "patient_id",
        "data_type": "VARCHAR",
        "sample_values": ["P123", "P456"]
    },
    {
        "name": "email_address",
        "data_type": "VARCHAR",
        "sample_values": ["john@example.com"]
    }
]

target_schema = [
    {
        "name": "id",
        "data_type": "VARCHAR",
        "sample_values": ["PAT123"]
    },
    {
        "name": "email",
        "data_type": "VARCHAR",
        "sample_values": ["jane@example.com"]
    }
]

# Map schemas
result = await mapper.map_schemas(source_schema, target_schema)
print(result["mappings"])
```

### 2. Profile Data

```python
from schema_mapper import DataProfiler

# Initialize profiler
profiler = DataProfiler()

# Profile column
profile = await profiler.profile_column(
    column_data=["P123", "P456", "P789"],
    column_info={
        "name": "patient_id",
        "data_type": "VARCHAR"
    }
)

print(profile["statistics"])
print(profile["patterns"])
print(profile["quality"])
```

### 3. Use API

```bash
# Start the API server
uvicorn src.app.main:app --reload

# Make a request
curl -X POST http://localhost:8000/api/v1/mapping/map-schemas \
  -H "Content-Type: application/json" \
  -d '{
    "source_schema": [...],
    "target_schema": [...]
  }'
```

## Configuration

### 1. Vector Store

```python
# Configure ChromaDB
VECTOR_STORE_PATH = "./chroma_db"
collection = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
```

### 2. Caching

```python
# Configure Redis
REDIS_URL = "redis://localhost:6379"
cache = CacheManager(redis_url=REDIS_URL)
```

### 3. LLM Service

```python
# Configure LLM
LLM_API_URL = "http://localhost:8000"
llm = LLMService(api_url=LLM_API_URL)
```

## Examples

### 1. High-Confidence Mapping

```python
# Map with high confidence threshold
result = await mapper.map_schemas(
    source_schema,
    target_schema,
    confidence_threshold=0.9
)

# Check high-confidence mappings
high_confidence = [
    m for m in result["mappings"]
    if m["confidence_score"] >= 0.9
]
```

### 2. Complex Mapping with LLM

```python
# Handle low-confidence case
if mapping["confidence_score"] < 0.8:
    analysis = await llm_service.analyze_mapping(
        source_column,
        target_column,
        mapping
    )
    if analysis["is_valid"]:
        mapping["confidence_score"] = analysis["confidence"]
        mapping["explanation"] = analysis["explanation"]
```

### 3. Data Quality Check

```python
# Profile and check quality
profile = await profiler.profile_column(data, info)
quality = profile["quality"]

if quality["completeness"] < 0.95:
    print(f"Warning: Low data completeness ({quality['completeness']})")
```

## Common Issues

### 1. Installation

If you encounter installation issues:

```bash
# Update pip
pip install --upgrade pip

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install python3-dev

# Install with verbose output
pip install -v -r requirements.txt
```

### 2. Vector Store

If ChromaDB fails:

```python
# Clear and rebuild index
await embedding_service.rebuild_vector_index()
```

### 3. Cache

If Redis is unavailable:

```python
# System will fall back to in-memory cache
cache = CacheManager(redis_url=None)
```

## Next Steps

1. Read the [Comprehensive Guide](COMPREHENSIVE_GUIDE.md)
2. Explore [Technical Details](TECHNICAL_DEEP_DIVE.md)
3. Check [API Reference](api/README.md)
4. Review [Examples](examples/)
