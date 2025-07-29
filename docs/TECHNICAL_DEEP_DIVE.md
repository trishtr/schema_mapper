# Schema Mapper: Technical Deep Dive

## Core Components

### 1. Embedding Generation

The embedding service uses SentenceTransformer to generate semantic representations:

```python
class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )
```

Key features:

- Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Vector store: ChromaDB (persistent storage)
- Rich text generation with business rules
- Contextual descriptions and semantic relationships

### 2. Vector Search

Vector similarity search using ChromaDB:

```python
async def find_similar_columns(
    self,
    query_embedding: np.ndarray,
    n_results: int = 5,
    score_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
```

Features:

- Cosine similarity search
- Configurable thresholds
- Metadata filtering
- Batch processing

### 3. Schema Mapping

The schema mapper orchestrates the mapping process:

```python
async def map_schemas(
    self,
    source_columns: List[Dict[str, Any]],
    target_columns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    # Get initial mappings
    initial_mappings = await self.embedding_service.map_schemas_with_embeddings(
        source_columns,
        target_columns
    )

    # Process each mapping
    for mapping in initial_mappings:
        processed_mapping = await self._process_column_mapping(
            mapping,
            source_columns,
            target_columns
        )
```

Components:

- Initial embedding-based mapping
- Rule-based processing
- Confidence scoring
- LLM integration for complex cases

### 4. Data Profiling

The profiler analyzes column characteristics:

```python
async def profile_column(
    self,
    column_name: str,
    data_type: str,
    sample_values: List[Any]
) -> Dict[str, Any]:
    # Basic statistics
    profile.update(self._analyze_basic_stats(series))

    # Type-specific analysis
    if np.issubdtype(series.dtype, np.number):
        profile.update(await self._analyze_numeric(series))
```

Analysis types:

- Statistical analysis
- Pattern detection
- Quality assessment
- Relationship inference

### 5. Error Handling

Comprehensive error handling system:

```python
@handle_errors
async def map_schemas(
    self,
    source_columns: List[Dict[str, Any]],
    target_columns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    try:
        # Implementation
    except Exception as e:
        raise SchemaMapperError(
            "Failed to map schemas",
            details={"error": str(e)}
        )
```

Features:

- Custom exceptions
- Error tracking
- Retry mechanism
- Fallback strategies

### 6. Caching

Multi-layer caching system:

```python
@cache_result(expire=3600)
async def generate_column_embedding(
    self,
    column_info: Dict[str, Any]
) -> Dict[str, Any]:
    # Implementation
```

Components:

- Redis primary cache
- In-memory fallback
- TTL-based expiration
- Cache monitoring

## Technical Details

### 1. Embedding Generation

The process of generating embeddings involves:

1. **Text Generation**:

   ```python
   def _build_column_text(
       self,
       column_name: str,
       data_type: str,
       sample_values: List[str]
   ) -> str:
       components = []

       # Add field description
       description = get_field_description(column_name)
       components.append(description)

       # Add contextual description
       context = generate_contextual_description(
           column_name,
           data_type,
           sample_values
       )
       components.append(context)
   ```

2. **Embedding Creation**:

   ```python
   embedding = self.model.encode(
       column_text,
       convert_to_tensor=False
   )
   ```

3. **Metadata Addition**:
   ```python
   metadata = {
       "name": column_name,
       "data_type": data_type,
       "compatible_types": json.dumps(
           get_compatible_types(data_type)
       )
   }
   ```

### 2. Mapping Process

The mapping process follows these steps:

1. **Initial Search**:

   ```python
   matches = await self.find_similar_columns(
       query_embedding,
       n_results=5,
       score_threshold=0.7
   )
   ```

2. **Rule Application**:

   ```python
   rule_results = apply_mapping_rules(
       source_column,
       mapping["matches"]
   )
   ```

3. **Confidence Scoring**:

   ```python
   confidence_score = calculate_confidence_score(
       rule_results,
       mapping.get("semantic_relationships", [])
   )
   ```

4. **LLM Analysis** (if needed):
   ```python
   if confidence_level == "low":
       llm_result = await self._handle_low_confidence(
           source_column,
           mapping["matches"],
           rule_results
       )
   ```

### 3. Data Analysis

The profiler performs multiple analyses:

1. **Statistical Analysis**:

   ```python
   async def _analyze_numeric(
       self,
       series: pd.Series
   ) -> Dict[str, Any]:
       stats = {
           "min": float(series.min()),
           "max": float(series.max()),
           "mean": float(series.mean()),
           "median": float(series.median()),
           "std": float(series.std())
       }
   ```

2. **Pattern Detection**:

   ```python
   async def _detect_value_patterns(
       self,
       series: pd.Series
   ) -> Dict[str, Any]:
       patterns = {
           "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
           "phone": r"^\+?1?\d{9,15}$",
           "date": r"^\d{4}-\d{2}-\d{2}$"
       }
   ```

3. **Quality Assessment**:
   ```python
   async def _assess_data_quality(
       self,
       series: pd.Series
   ) -> Dict[str, Any]:
       quality = {
           "completeness": 1 - (series.isnull().sum() / len(series)),
           "validity": self._check_validity(series),
           "consistency": self._check_consistency(series)
       }
   ```

### 4. Error Recovery

The fallback service provides multiple recovery strategies:

1. **Rule-Based Matching**:

   ```python
   async def _apply_rule_based_matching(
       self,
       column_name: str,
       data_type: str,
       sample_values: List[str]
   ) -> Dict[str, Any]:
       confidence = 0.0
       rules_matched = []

       # Name-based rules
       name_patterns = {
           "id": r".*_id$",
           "email": r".*_?email.*",
           "date": r".*_?(date|dt)$"
       }
   ```

2. **Pattern Matching**:

   ```python
   async def _apply_pattern_matching(
       self,
       column_name: str,
       sample_values: List[str]
   ) -> Dict[str, Any]:
       confidence = 0.0
       patterns_matched = []

       # Value patterns
       value_patterns = {
           "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
           "date_iso": r"^\d{4}-\d{2}-\d{2}$"
       }
   ```

3. **Semantic Relations**:
   ```python
   async def _check_semantic_relation(
       self,
       source_column: Dict[str, Any],
       target_column: Dict[str, Any]
   ) -> Dict[str, Any]:
       confidence = 0.0
       relations = []

       # Name similarity
       similarity = SequenceMatcher(
           None,
           source_column["name"],
           target_column["name"]
       ).ratio()
   ```

### 5. Performance Optimization

Key optimization strategies:

1. **Caching**:

   ```python
   class CacheManager:
       def __init__(self):
           self.redis_client = aioredis.Redis()
           self.memory_cache = TTLCache(
               maxsize=1000,
               ttl=3600
           )
   ```

2. **Batch Processing**:

   ```python
   async def build_vector_index(
       self,
       columns: List[Dict[str, Any]]
   ) -> None:
       embeddings = []
       metadatas = []
       documents = []

       for column in columns:
           result = await self.generate_column_embedding(
               column
           )
           embeddings.append(result["embedding"])
           metadatas.append(result["metadata"])
           documents.append(result["text"])
   ```

3. **Connection Pooling**:
   ```python
   class DatabaseService:
       def __init__(self):
           self.pool = aiosqlite.create_pool(
               database="schema_mapper.db",
               minsize=5,
               maxsize=20
           )
   ```

### 6. Monitoring

Monitoring components:

1. **Metrics Collection**:

   ```python
   class MetricsCollector:
       def __init__(self):
           self.mapping_latency = Histogram()
           self.cache_hits = Counter()
           self.error_rate = Gauge()
   ```

2. **Health Checks**:

   ```python
   class HealthCheck:
       async def check_services(self):
           return {
               "embedding": await self._check_embedding(),
               "vector_store": await self._check_vectors(),
               "llm": await self._check_llm(),
               "cache": await self._check_cache()
           }
   ```

3. **Performance Tracking**:
   ```python
   class PerformanceTracker:
       async def track_operation(
           self,
           operation_name: str,
           start_time: float
       ):
           duration = time.time() - start_time
           self.metrics.observe(
               operation_name,
               duration
           )
   ```
