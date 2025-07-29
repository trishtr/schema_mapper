"""
Data models for schema mapping.
"""
from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ColumnInfo(BaseModel):
    """Information about a database column."""
    name: str
    data_type: str
    description: Optional[str] = None
    sample_values: Optional[List[str]] = None
    constraints: Optional[Dict[str, Any]] = None
    business_rules: Optional[Dict[str, Any]] = None

class ColumnEmbedding(BaseModel):
    """Column embedding with metadata."""
    column_name: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SchemaMapping(BaseModel):
    """Schema mapping result."""
    source_column: str
    target_column: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    compatible_types: List[str]
    transformations: Optional[List[str]] = None
    validation_rules: Optional[List[str]] = None
    semantic_relationships: Optional[Dict[str, List[str]]] = None
    llm_analysis: Optional[str] = None

class SchemaMappingRequest(BaseModel):
    """Request for schema mapping."""
    source_schema: List[ColumnInfo]
    target_schema: List[ColumnInfo]
    confidence_threshold: Optional[float] = 0.8
    include_analysis: Optional[bool] = False

class SchemaMappingResponse(BaseModel):
    """Response for schema mapping."""
    mappings: List[SchemaMapping]
    stats: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class MetadataMapping(BaseModel):
    """Metadata mapping record."""
    mapping_id: int
    source_id: str
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    confidence_score: float
    mapping_type: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class LLMRequest(BaseModel):
    """Request for LLM analysis."""
    source_column: ColumnInfo
    target_column: ColumnInfo
    current_mapping: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class LLMResponse(BaseModel):
    """Response from LLM analysis."""
    is_valid: bool
    confidence: float
    explanation: str
    transformations: Optional[List[str]] = None
    warnings: Optional[List[str]] = None 