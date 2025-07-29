"""
Router for schema mapping endpoints.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime

from src.app.services.schema.schema_mapper import SchemaMapper
from src.app.services.profiler.profiler_service import ProfilerService
from src.app.models.models import (
    SchemaMapping,
    SchemaMappingRequest,
    SchemaMappingResponse,
    MetadataMapping
)

router = APIRouter(
    prefix="/schema",
    tags=["schema"]
)

async def get_schema_mapper() -> SchemaMapper:
    """Dependency for schema mapper service."""
    return SchemaMapper()

async def get_profiler() -> ProfilerService:
    """Dependency for profiler service."""
    return ProfilerService()

@router.post(
    "/map",
    response_model=SchemaMappingResponse,
    summary="Map source schema to target schema",
    description="Map source database schema to target schema using embeddings and rules."
)
async def map_schemas(
    request: SchemaMappingRequest,
    mapper: SchemaMapper = Depends(get_schema_mapper),
    profiler: ProfilerService = Depends(get_profiler)
) -> Dict[str, Any]:
    """Map source schema to target schema."""
    try:
        # Profile source columns
        source_profiles = []
        for column in request.source_columns:
            profile = await profiler.profile_column(
                column["name"],
                column["data_type"],
                column.get("sample_values", [])
            )
            source_profiles.append(profile)
        
        # Profile target columns
        target_profiles = []
        for column in request.target_columns:
            profile = await profiler.profile_column(
                column["name"],
                column["data_type"],
                column.get("sample_values", [])
            )
            target_profiles.append(profile)
        
        # Map schemas
        mappings = await mapper.map_schemas(
            request.source_columns,
            request.target_columns
        )
        
        # Get mapping statistics
        stats = mapper.get_mapping_statistics()
        
        return {
            "mappings": mappings,
            "statistics": stats,
            "source_profiles": source_profiles,
            "target_profiles": target_profiles,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get(
    "/metadata",
    response_model=List[MetadataMapping],
    summary="Get metadata mappings",
    description="Get all stored metadata mappings with optional filters."
)
async def get_metadata_mappings(
    source_table: str = None,
    target_table: str = None,
    confidence_level: str = None,
    mapper: SchemaMapper = Depends(get_schema_mapper)
) -> List[Dict[str, Any]]:
    """Get metadata mappings with filters."""
    try:
        # Get mappings from database
        mappings = []  # Implementation depends on database service
        
        # Apply filters
        if source_table:
            mappings = [
                m for m in mappings
                if m["source_table"] == source_table
            ]
        
        if target_table:
            mappings = [
                m for m in mappings
                if m["target_table"] == target_table
            ]
        
        if confidence_level:
            mappings = [
                m for m in mappings
                if m["confidence_level"] == confidence_level
            ]
        
        return mappings
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.delete(
    "/metadata/{mapping_id}",
    summary="Delete metadata mapping",
    description="Delete a specific metadata mapping by ID."
)
async def delete_metadata_mapping(
    mapping_id: str,
    mapper: SchemaMapper = Depends(get_schema_mapper)
) -> Dict[str, Any]:
    """Delete metadata mapping."""
    try:
        # Delete mapping from database
        success = True  # Implementation depends on database service
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Mapping {mapping_id} not found"
            )
        
        return {
            "status": "success",
            "message": f"Mapping {mapping_id} deleted",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get(
    "/statistics",
    summary="Get mapping statistics",
    description="Get current mapping statistics and metrics."
)
async def get_mapping_statistics(
    mapper: SchemaMapper = Depends(get_schema_mapper)
) -> Dict[str, Any]:
    """Get mapping statistics."""
    try:
        return mapper.get_mapping_statistics()
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 