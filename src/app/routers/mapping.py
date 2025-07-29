"""
API routes for schema mapping.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime

from src.app.models.models import (
    SchemaMappingRequest,
    SchemaMappingResponse,
    MetadataMapping
)
from src.app.services.schema.schema_mapper import SchemaMapper
from src.app.database.database import Database

router = APIRouter(prefix="/mapping", tags=["mapping"])

@router.post("/map-schemas", response_model=SchemaMappingResponse)
async def map_schemas(
    request: SchemaMappingRequest,
    mapper: SchemaMapper = Depends(SchemaMapper),
    db: Database = Depends(Database)
) -> SchemaMappingResponse:
    """Map source schema to target schema."""
    try:
        # Map schemas
        result = await mapper.map_schemas(
            request.source_schema,
            request.target_schema,
            request.confidence_threshold
        )
        
        # Save high confidence mappings
        for mapping in result["mappings"]:
            if mapping.confidence_score >= request.confidence_threshold:
                metadata_mapping = MetadataMapping(
                    mapping_id=len(result["mappings"]),
                    source_id="default",
                    source_table=request.source_schema[0].name.split(".")[0],
                    source_column=mapping.source_column,
                    target_table=request.target_schema[0].name.split(".")[0],
                    target_column=mapping.target_column,
                    confidence_score=mapping.confidence_score,
                    mapping_type="automatic"
                )
                await db.save_metadata_mapping(metadata_mapping)
        
        return SchemaMappingResponse(
            mappings=result["mappings"],
            stats=result["stats"],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to map schemas: {str(e)}"
        )

@router.get("/metadata-mappings", response_model=List[MetadataMapping])
async def get_metadata_mappings(
    source_id: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
    db: Database = Depends(Database)
) -> List[MetadataMapping]:
    """Get metadata mappings."""
    try:
        return await db.get_metadata_mappings(
            source_id,
            confidence_threshold
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metadata mappings: {str(e)}"
        )

@router.delete("/metadata-mappings/{mapping_id}")
async def delete_metadata_mapping(
    mapping_id: int,
    db: Database = Depends(Database)
) -> None:
    """Delete a metadata mapping."""
    try:
        await db.delete_metadata_mapping(mapping_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete metadata mapping: {str(e)}"
        )

@router.get("/mapping-statistics")
async def get_mapping_statistics(
    db: Database = Depends(Database)
) -> dict:
    """Get mapping statistics."""
    try:
        mappings = await db.get_metadata_mappings()
        
        # Calculate statistics
        total_mappings = len(mappings)
        avg_confidence = sum(m.confidence_score for m in mappings) / total_mappings if total_mappings > 0 else 0
        
        confidence_levels = {
            "high": len([m for m in mappings if m.confidence_score >= 0.8]),
            "medium": len([m for m in mappings if 0.5 <= m.confidence_score < 0.8]),
            "low": len([m for m in mappings if m.confidence_score < 0.5])
        }
        
        return {
            "total_mappings": total_mappings,
            "average_confidence": avg_confidence,
            "confidence_levels": confidence_levels,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get mapping statistics: {str(e)}"
        ) 