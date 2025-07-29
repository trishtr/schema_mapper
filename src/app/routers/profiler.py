"""
Router for data profiling endpoints.
"""
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime

from src.app.services.profiler.profiler_service import ProfilerService
from src.app.models.models import ColumnProfile, ProfileRequest, ProfileResponse

router = APIRouter(
    prefix="/profiler",
    tags=["profiler"]
)

async def get_profiler() -> ProfilerService:
    """Dependency for profiler service."""
    return ProfilerService()

@router.post(
    "/profile",
    response_model=ProfileResponse,
    summary="Profile data columns",
    description="Generate detailed profile for data columns."
)
async def profile_data(
    request: ProfileRequest,
    profiler: ProfilerService = Depends(get_profiler)
) -> Dict[str, Any]:
    """Profile data columns."""
    try:
        profiles = []
        
        # Profile each column
        for column in request.columns:
            profile = await profiler.profile_column(
                column["name"],
                column["data_type"],
                column.get("sample_values", [])
            )
            profiles.append(profile)
        
        return {
            "profiles": profiles,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get(
    "/profile/{column_name}",
    response_model=ColumnProfile,
    summary="Get column profile",
    description="Get profile for a specific column."
)
async def get_column_profile(
    column_name: str,
    profiler: ProfilerService = Depends(get_profiler)
) -> Dict[str, Any]:
    """Get column profile."""
    try:
        # Get profile from cache
        profile = profiler.profile_cache.get(column_name)
        
        if not profile:
            raise HTTPException(
                status_code=404,
                detail=f"Profile for column {column_name} not found"
            )
        
        return profile
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post(
    "/analyze-relationships",
    summary="Analyze column relationships",
    description="Analyze relationships between columns."
)
async def analyze_relationships(
    columns: List[Dict[str, Any]],
    profiler: ProfilerService = Depends(get_profiler)
) -> Dict[str, Any]:
    """Analyze column relationships."""
    try:
        relationships = []
        
        # Analyze each column pair
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Get profiles
                profile1 = await profiler.profile_column(
                    col1["name"],
                    col1["data_type"],
                    col1.get("sample_values", [])
                )
                
                profile2 = await profiler.profile_column(
                    col2["name"],
                    col2["data_type"],
                    col2.get("sample_values", [])
                )
                
                # Detect relationship
                relationship = await profiler._detect_relationships(
                    col1["name"],
                    profile1["statistics"]
                )
                
                if relationship:
                    relationships.append({
                        "source_column": col1["name"],
                        "target_column": col2["name"],
                        "relationship_type": relationship["type"],
                        "confidence": relationship.get("confidence", 0.0)
                    })
        
        return {
            "relationships": relationships,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get(
    "/statistics",
    summary="Get profiling statistics",
    description="Get current profiling statistics and metrics."
)
async def get_profiling_statistics(
    profiler: ProfilerService = Depends(get_profiler)
) -> Dict[str, Any]:
    """Get profiling statistics."""
    try:
        return {
            "cached_profiles": len(profiler.profile_cache),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 