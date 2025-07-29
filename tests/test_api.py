"""
Tests for API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime

from src.app.main import app
from src.app.models.models import (
    ColumnInfo,
    SchemaMappingRequest,
    MetadataMapping
)

client = TestClient(app)

@pytest.fixture
def sample_mapping_request():
    """Sample schema mapping request."""
    return SchemaMappingRequest(
        source_schema=[
            ColumnInfo(
                name="patient_id",
                data_type="VARCHAR",
                sample_values=["P123", "P456"]
            ),
            ColumnInfo(
                name="email_address",
                data_type="VARCHAR",
                sample_values=["john@example.com"]
            )
        ],
        target_schema=[
            ColumnInfo(
                name="id",
                data_type="VARCHAR",
                sample_values=["PAT123"]
            ),
            ColumnInfo(
                name="email",
                data_type="VARCHAR",
                sample_values=["user@example.com"]
            )
        ],
        confidence_threshold=0.8
    )

@pytest.fixture
def sample_metadata_mapping():
    """Sample metadata mapping."""
    return MetadataMapping(
        mapping_id=1,
        source_id="test_source",
        source_table="patients",
        source_column="patient_id",
        target_table="users",
        target_column="id",
        confidence_score=0.95,
        mapping_type="automatic",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Schema Mapper API"

def test_map_schemas(sample_mapping_request):
    """Test schema mapping endpoint."""
    response = client.post(
        "/mapping/map-schemas",
        json=sample_mapping_request.dict()
    )
    assert response.status_code == 200
    data = response.json()
    assert "mappings" in data
    assert "stats" in data
    assert "timestamp" in data

def test_get_metadata_mappings(sample_metadata_mapping):
    """Test get metadata mappings endpoint."""
    response = client.get("/mapping/metadata-mappings")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_metadata_mappings_with_filters(sample_metadata_mapping):
    """Test get metadata mappings with filters."""
    response = client.get(
        "/mapping/metadata-mappings",
        params={
            "source_id": "test_source",
            "confidence_threshold": 0.8
        }
    )
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_delete_metadata_mapping(sample_metadata_mapping):
    """Test delete metadata mapping endpoint."""
    response = client.delete(f"/mapping/metadata-mappings/{sample_metadata_mapping.mapping_id}")
    assert response.status_code == 200

def test_get_mapping_statistics():
    """Test get mapping statistics endpoint."""
    response = client.get("/mapping/mapping-statistics")
    assert response.status_code == 200
    data = response.json()
    assert "total_mappings" in data
    assert "average_confidence" in data
    assert "confidence_levels" in data
    assert "timestamp" in data 