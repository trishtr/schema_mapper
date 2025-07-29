"""
Main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.routers import mapping
from src.app.config import settings

app = FastAPI(
    title="Schema Mapper",
    description="Intelligent schema mapping using embeddings and LLM",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include routers
app.include_router(mapping.router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Schema Mapper API",
        "version": "0.1.0",
        "docs_url": "/docs",
        "endpoints": [
            "/mapping/map-schemas",
            "/mapping/metadata-mappings",
            "/mapping/mapping-statistics"
        ]
    } 