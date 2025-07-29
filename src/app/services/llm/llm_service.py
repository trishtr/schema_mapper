"""
LLM service for complex schema mapping analysis.
"""
from typing import Dict, Any, Optional
import logging
import aiohttp
import json
from datetime import datetime

from src.app.utils.error_handling import LLMError
from src.app.utils.caching import cache_result
from .config.prompts import (
    generate_mapping_prompt,
    generate_validation_prompt,
    parse_llm_response,
    validate_mapping_result
)

class LLMService:
    """Service for LLM-based schema mapping analysis."""
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.api_url = api_url or "http://localhost:8000/v1/completions"
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    @cache_result(expire=3600)
    async def analyze_mapping(
        self,
        source_column: Dict[str, Any],
        target_column: Dict[str, Any],
        current_mapping: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze column mapping using LLM."""
        try:
            # Generate prompt
            prompt = generate_mapping_prompt(
                source_column,
                target_column,
                current_mapping
            )
            
            # Get LLM response
            response = await self._call_llm(prompt)
            
            # Parse and validate response
            result = parse_llm_response(response)
            
            if not validate_mapping_result(result):
                raise ValueError("Invalid LLM response format")
            
            return result
            
        except Exception as e:
            logging.error(f"LLM analysis failed: {str(e)}")
            raise LLMError(
                "Failed to analyze mapping with LLM",
                details={"error": str(e)}
            )
    
    @cache_result(expire=3600)
    async def validate_mapping(
        self,
        mapping_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate mapping result using LLM."""
        try:
            # Generate prompt
            prompt = generate_validation_prompt(mapping_result)
            
            # Get LLM response
            response = await self._call_llm(prompt)
            
            # Parse and validate response
            result = parse_llm_response(response)
            
            if not validate_mapping_result(result):
                raise ValueError("Invalid LLM response format")
            
            return result
            
        except Exception as e:
            logging.error(f"LLM validation failed: {str(e)}")
            raise LLMError(
                "Failed to validate mapping with LLM",
                details={"error": str(e)}
            )
    
    async def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM service."""
        try:
            # Prepare request
            payload = {
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.3,
                "stop": ["```"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        raise LLMError(
                            f"LLM API returned status {response.status}",
                            details={"status": response.status}
                        )
                    
                    result = await response.json()
                    
                    if "error" in result:
                        raise LLMError(
                            "LLM API returned error",
                            details={"error": result["error"]}
                        )
                    
                    return result["choices"][0]["text"]
                    
        except aiohttp.ClientError as e:
            raise LLMError(
                "Failed to connect to LLM API",
                details={"error": str(e)}
            )
        except Exception as e:
            raise LLMError(
                "Unexpected error calling LLM API",
                details={"error": str(e)}
            )
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get LLM service status."""
        try:
            # Simple health check
            response = await self._call_llm("Return 'ok' as JSON")
            result = parse_llm_response(response)
            
            return {
                "status": "healthy",
                "latency_ms": result.get("latency", 0),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _mock_llm_response(self, prompt: str) -> str:
        """Generate mock LLM response for testing."""
        return json.dumps({
            "is_valid": True,
            "confidence": 0.85,
            "explanation": "Mock LLM response for testing",
            "transformations": ["no_transformation_needed"]
        }) 