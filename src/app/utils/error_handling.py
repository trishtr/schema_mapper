"""
Error handling utilities and custom exceptions.
"""
from typing import Dict, Any, Optional, Callable, Type
from functools import wraps
import logging
from datetime import datetime
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    RetryError
)

# Base exception
class SchemaMapperError(Exception):
    """Base exception for schema mapper."""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(self.message)

# Service-specific exceptions
class EmbeddingError(SchemaMapperError):
    """Exception for embedding service failures."""
    pass

class VectorStoreError(SchemaMapperError):
    """Exception for vector store failures."""
    pass

class LLMError(SchemaMapperError):
    """Exception for LLM service failures."""
    pass

class DatabaseError(SchemaMapperError):
    """Exception for database failures."""
    pass

class ValidationError(SchemaMapperError):
    """Exception for validation failures."""
    pass

class CacheError(SchemaMapperError):
    """Exception for cache failures."""
    pass

def handle_errors(
    error_types: Optional[Dict[Type[Exception], str]] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
):
    """Decorator for error handling with retries."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            error_map = error_types or {
                EmbeddingError: "Embedding service error",
                VectorStoreError: "Vector store error",
                LLMError: "LLM service error",
                DatabaseError: "Database error",
                ValidationError: "Validation error",
                CacheError: "Cache error"
            }
            
            try:
                # Retry logic
                @retry(
                    stop=stop_after_attempt(max_retries),
                    wait=wait_exponential(
                        multiplier=retry_delay,
                        min=retry_delay,
                        max=retry_delay * 4
                    )
                )
                async def _execute():
                    return await func(*args, **kwargs)
                
                return await _execute()
                
            except RetryError as e:
                # Log retry failure
                logging.error(
                    f"Operation failed after {max_retries} retries: {str(e)}"
                )
                raise
                
            except tuple(error_map.keys()) as e:
                # Handle known errors
                error_type = type(e).__name__
                error_message = error_map.get(type(e), "Unknown error")
                
                logging.error(
                    f"{error_type}: {error_message}",
                    extra={
                        "error_type": error_type,
                        "details": e.details,
                        "timestamp": e.timestamp
                    }
                )
                
                raise
                
            except Exception as e:
                # Handle unexpected errors
                logging.error(
                    f"Unexpected error: {str(e)}",
                    exc_info=True
                )
                
                raise SchemaMapperError(
                    "An unexpected error occurred",
                    details={"error": str(e)}
                )
        
        return wrapper
    return decorator

class ValidationUtils:
    """Utilities for data validation."""
    
    @staticmethod
    def validate_column_info(column_info: Dict[str, Any]) -> bool:
        """Validate column information."""
        required_fields = ["name", "data_type"]
        return all(field in column_info for field in required_fields)
    
    @staticmethod
    def validate_mapping_result(
        mapping: Dict[str, Any]
    ) -> bool:
        """Validate mapping result."""
        required_fields = [
            "source_column",
            "target_column",
            "confidence_level"
        ]
        return all(field in mapping for field in required_fields)
    
    @staticmethod
    def validate_profile_result(
        profile: Dict[str, Any]
    ) -> bool:
        """Validate profile result."""
        required_fields = [
            "name",
            "data_type",
            "statistics",
            "quality"
        ]
        return all(field in profile for field in required_fields)

class OperationMonitor:
    """Monitor for operation tracking."""
    
    def __init__(self):
        self.operations = {}
    
    async def start_operation(
        self,
        operation_id: str,
        operation_type: str
    ) -> None:
        """Start tracking operation."""
        self.operations[operation_id] = {
            "type": operation_type,
            "start_time": datetime.utcnow().isoformat(),
            "status": "in_progress"
        }
    
    async def end_operation(
        self,
        operation_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """End tracking operation."""
        if operation_id in self.operations:
            self.operations[operation_id].update({
                "end_time": datetime.utcnow().isoformat(),
                "status": status,
                "result": result
            })
    
    def get_operation_status(
        self,
        operation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get operation status."""
        return self.operations.get(operation_id)

class HealthCheck:
    """Health check utilities."""
    
    @staticmethod
    async def check_service(
        service_name: str,
        check_func: Callable
    ) -> Dict[str, Any]:
        """Check service health."""
        try:
            start_time = datetime.utcnow()
            await check_func()
            latency = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                "status": "healthy",
                "latency": latency,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @staticmethod
    async def check_all_services(
        services: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """Check all services health."""
        results = {}
        
        for name, check_func in services.items():
            results[name] = await HealthCheck.check_service(
                name,
                check_func
            )
        
        return {
            "services": results,
            "timestamp": datetime.utcnow().isoformat()
        }

class ErrorResponseBuilder:
    """Builder for error responses."""
    
    @staticmethod
    def build_error_response(
        error: Exception,
        status_code: int = 500
    ) -> Dict[str, Any]:
        """Build error response."""
        if isinstance(error, SchemaMapperError):
            return {
                "error": {
                    "type": type(error).__name__,
                    "message": error.message,
                    "details": error.details,
                    "timestamp": error.timestamp
                },
                "status_code": status_code
            }
        
        return {
            "error": {
                "type": "UnexpectedError",
                "message": str(error),
                "timestamp": datetime.utcnow().isoformat()
            },
            "status_code": status_code
        }
    
    @staticmethod
    def build_validation_error(
        field: str,
        message: str
    ) -> Dict[str, Any]:
        """Build validation error response."""
        return {
            "error": {
                "type": "ValidationError",
                "field": field,
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            },
            "status_code": 400
        } 