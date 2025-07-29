"""
Caching utilities and management.
"""
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from functools import wraps
import logging
from datetime import datetime, timedelta
import json
import aioredis
from cachetools import TTLCache

T = TypeVar('T')

class CacheManager:
    """Multi-layer cache manager."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        memory_size: int = 1000,
        memory_ttl: int = 3600
    ):
        # Redis client
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = aioredis.from_url(redis_url)
            except Exception as e:
                logging.warning(
                    f"Failed to initialize Redis: {str(e)}"
                )
        
        # In-memory cache
        self.memory_cache = TTLCache(
            maxsize=memory_size,
            ttl=memory_ttl
        )
        
        # Cache statistics
        self.stats = {
            "redis_hits": 0,
            "redis_misses": 0,
            "memory_hits": 0,
            "memory_misses": 0
        }
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Get value from cache."""
        try:
            # Try Redis first
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    self.stats["redis_hits"] += 1
                    return json.loads(value)
                self.stats["redis_misses"] += 1
            
            # Try memory cache
            if key in self.memory_cache:
                self.stats["memory_hits"] += 1
                return self.memory_cache[key]
            self.stats["memory_misses"] += 1
            
            return default
            
        except Exception as e:
            logging.error(f"Cache get failed: {str(e)}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """Set value in cache."""
        try:
            # Set in Redis
            if self.redis_client:
                await self.redis_client.set(
                    key,
                    json.dumps(value),
                    ex=expire
                )
            
            # Set in memory cache
            self.memory_cache[key] = value
            return True
            
        except Exception as e:
            logging.error(f"Cache set failed: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            # Delete from Redis
            if self.redis_client:
                await self.redis_client.delete(key)
            
            # Delete from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
            
            return True
            
        except Exception as e:
            logging.error(f"Cache delete failed: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear all caches."""
        try:
            # Clear Redis
            if self.redis_client:
                await self.redis_client.flushdb()
            
            # Clear memory cache
            self.memory_cache.clear()
            
            # Reset statistics
            self.stats = {
                "redis_hits": 0,
                "redis_misses": 0,
                "memory_hits": 0,
                "memory_misses": 0
            }
            
            return True
            
        except Exception as e:
            logging.error(f"Cache clear failed: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            "memory_size": len(self.memory_cache),
            "timestamp": datetime.utcnow().isoformat()
        }

class SchemaCache:
    """Specialized cache for schema mapping."""
    
    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None
    ):
        self.cache = cache_manager or CacheManager()
    
    async def get_embedding(
        self,
        column_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get column embedding from cache."""
        key = f"embedding:{column_name}"
        return await self.cache.get(key)
    
    async def set_embedding(
        self,
        column_name: str,
        embedding: Dict[str, Any],
        expire: int = 86400  # 24 hours
    ) -> bool:
        """Set column embedding in cache."""
        key = f"embedding:{column_name}"
        return await self.cache.set(key, embedding, expire)
    
    async def get_mapping(
        self,
        source_column: str,
        target_column: str
    ) -> Optional[Dict[str, Any]]:
        """Get schema mapping from cache."""
        key = f"mapping:{source_column}:{target_column}"
        return await self.cache.get(key)
    
    async def set_mapping(
        self,
        source_column: str,
        target_column: str,
        mapping: Dict[str, Any],
        expire: int = 3600  # 1 hour
    ) -> bool:
        """Set schema mapping in cache."""
        key = f"mapping:{source_column}:{target_column}"
        return await self.cache.set(key, mapping, expire)
    
    async def get_profile(
        self,
        column_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get column profile from cache."""
        key = f"profile:{column_name}"
        return await self.cache.get(key)
    
    async def set_profile(
        self,
        column_name: str,
        profile: Dict[str, Any],
        expire: int = 43200  # 12 hours
    ) -> bool:
        """Set column profile in cache."""
        key = f"profile:{column_name}"
        return await self.cache.set(key, profile, expire)

def cache_result(
    expire: Optional[int] = None,
    key_prefix: Optional[str] = None
):
    """Decorator for caching function results."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            cache_key = _generate_cache_key(
                func.__name__,
                key_prefix,
                args,
                kwargs
            )
            
            # Get cache manager
            cache = CacheManager()
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, expire)
            
            return result
        return wrapper
    return decorator

def _generate_cache_key(
    func_name: str,
    prefix: Optional[str],
    args: tuple,
    kwargs: dict
) -> str:
    """Generate cache key from function arguments."""
    key_parts = [prefix] if prefix else []
    key_parts.append(func_name)
    
    # Add args
    if args:
        key_parts.extend(str(arg) for arg in args)
    
    # Add kwargs
    if kwargs:
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")
    
    return ":".join(key_parts)

class CacheConfig:
    """Configuration for caching."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        memory_size: int = 1000,
        memory_ttl: int = 3600,
        default_expire: int = 300
    ):
        self.redis_url = redis_url
        self.memory_size = memory_size
        self.memory_ttl = memory_ttl
        self.default_expire = default_expire
        
        # TTL configuration
        self.ttl_config = {
            "embedding": 86400,    # 24 hours
            "mapping": 3600,       # 1 hour
            "profile": 43200,      # 12 hours
            "metadata": 300        # 5 minutes
        }
    
    def get_ttl(
        self,
        cache_type: str
    ) -> int:
        """Get TTL for cache type."""
        return self.ttl_config.get(
            cache_type,
            self.default_expire
        )

class CacheMonitor:
    """Monitor for cache operations."""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
        self.operations = []
    
    def record_operation(
        self,
        operation_type: str,
        key: str,
        success: bool,
        latency: float
    ) -> None:
        """Record cache operation."""
        self.operations.append({
            "type": operation_type,
            "key": key,
            "success": success,
            "latency": latency,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache monitoring statistics."""
        if not self.operations:
            return {
                "operations": 0,
                "success_rate": 0.0,
                "avg_latency": 0.0
            }
        
        total_ops = len(self.operations)
        success_ops = sum(
            1 for op in self.operations
            if op["success"]
        )
        total_latency = sum(
            op["latency"]
            for op in self.operations
        )
        
        return {
            "operations": total_ops,
            "success_rate": success_ops / total_ops,
            "avg_latency": total_latency / total_ops,
            "uptime": (
                datetime.utcnow() - self.start_time
            ).total_seconds()
        } 