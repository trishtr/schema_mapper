"""
Service for handling failures with fallback strategies.
"""
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import re
from difflib import SequenceMatcher

from src.app.utils.error_handling import FallbackError
from src.app.utils.caching import CacheManager

class FallbackService:
    """Service for managing fallback strategies."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        self.retry_queue = []
        self.fallback_stats = {
            "total_failures": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0
        }
    
    async def handle_failure(
        self,
        service_name: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle service failure with appropriate fallback."""
        try:
            self.fallback_stats["total_failures"] += 1
            
            # Log failure
            logging.error(
                f"Service {service_name} failed: {str(error)}"
            )
            
            # Choose fallback strategy
            if service_name == "embedding":
                result = await self._fallback_embedding(context)
            elif service_name == "vector_store":
                result = await self._fallback_vector_store(context)
            elif service_name == "llm":
                result = await self._fallback_llm(context)
            elif service_name == "database":
                result = await self._fallback_database(context)
            else:
                raise ValueError(f"Unknown service: {service_name}")
            
            if result["success"]:
                self.fallback_stats["successful_fallbacks"] += 1
            else:
                self.fallback_stats["failed_fallbacks"] += 1
                
                # Queue for retry if appropriate
                if result.get("should_retry", False):
                    await self._queue_for_retry(
                        service_name,
                        context
                    )
            
            return result
            
        except Exception as e:
            logging.error(f"Fallback handling failed: {str(e)}")
            raise FallbackError(
                "Failed to handle service failure",
                details={"error": str(e)}
            )
    
    async def _fallback_embedding(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle embedding service failure."""
        try:
            # Try cache first
            cache_key = f"embedding_{context['column_name']}"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return {
                    "success": True,
                    "source": "cache",
                    "result": cached
                }
            
            # Apply rule-based matching
            rule_result = await self._apply_rule_based_matching(
                context["column_name"],
                context["data_type"],
                context.get("sample_values", [])
            )
            
            if rule_result["confidence"] >= 0.6:
                return {
                    "success": True,
                    "source": "rules",
                    "result": rule_result
                }
            
            # Try pattern matching
            pattern_result = await self._apply_pattern_matching(
                context["column_name"],
                context.get("sample_values", [])
            )
            
            if pattern_result["confidence"] >= 0.6:
                return {
                    "success": True,
                    "source": "patterns",
                    "result": pattern_result
                }
            
            return {
                "success": False,
                "should_retry": True,
                "error": "No fallback strategies succeeded"
            }
            
        except Exception as e:
            logging.error(f"Embedding fallback failed: {str(e)}")
            return {
                "success": False,
                "should_retry": False,
                "error": str(e)
            }
    
    async def _fallback_vector_store(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle vector store failure."""
        try:
            # Try cache first
            cache_key = f"vectors_{context['query_id']}"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return {
                    "success": True,
                    "source": "cache",
                    "result": cached
                }
            
            # Use in-memory fallback
            if "embeddings" in context:
                matches = []
                query_embedding = context["query_embedding"]
                
                for idx, embedding in enumerate(context["embeddings"]):
                    similarity = self._calculate_cosine_similarity(
                        query_embedding,
                        embedding
                    )
                    if similarity >= 0.7:
                        matches.append({
                            "id": idx,
                            "similarity": similarity
                        })
                
                if matches:
                    return {
                        "success": True,
                        "source": "memory",
                        "result": {
                            "matches": matches
                        }
                    }
            
            return {
                "success": False,
                "should_retry": True,
                "error": "No fallback vectors available"
            }
            
        except Exception as e:
            logging.error(f"Vector store fallback failed: {str(e)}")
            return {
                "success": False,
                "should_retry": False,
                "error": str(e)
            }
    
    async def _fallback_llm(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle LLM service failure."""
        try:
            # Try cache first
            cache_key = f"llm_{context['prompt_hash']}"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return {
                    "success": True,
                    "source": "cache",
                    "result": cached
                }
            
            # Check semantic relation
            relation_result = await self._check_semantic_relation(
                context["source_column"],
                context["target_column"]
            )
            
            if relation_result["confidence"] >= 0.6:
                return {
                    "success": True,
                    "source": "semantic",
                    "result": relation_result
                }
            
            return {
                "success": False,
                "should_retry": True,
                "error": "No fallback analysis available"
            }
            
        except Exception as e:
            logging.error(f"LLM fallback failed: {str(e)}")
            return {
                "success": False,
                "should_retry": False,
                "error": str(e)
            }
    
    async def _fallback_database(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle database service failure."""
        try:
            # Try cache first
            cache_key = f"db_{context['query_hash']}"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return {
                    "success": True,
                    "source": "cache",
                    "result": cached
                }
            
            # Use backup if available
            if "backup_connection" in context:
                # Implementation depends on database service
                return {
                    "success": True,
                    "source": "backup",
                    "result": {"status": "using_backup"}
                }
            
            return {
                "success": False,
                "should_retry": True,
                "error": "No fallback database available"
            }
            
        except Exception as e:
            logging.error(f"Database fallback failed: {str(e)}")
            return {
                "success": False,
                "should_retry": False,
                "error": str(e)
            }
    
    async def _apply_rule_based_matching(
        self,
        column_name: str,
        data_type: str,
        sample_values: List[str]
    ) -> Dict[str, Any]:
        """Apply rule-based matching."""
        confidence = 0.0
        rules_matched = []
        
        # Name-based rules
        name_patterns = {
            "id": r".*_id$",
            "email": r".*_?email.*",
            "phone": r".*_?(phone|tel|mobile).*",
            "date": r".*_?(date|dt)$",
            "name": r".*_?name$",
            "code": r".*_?code$"
        }
        
        for rule, pattern in name_patterns.items():
            if re.match(pattern, column_name.lower()):
                confidence += 0.2
                rules_matched.append(f"name_pattern_{rule}")
        
        # Type-based rules
        type_patterns = {
            "INT": r".*INT.*",
            "VARCHAR": r".*VAR.*CHAR.*",
            "DATE": r".*DATE.*",
            "BOOL": r".*BOOL.*"
        }
        
        for rule, pattern in type_patterns.items():
            if re.match(pattern, data_type.upper()):
                confidence += 0.2
                rules_matched.append(f"type_pattern_{rule}")
        
        # Value-based rules
        if sample_values:
            value = str(sample_values[0])
            if re.match(r"^\d+$", value):
                confidence += 0.2
                rules_matched.append("numeric_values")
            elif re.match(r"^[A-Za-z\s]+$", value):
                confidence += 0.2
                rules_matched.append("text_values")
            elif re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                confidence += 0.2
                rules_matched.append("date_values")
        
        return {
            "confidence": min(confidence, 1.0),
            "rules_matched": rules_matched
        }
    
    async def _apply_pattern_matching(
        self,
        column_name: str,
        sample_values: List[str]
    ) -> Dict[str, Any]:
        """Apply pattern-based matching."""
        confidence = 0.0
        patterns_matched = []
        
        # Common patterns
        value_patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?1?\d{9,15}$",
            "date_iso": r"^\d{4}-\d{2}-\d{2}$",
            "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        }
        
        if sample_values:
            for pattern_name, pattern in value_patterns.items():
                matches = [
                    bool(re.match(pattern, str(v)))
                    for v in sample_values if v
                ]
                if matches and all(matches):
                    confidence += 0.3
                    patterns_matched.append(pattern_name)
        
        # Name patterns
        name_parts = re.split(r"[_\s]", column_name.lower())
        common_terms = {
            "identifier": ["id", "code", "key"],
            "temporal": ["date", "time", "timestamp"],
            "contact": ["email", "phone", "address"],
            "person": ["name", "user", "customer"]
        }
        
        for category, terms in common_terms.items():
            if any(part in terms for part in name_parts):
                confidence += 0.2
                patterns_matched.append(f"name_{category}")
        
        return {
            "confidence": min(confidence, 1.0),
            "patterns_matched": patterns_matched
        }
    
    async def _check_semantic_relation(
        self,
        source_column: Dict[str, Any],
        target_column: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check semantic relationships."""
        confidence = 0.0
        relations = []
        
        # Name similarity
        source_name = source_column["name"].lower()
        target_name = target_column["name"].lower()
        
        similarity = SequenceMatcher(
            None,
            source_name,
            target_name
        ).ratio()
        
        if similarity >= 0.8:
            confidence += 0.3
            relations.append("high_name_similarity")
        elif similarity >= 0.6:
            confidence += 0.2
            relations.append("medium_name_similarity")
        
        # Type compatibility
        if source_column["data_type"] == target_column["data_type"]:
            confidence += 0.3
            relations.append("exact_type_match")
        
        # Value patterns
        if (
            "sample_values" in source_column and
            "sample_values" in target_column
        ):
            source_values = source_column["sample_values"]
            target_values = target_column["sample_values"]
            
            if source_values and target_values:
                # Format similarity
                source_format = self._detect_format(str(source_values[0]))
                target_format = self._detect_format(str(target_values[0]))
                
                if source_format == target_format:
                    confidence += 0.2
                    relations.append("matching_format")
        
        return {
            "confidence": min(confidence, 1.0),
            "relations": relations
        }
    
    async def _queue_for_retry(
        self,
        service_name: str,
        context: Dict[str, Any]
    ) -> None:
        """Queue failed operation for retry."""
        self.retry_queue.append({
            "service": service_name,
            "context": context,
            "attempts": 0,
            "next_retry": datetime.utcnow() + timedelta(minutes=5),
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def _calculate_cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return dot_product / (norm1 * norm2)
    
    def _detect_format(self, value: str) -> str:
        """Detect format of a value."""
        if re.match(r"^\d+$", value):
            return "numeric"
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", value):
            return "date_iso"
        elif re.match(r"^[A-Za-z\s]+$", value):
            return "text"
        elif re.match(r"^[A-Za-z0-9\s\-_]+$", value):
            return "alphanumeric"
        else:
            return "unknown"
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get current fallback statistics."""
        return {
            **self.fallback_stats,
            "retry_queue_size": len(self.retry_queue),
            "timestamp": datetime.utcnow().isoformat()
        } 