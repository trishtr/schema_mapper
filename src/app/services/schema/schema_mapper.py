"""
Schema mapping orchestration service.
"""
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from src.app.utils.error_handling import SchemaMapperError
from src.app.utils.caching import cache_result
from src.app.services.embedding.embedding_service import EmbeddingService
from src.app.services.llm.llm_service import LLMService
from .config.mapping_rules import (
    apply_mapping_rules,
    calculate_confidence_score,
    validate_mapping
)

class SchemaMapper:
    """Service for orchestrating schema mapping."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service or LLMService()
        self.mapping_stats = {
            "total_mappings": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "llm_analyzed": 0
        }
    
    @cache_result(expire=3600)
    async def map_schemas(
        self,
        source_columns: List[Dict[str, Any]],
        target_columns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Map source columns to target columns."""
        try:
            # Get initial mappings from embeddings
            initial_mappings = await self.embedding_service.map_schemas_with_embeddings(
                source_columns,
                target_columns
            )
            
            final_mappings = []
            
            # Process each mapping
            for mapping in initial_mappings:
                processed_mapping = await self._process_column_mapping(
                    mapping,
                    source_columns,
                    target_columns
                )
                
                if processed_mapping:
                    final_mappings.append(processed_mapping)
                    
                    # Update statistics
                    self.mapping_stats["total_mappings"] += 1
                    confidence = processed_mapping["confidence_level"]
                    if confidence == "high":
                        self.mapping_stats["high_confidence"] += 1
                    elif confidence == "medium":
                        self.mapping_stats["medium_confidence"] += 1
                    else:
                        self.mapping_stats["low_confidence"] += 1
            
            return final_mappings
            
        except Exception as e:
            logging.error(f"Schema mapping failed: {str(e)}")
            raise SchemaMapperError(
                "Failed to map schemas",
                details={"error": str(e)}
            )
    
    async def _process_column_mapping(
        self,
        mapping: Dict[str, Any],
        source_columns: List[Dict[str, Any]],
        target_columns: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Process a single column mapping."""
        try:
            source_column = next(
                col for col in source_columns
                if col["name"] == mapping["source_column"]
            )
            
            # Apply mapping rules
            rule_results = apply_mapping_rules(
                source_column,
                mapping["matches"]
            )
            
            # Calculate confidence
            confidence_score = calculate_confidence_score(
                rule_results,
                mapping.get("semantic_relationships", [])
            )
            
            # Determine confidence level
            if confidence_score >= 0.8:
                confidence_level = "high"
            elif confidence_score >= 0.6:
                confidence_level = "medium"
            else:
                confidence_level = "low"
            
            # For low confidence, use LLM
            if confidence_level == "low":
                llm_result = await self._handle_low_confidence(
                    source_column,
                    mapping["matches"],
                    rule_results
                )
                
                if llm_result:
                    confidence_level = llm_result["confidence_level"]
                    rule_results.update(llm_result["additional_rules"])
                    self.mapping_stats["llm_analyzed"] += 1
            
            # Validate final mapping
            if not validate_mapping(rule_results):
                return None
            
            return {
                "source_column": mapping["source_column"],
                "target_column": mapping["matches"][0]["column_name"],
                "confidence_level": confidence_level,
                "confidence_score": confidence_score,
                "rule_results": rule_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logging.error(
                f"Column mapping processing failed: {str(e)}"
            )
            return None
    
    async def _handle_low_confidence(
        self,
        source_column: Dict[str, Any],
        matches: List[Dict[str, Any]],
        rule_results: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle low confidence mappings with LLM."""
        try:
            if not matches:
                return None
            
            # Get LLM analysis
            llm_result = await self.llm_service.analyze_mapping(
                source_column,
                matches[0],
                {
                    "rule_results": rule_results,
                    "matches": matches
                }
            )
            
            # Validate LLM result
            llm_validation = await self.llm_service.validate_mapping(
                llm_result
            )
            
            if not llm_validation["is_valid"]:
                return None
            
            return {
                "confidence_level": "medium" if llm_result["confidence"] >= 0.7 else "low",
                "additional_rules": llm_result.get("transformations", {})
            }
            
        except Exception as e:
            logging.error(f"LLM analysis failed: {str(e)}")
            return None
    
    async def _save_high_confidence_mapping(
        self,
        mapping: Dict[str, Any]
    ) -> bool:
        """Save high confidence mapping to metadata."""
        try:
            # Save to database
            # Implementation depends on database service
            return True
            
        except Exception as e:
            logging.error(
                f"Failed to save mapping: {str(e)}"
            )
            return False
    
    def get_mapping_statistics(self) -> Dict[str, Any]:
        """Get current mapping statistics."""
        return {
            **self.mapping_stats,
            "timestamp": datetime.utcnow().isoformat()
        } 