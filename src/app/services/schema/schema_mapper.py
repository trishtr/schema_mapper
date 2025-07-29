"""
Schema mapping service that orchestrates the mapping process.
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
    validate_mapping
)

class SchemaMapper:
    """Service for mapping database schemas."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        llm_service: Optional[LLMService] = None
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service or LLMService()
        
    @cache_result(expire=3600)
    async def map_schemas(
        self,
        source_columns: List[Dict[str, Any]],
        target_columns: List[Dict[str, Any]],
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Map source columns to target columns using embeddings and LLM.
        """
        try:
            # Get embedding-based mappings
            embedding_mappings = await self.embedding_service.map_schemas_with_embeddings(
                source_columns,
                target_columns
            )
            
            results = []
            low_confidence = []
            
            # Process each mapping
            for mapping in embedding_mappings:
                source_name = mapping["source_column"]
                source_col = next(
                    col for col in source_columns
                    if col["name"] == source_name
                )
                
                # Process matches
                for match in mapping["matches"]:
                    target_name = match["column_name"]
                    target_col = next(
                        col for col in target_columns
                        if col["name"] == target_name
                    )
                    
                    # Apply mapping rules
                    rule_result = apply_mapping_rules(
                        source_col,
                        target_col
                    )
                    
                    # Combine scores
                    combined_score = (
                        match["similarity_score"] * 0.6 +
                        rule_result["confidence_score"] * 0.4
                    )
                    
                    mapping_result = {
                        "source_column": source_name,
                        "target_column": target_name,
                        "confidence_score": combined_score,
                        "similarity_score": match["similarity_score"],
                        "rule_score": rule_result["confidence_score"],
                        "compatible_types": match["compatible_types"],
                        "semantic_relationships": mapping["semantic_relationships"],
                        "validation": validate_mapping(rule_result)
                    }
                    
                    # Check confidence threshold
                    if combined_score >= confidence_threshold:
                        results.append(mapping_result)
                    else:
                        low_confidence.append(mapping_result)
            
            # Process low confidence mappings with LLM
            if low_confidence:
                llm_results = await self._process_low_confidence(
                    low_confidence,
                    source_columns,
                    target_columns
                )
                results.extend(llm_results)
            
            return {
                "mappings": sorted(
                    results,
                    key=lambda x: x["confidence_score"],
                    reverse=True
                ),
                "stats": {
                    "total_mappings": len(results),
                    "high_confidence": len([
                        m for m in results
                        if m["confidence_score"] >= confidence_threshold
                    ]),
                    "low_confidence": len([
                        m for m in results
                        if m["confidence_score"] < confidence_threshold
                    ]),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logging.error(f"Schema mapping failed: {str(e)}")
            raise SchemaMapperError(
                "Failed to map schemas",
                details={"error": str(e)}
            )
    
    async def _process_low_confidence(
        self,
        low_confidence: List[Dict[str, Any]],
        source_columns: List[Dict[str, Any]],
        target_columns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process low confidence mappings using LLM."""
        llm_results = []
        
        for mapping in low_confidence:
            source_col = next(
                col for col in source_columns
                if col["name"] == mapping["source_column"]
            )
            target_col = next(
                col for col in target_columns
                if col["name"] == mapping["target_column"]
            )
            
            try:
                # Get LLM analysis
                llm_analysis = await self.llm_service.analyze_mapping(
                    source_col,
                    target_col,
                    mapping
                )
                
                if llm_analysis["is_valid"]:
                    # Update confidence score
                    mapping["confidence_score"] = (
                        mapping["confidence_score"] * 0.4 +
                        llm_analysis["confidence"] * 0.6
                    )
                    mapping["llm_analysis"] = llm_analysis["explanation"]
                    llm_results.append(mapping)
                    
            except Exception as e:
                logging.warning(
                    f"LLM analysis failed for {mapping['source_column']}: {str(e)}"
                )
                continue
        
        return llm_results
    
    @cache_result(expire=300)
    async def get_mapping_suggestions(
        self,
        column_info: Dict[str, Any],
        target_columns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get mapping suggestions for a single column."""
        try:
            # Generate embedding
            embedding_result = await self.embedding_service.generate_column_embedding(
                column_info
            )
            
            # Find similar columns
            matches = await self.embedding_service.find_similar_columns(
                embedding_result["embedding"]
            )
            
            suggestions = []
            for match in matches:
                target_col = next(
                    col for col in target_columns
                    if col["name"] == match["column_name"]
                )
                
                # Apply mapping rules
                rule_result = apply_mapping_rules(
                    column_info,
                    target_col
                )
                
                suggestions.append({
                    "target_column": match["column_name"],
                    "confidence_score": (
                        match["similarity_score"] * 0.6 +
                        rule_result["confidence_score"] * 0.4
                    ),
                    "compatible_types": match["compatible_types"],
                    "explanation": rule_result["factors"]
                })
            
            return sorted(
                suggestions,
                key=lambda x: x["confidence_score"],
                reverse=True
            )
            
        except Exception as e:
            logging.error(f"Mapping suggestions failed: {str(e)}")
            raise SchemaMapperError(
                "Failed to get mapping suggestions",
                details={"error": str(e)}
            )
    
    async def validate_mapping_result(
        self,
        mapping_result: Dict[str, Any],
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate a mapping result."""
        try:
            # Basic validation
            validation = validate_mapping(
                mapping_result,
                validation_rules
            )
            
            # Additional validation with LLM for complex cases
            if validation["is_valid"] and mapping_result["confidence_score"] < 0.9:
                try:
                    llm_validation = await self.llm_service.validate_mapping(
                        mapping_result
                    )
                    
                    # Combine validations
                    validation["warnings"].extend(llm_validation["warnings"])
                    validation["errors"].extend(llm_validation["errors"])
                    validation["is_valid"] = (
                        validation["is_valid"] and
                        llm_validation["is_valid"]
                    )
                    
                except Exception as e:
                    logging.warning(f"LLM validation failed: {str(e)}")
            
            return validation
            
        except Exception as e:
            logging.error(f"Mapping validation failed: {str(e)}")
            raise SchemaMapperError(
                "Failed to validate mapping",
                details={"error": str(e)}
            ) 