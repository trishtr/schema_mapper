"""
Configuration for schema mapping rules and validation.
"""
from typing import Dict, List, Any
import re
from difflib import SequenceMatcher

# Type compatibility matrix with confidence scores
TYPE_COMPATIBILITY = {
    "VARCHAR": {
        "VARCHAR": 1.0,
        "TEXT": 0.9,
        "CHAR": 0.8,
        "STRING": 0.8
    },
    "INTEGER": {
        "INTEGER": 1.0,
        "BIGINT": 0.9,
        "SMALLINT": 0.8,
        "INT": 0.9
    },
    "FLOAT": {
        "FLOAT": 1.0,
        "DOUBLE": 0.9,
        "DECIMAL": 0.8,
        "NUMERIC": 0.8
    },
    "DATE": {
        "DATE": 1.0,
        "TIMESTAMP": 0.9,
        "DATETIME": 0.9
    }
}

# Field name patterns with confidence scores
FIELD_PATTERNS = {
    "id": {
        "patterns": [r"_id$", r"^id_", r"^id$"],
        "score": 0.9
    },
    "name": {
        "patterns": [r"_name$", r"^name_", r"^name$"],
        "score": 0.8
    },
    "date": {
        "patterns": [r"_date$", r"_at$", r"timestamp"],
        "score": 0.8
    },
    "email": {
        "patterns": [r"_email$", r"^email_", r"^email$"],
        "score": 0.9
    },
    "code": {
        "patterns": [r"_code$", r"^code_"],
        "score": 0.8
    }
}

# Value patterns with validation rules
VALUE_PATTERNS = {
    "email": {
        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "score": 0.9
    },
    "date": {
        "pattern": r"^\d{4}-\d{2}-\d{2}",
        "score": 0.8
    },
    "phone": {
        "pattern": r"^\+?[\d\-\(\)]+$",
        "score": 0.8
    },
    "numeric": {
        "pattern": r"^-?\d*\.?\d+$",
        "score": 0.7
    }
}

def calculate_name_similarity(source: str, target: str) -> float:
    """Calculate similarity between field names."""
    # Convert to lowercase
    source = source.lower()
    target = target.lower()
    
    # Remove common prefixes/suffixes
    common_affixes = ["id", "code", "name", "date", "key"]
    for affix in common_affixes:
        source = source.replace(affix, "")
        target = target.replace(affix, "")
    
    # Calculate similarity
    similarity = SequenceMatcher(None, source, target).ratio()
    return similarity

def calculate_type_compatibility(
    source_type: str,
    target_type: str
) -> float:
    """Calculate type compatibility score."""
    source_type = source_type.upper()
    target_type = target_type.upper()
    
    # Check direct compatibility
    if source_type in TYPE_COMPATIBILITY:
        compatibility = TYPE_COMPATIBILITY[source_type]
        if target_type in compatibility:
            return compatibility[target_type]
    
    # Check reverse compatibility
    for base_type, compatibles in TYPE_COMPATIBILITY.items():
        if target_type in compatibles:
            if source_type == base_type:
                return compatibles[target_type]
    
    return 0.0

def check_field_pattern(field_name: str) -> Dict[str, Any]:
    """Check field name against known patterns."""
    field_name = field_name.lower()
    matches = []
    
    for category, info in FIELD_PATTERNS.items():
        for pattern in info["patterns"]:
            if re.search(pattern, field_name):
                matches.append({
                    "category": category,
                    "score": info["score"]
                })
    
    return {
        "matches": matches,
        "max_score": max([m["score"] for m in matches]) if matches else 0.0
    }

def check_value_pattern(
    value: str,
    data_type: str
) -> Dict[str, Any]:
    """Check value against known patterns."""
    matches = []
    
    for pattern_type, info in VALUE_PATTERNS.items():
        if re.match(info["pattern"], str(value)):
            matches.append({
                "type": pattern_type,
                "score": info["score"]
            })
    
    return {
        "matches": matches,
        "max_score": max([m["score"] for m in matches]) if matches else 0.0
    }

def calculate_multi_factor_score(
    name_similarity: float,
    type_compatibility: float,
    pattern_score: float,
    value_score: float
) -> float:
    """Calculate combined confidence score."""
    weights = {
        "name": 0.3,
        "type": 0.3,
        "pattern": 0.2,
        "value": 0.2
    }
    
    score = (
        name_similarity * weights["name"] +
        type_compatibility * weights["type"] +
        pattern_score * weights["pattern"] +
        value_score * weights["value"]
    )
    
    return round(score, 2)

def apply_mapping_rules(
    source_column: Dict[str, Any],
    target_column: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply all mapping rules to calculate confidence."""
    # Calculate name similarity
    name_similarity = calculate_name_similarity(
        source_column["name"],
        target_column["name"]
    )
    
    # Calculate type compatibility
    type_compatibility = calculate_type_compatibility(
        source_column["data_type"],
        target_column["data_type"]
    )
    
    # Check field patterns
    source_patterns = check_field_pattern(source_column["name"])
    target_patterns = check_field_pattern(target_column["name"])
    pattern_score = min(
        source_patterns["max_score"],
        target_patterns["max_score"]
    )
    
    # Check value patterns
    value_score = 0.0
    if "sample_values" in source_column and "sample_values" in target_column:
        source_values = source_column["sample_values"][:3]
        target_values = target_column["sample_values"][:3]
        
        source_scores = [
            check_value_pattern(v, source_column["data_type"])["max_score"]
            for v in source_values
        ]
        target_scores = [
            check_value_pattern(v, target_column["data_type"])["max_score"]
            for v in target_values
        ]
        
        value_score = (
            sum(source_scores) / len(source_scores) if source_scores else 0.0 +
            sum(target_scores) / len(target_scores) if target_scores else 0.0
        ) / 2
    
    # Calculate final score
    confidence_score = calculate_multi_factor_score(
        name_similarity,
        type_compatibility,
        pattern_score,
        value_score
    )
    
    return {
        "confidence_score": confidence_score,
        "factors": {
            "name_similarity": name_similarity,
            "type_compatibility": type_compatibility,
            "pattern_score": pattern_score,
            "value_score": value_score
        },
        "pattern_matches": {
            "source": source_patterns["matches"],
            "target": target_patterns["matches"]
        }
    }

def validate_mapping(
    mapping_result: Dict[str, Any],
    validation_rules: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Validate mapping result against rules."""
    validation = {
        "is_valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check confidence threshold
    if mapping_result["confidence_score"] < 0.7:
        validation["warnings"].append(
            "Low confidence score, manual review recommended"
        )
    
    # Check type compatibility
    if mapping_result["factors"]["type_compatibility"] < 0.5:
        validation["errors"].append(
            "Incompatible data types"
        )
        validation["is_valid"] = False
    
    # Apply custom validation rules
    if validation_rules:
        for rule_name, rule_func in validation_rules.items():
            result = rule_func(mapping_result)
            if not result["is_valid"]:
                validation["errors"].extend(result["errors"])
                validation["is_valid"] = False
    
    return validation 