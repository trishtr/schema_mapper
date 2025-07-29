"""
Configuration for schema mapping rules and validation.
"""
from typing import Dict, Any, List
import re
from difflib import SequenceMatcher

# Type compatibility matrix
TYPE_COMPATIBILITY = {
    "VARCHAR": ["VARCHAR", "TEXT", "CHAR", "STRING"],
    "INTEGER": ["INTEGER", "BIGINT", "SMALLINT", "INT"],
    "FLOAT": ["FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"],
    "DATE": ["DATE", "TIMESTAMP", "DATETIME"],
    "BOOLEAN": ["BOOLEAN", "BOOL", "BIT"],
    "TIMESTAMP": ["TIMESTAMP", "DATETIME", "DATE"],
    "TEXT": ["TEXT", "VARCHAR", "CHAR", "STRING"],
    "NUMERIC": ["NUMERIC", "DECIMAL", "FLOAT", "DOUBLE"]
}

# Field name patterns
FIELD_PATTERNS = {
    "id": r".*_?id$",
    "email": r".*_?email.*",
    "phone": r".*_?(phone|tel|mobile).*",
    "date": r".*_?(date|dt)$",
    "name": r".*_?name$",
    "code": r".*_?code$",
    "status": r".*_?status$",
    "type": r".*_?type$",
    "description": r".*_?(desc|description)$"
}

# Value patterns
VALUE_PATTERNS = {
    "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "phone": r"^\+?1?\d{9,15}$",
    "date_iso": r"^\d{4}-\d{2}-\d{2}$",
    "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    "numeric": r"^-?\d*\.?\d+$"
}

def calculate_name_similarity(
    source_name: str,
    target_name: str
) -> float:
    """Calculate similarity between column names."""
    # Clean names
    source_clean = re.sub(r'[_\s]', '', source_name.lower())
    target_clean = re.sub(r'[_\s]', '', target_name.lower())
    
    # Use sequence matcher
    matcher = SequenceMatcher(None, source_clean, target_clean)
    return matcher.ratio()

def calculate_type_compatibility(
    source_type: str,
    target_type: str
) -> float:
    """Calculate type compatibility score."""
    source_type = source_type.upper()
    target_type = target_type.upper()
    
    # Direct match
    if source_type == target_type:
        return 1.0
    
    # Compatible types
    if source_type in TYPE_COMPATIBILITY:
        compatible_types = TYPE_COMPATIBILITY[source_type]
        if target_type in compatible_types:
            return 0.8
    
    # Incompatible
    return 0.0

def check_field_pattern(
    column_name: str,
    sample_values: List[str]
) -> Dict[str, bool]:
    """Check if column name matches known patterns."""
    matches = {}
    
    for pattern_name, pattern in FIELD_PATTERNS.items():
        if re.match(pattern, column_name.lower()):
            matches[pattern_name] = True
            
    return matches

def check_value_pattern(
    sample_values: List[str]
) -> Dict[str, float]:
    """Check if sample values match known patterns."""
    pattern_matches = {
        pattern: 0.0 for pattern in VALUE_PATTERNS
    }
    
    if not sample_values:
        return pattern_matches
    
    # Check each value against each pattern
    for value in sample_values:
        if not value:
            continue
            
        value_str = str(value)
        for pattern_name, pattern in VALUE_PATTERNS.items():
            if re.match(pattern, value_str):
                pattern_matches[pattern_name] += 1
    
    # Calculate match percentages
    total_values = len([v for v in sample_values if v])
    if total_values > 0:
        for pattern in pattern_matches:
            pattern_matches[pattern] /= total_values
    
    return pattern_matches

def calculate_multi_factor_score(
    name_similarity: float,
    type_compatibility: float,
    pattern_matches: Dict[str, bool],
    value_matches: Dict[str, float]
) -> float:
    """Calculate overall mapping score."""
    # Base score from name and type
    base_score = (
        name_similarity * 0.4 +
        type_compatibility * 0.3
    )
    
    # Pattern bonus
    pattern_score = len(pattern_matches) * 0.1
    base_score += min(pattern_score, 0.2)
    
    # Value pattern bonus
    value_score = max(value_matches.values()) * 0.1
    base_score += value_score
    
    return min(base_score, 1.0)

def apply_mapping_rules(
    source_column: Dict[str, Any],
    target_matches: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply mapping rules to potential matches."""
    if not target_matches:
        return {
            "valid": False,
            "reason": "No target matches found"
        }
    
    best_match = target_matches[0]
    
    # Calculate scores
    name_similarity = calculate_name_similarity(
        source_column["name"],
        best_match["column_name"]
    )
    
    type_compatibility = calculate_type_compatibility(
        source_column["data_type"],
        best_match["data_type"]
    )
    
    pattern_matches = check_field_pattern(
        source_column["name"],
        source_column.get("sample_values", [])
    )
    
    value_matches = check_value_pattern(
        source_column.get("sample_values", [])
    )
    
    # Calculate final score
    final_score = calculate_multi_factor_score(
        name_similarity,
        type_compatibility,
        pattern_matches,
        value_matches
    )
    
    return {
        "valid": final_score >= 0.6,
        "name_similarity": name_similarity,
        "type_compatibility": type_compatibility,
        "pattern_matches": pattern_matches,
        "value_matches": value_matches,
        "final_score": final_score
    }

def calculate_confidence_score(
    rule_results: Dict[str, Any],
    semantic_relationships: List[str]
) -> float:
    """Calculate confidence score from rule results."""
    if not rule_results["valid"]:
        return 0.0
    
    base_score = rule_results["final_score"]
    
    # Semantic relationship bonus
    relationship_bonus = len(semantic_relationships) * 0.1
    final_score = base_score + min(relationship_bonus, 0.2)
    
    return min(final_score, 1.0)

def validate_mapping(
    rule_results: Dict[str, Any]
) -> bool:
    """Validate mapping results."""
    if not rule_results["valid"]:
        return False
    
    # Check minimum scores
    if (
        rule_results["name_similarity"] < 0.3 or
        rule_results["type_compatibility"] < 0.5 or
        rule_results["final_score"] < 0.6
    ):
        return False
    
    return True 