"""
Configuration for data profiling rules and thresholds.
"""
from typing import Dict, Any, List, Optional
import re

# Quality thresholds
THRESHOLDS = {
    "completeness": 0.95,  # Max 5% null values
    "validity": 0.98,      # Max 2% invalid values
    "consistency": 0.90,   # Max 10% inconsistent values
    "uniqueness": 0.05     # Min 5% unique values
}

# Quality assessment weights
QUALITY_WEIGHTS = {
    "completeness": 0.3,
    "validity": 0.3,
    "consistency": 0.2,
    "uniqueness": 0.2
}

# Pattern detection thresholds
PATTERN_THRESHOLDS = {
    "email": 0.9,      # 90% match for email pattern
    "phone": 0.8,      # 80% match for phone pattern
    "url": 0.9,        # 90% match for URL pattern
    "date": 0.9,       # 90% match for date pattern
    "time": 0.9,       # 90% match for time pattern
    "ip": 0.95,        # 95% match for IP pattern
    "uuid": 0.95       # 95% match for UUID pattern
}

# Statistical thresholds
STATISTICAL_THRESHOLDS = {
    "outlier_zscore": 3.0,     # Z-score for outlier detection
    "normal_skewness": 0.5,    # Max absolute skewness for normal distribution
    "normal_kurtosis": 0.5,    # Max absolute kurtosis for normal distribution
    "correlation": 0.7         # Min correlation coefficient for strong relationship
}

# Type inference rules
TYPE_INFERENCE_RULES = {
    "integer": {
        "pattern": r"^\d+$",
        "min_match": 0.9
    },
    "float": {
        "pattern": r"^-?\d*\.?\d+$",
        "min_match": 0.9
    },
    "date": {
        "pattern": r"^\d{4}-\d{2}-\d{2}$",
        "min_match": 0.9
    },
    "datetime": {
        "pattern": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",
        "min_match": 0.9
    },
    "boolean": {
        "values": ["true", "false", "0", "1", "yes", "no"],
        "min_match": 0.9
    },
    "email": {
        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "min_match": 0.9
    }
}

# Relationship rules
RELATIONSHIP_RULES = {
    "one_to_one": {
        "unique_ratio_min": 0.95,
        "null_ratio_max": 0.05
    },
    "one_to_many": {
        "unique_ratio_min": 0.1,
        "null_ratio_max": 0.1
    },
    "many_to_one": {
        "unique_ratio_max": 0.1,
        "null_ratio_max": 0.1
    }
}

def get_quality_score(
    metric_scores: Dict[str, float]
) -> float:
    """Calculate overall quality score."""
    weighted_score = sum(
        score * QUALITY_WEIGHTS[metric]
        for metric, score in metric_scores.items()
        if metric in QUALITY_WEIGHTS
    )
    
    return min(weighted_score, 1.0)

def get_type_confidence(
    value: str,
    inferred_type: str
) -> float:
    """Get confidence score for type inference."""
    if inferred_type not in TYPE_INFERENCE_RULES:
        return 0.0
    
    rule = TYPE_INFERENCE_RULES[inferred_type]
    value_lower = str(value).lower()
    
    if "pattern" in rule:
        return 1.0 if re.match(rule["pattern"], value) else 0.0
    elif "values" in rule:
        return 1.0 if value_lower in rule["values"] else 0.0
    
    return 0.0

def check_relationship_type(
    unique_ratio: float,
    null_ratio: float
) -> Optional[str]:
    """Determine relationship type based on ratios."""
    for rel_type, rules in RELATIONSHIP_RULES.items():
        if rel_type == "one_to_one":
            if (
                unique_ratio >= rules["unique_ratio_min"] and
                null_ratio <= rules["null_ratio_max"]
            ):
                return rel_type
        elif rel_type == "one_to_many":
            if (
                unique_ratio >= rules["unique_ratio_min"] and
                null_ratio <= rules["null_ratio_max"]
            ):
                return rel_type
        elif rel_type == "many_to_one":
            if (
                unique_ratio <= rules["unique_ratio_max"] and
                null_ratio <= rules["null_ratio_max"]
            ):
                return rel_type
    
    return None

def suggest_column_type(
    sample_values: List[str]
) -> Dict[str, Any]:
    """Suggest column type based on sample values."""
    if not sample_values:
        return {
            "suggested_type": "unknown",
            "confidence": 0.0
        }
    
    type_matches = {
        t: 0 for t in TYPE_INFERENCE_RULES
    }
    total_values = len([v for v in sample_values if v])
    
    if total_values == 0:
        return {
            "suggested_type": "unknown",
            "confidence": 0.0
        }
    
    # Check each value against each type
    for value in sample_values:
        if not value:
            continue
            
        for type_name, rule in TYPE_INFERENCE_RULES.items():
            confidence = get_type_confidence(value, type_name)
            if confidence > 0:
                type_matches[type_name] += 1
    
    # Calculate match ratios
    type_ratios = {
        t: matches / total_values
        for t, matches in type_matches.items()
    }
    
    # Find best match
    best_type = max(
        type_ratios.items(),
        key=lambda x: x[1]
    )
    
    if best_type[1] >= TYPE_INFERENCE_RULES[best_type[0]]["min_match"]:
        return {
            "suggested_type": best_type[0],
            "confidence": best_type[1]
        }
    
    return {
        "suggested_type": "string",
        "confidence": 1.0
    } 