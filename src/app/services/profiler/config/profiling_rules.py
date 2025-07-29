"""
Configuration for data profiling rules and thresholds.
"""
from typing import Dict, Any
import re

# Profiling thresholds
THRESHOLDS = {
    "completeness": 0.95,  # Minimum completeness ratio
    "validity": 0.98,      # Minimum validity ratio
    "consistency": 0.90,   # Minimum consistency ratio
    "uniqueness": 0.01,    # Minimum unique ratio for non-key fields
    "key_uniqueness": 0.95  # Minimum unique ratio for key fields
}

# Quality assessment weights
QUALITY_WEIGHTS = {
    "completeness": 0.4,
    "validity": 0.3,
    "consistency": 0.3
}

# Pattern recognition thresholds
PATTERN_THRESHOLDS = {
    "min_pattern_frequency": 0.1,  # Minimum frequency to consider a pattern
    "max_unique_ratio": 0.9,       # Maximum unique ratio for categorical data
    "min_date_range_days": 30      # Minimum date range for temporal analysis
}

# Data type inference rules
TYPE_INFERENCE_RULES = {
    "numeric": {
        "integer": {
            "patterns": [r"^\d+$"],
            "min_match_ratio": 0.95
        },
        "decimal": {
            "patterns": [r"^\d*\.\d+$"],
            "min_match_ratio": 0.95
        }
    },
    "temporal": {
        "date": {
            "patterns": [
                r"^\d{4}-\d{2}-\d{2}$",
                r"^\d{2}/\d{2}/\d{4}$"
            ],
            "min_match_ratio": 0.95
        },
        "timestamp": {
            "patterns": [
                r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
            ],
            "min_match_ratio": 0.95
        }
    },
    "string": {
        "email": {
            "patterns": [r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"],
            "min_match_ratio": 0.95
        },
        "phone": {
            "patterns": [r"^\+?[\d\-\(\)]+$"],
            "min_match_ratio": 0.95
        },
        "url": {
            "patterns": [r"^https?://\S+$"],
            "min_match_ratio": 0.95
        }
    }
}

# Statistical analysis thresholds
STATISTICAL_THRESHOLDS = {
    "outlier_zscore": 3.0,        # Z-score threshold for outliers
    "normal_skewness": 0.5,       # Maximum skewness for normal distribution
    "normal_kurtosis": 0.5,       # Maximum kurtosis for normal distribution
    "min_samples": 30             # Minimum samples for statistical analysis
}

# Column relationship rules
RELATIONSHIP_RULES = {
    "key": {
        "unique_ratio": 0.95,
        "null_ratio": 0.05,
        "patterns": [r"_id$", r"^id_", r"_key$"]
    },
    "foreign_key": {
        "unique_ratio": 0.1,
        "null_ratio": 0.1,
        "patterns": [r"_id$", r"_fk$"]
    },
    "categorical": {
        "unique_ratio": 0.1,
        "min_occurrences": 5
    }
}

def get_quality_score(metrics: Dict[str, float]) -> float:
    """Calculate overall quality score."""
    score = 0.0
    for metric, weight in QUALITY_WEIGHTS.items():
        if metric in metrics:
            score += metrics[metric] * weight
    return score

def get_type_confidence(
    pattern_matches: Dict[str, int],
    total_count: int,
    type_rules: Dict[str, Any]
) -> float:
    """Calculate confidence for type inference."""
    if total_count == 0:
        return 0.0
        
    matched_count = sum(pattern_matches.values())
    match_ratio = matched_count / total_count
    
    return match_ratio if match_ratio >= type_rules["min_match_ratio"] else 0.0

def check_relationship_type(
    profile: Dict[str, Any],
    relationship_type: str
) -> bool:
    """Check if column matches a relationship type."""
    rules = RELATIONSHIP_RULES[relationship_type]
    
    # Check unique ratio
    if "unique_ratio" in rules:
        actual_ratio = profile["statistics"]["unique_ratio"]
        if actual_ratio < rules["unique_ratio"]:
            return False
    
    # Check null ratio
    if "null_ratio" in rules:
        null_ratio = profile["statistics"]["null_count"] / profile["statistics"]["count"]
        if null_ratio > rules["null_ratio"]:
            return False
    
    # Check patterns
    if "patterns" in rules:
        name = profile["name"].lower()
        if not any(re.search(pattern, name) for pattern in rules["patterns"]):
            return False
    
    # Check categorical criteria
    if relationship_type == "categorical":
        unique_count = profile["statistics"]["unique_count"]
        if unique_count < rules["min_occurrences"]:
            return False
    
    return True

def suggest_column_type(profile: Dict[str, Any]) -> Dict[str, float]:
    """Suggest data type with confidence scores."""
    suggestions = {}
    
    # Check each type category
    for category, types in TYPE_INFERENCE_RULES.items():
        for type_name, rules in types.items():
            pattern_matches = {}
            
            # Check each pattern
            for pattern in rules["patterns"]:
                matches = sum(1 for value in profile["sample_values"]
                            if re.match(pattern, str(value)))
                if matches > 0:
                    pattern_matches[pattern] = matches
            
            # Calculate confidence
            if pattern_matches:
                confidence = get_type_confidence(
                    pattern_matches,
                    profile["statistics"]["count"],
                    rules
                )
                if confidence > 0:
                    suggestions[type_name] = confidence
    
    return suggestions 