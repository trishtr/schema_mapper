"""
Business rules and data type compatibility configuration.
"""
from typing import Dict, List, Any, Optional

# Healthcare field descriptions
HEALTHCARE_FIELDS = {
    "patient_id": "Unique identifier for patient records",
    "npi": "National Provider Identifier",
    "icd_code": "International Classification of Diseases code",
    "provider_id": "Unique identifier for healthcare providers",
    "specialty": "Medical specialty or practice area",
    "license_number": "Professional license or certification number",
    "dea_number": "Drug Enforcement Administration number",
    "facility_id": "Healthcare facility identifier",
    "diagnosis": "Medical diagnosis or condition",
    "procedure_code": "Medical procedure identifier",
    "medication_code": "Medication or drug identifier",
    "visit_id": "Healthcare visit or encounter identifier",
    "insurance_id": "Insurance policy or member identifier",
    "claim_id": "Healthcare claim identifier",
    "lab_result_id": "Laboratory result identifier",
    "order_id": "Medical order or prescription identifier"
}

# Data type compatibility matrix
DATA_TYPE_COMPATIBILITY = {
    "VARCHAR": ["VARCHAR", "TEXT", "CHAR", "STRING"],
    "INTEGER": ["INTEGER", "BIGINT", "SMALLINT", "INT"],
    "FLOAT": ["FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"],
    "DATE": ["DATE", "TIMESTAMP", "DATETIME"],
    "BOOLEAN": ["BOOLEAN", "BOOL", "BIT"],
    "TIMESTAMP": ["TIMESTAMP", "DATETIME", "DATE"],
    "TEXT": ["TEXT", "VARCHAR", "CHAR", "STRING"],
    "NUMERIC": ["NUMERIC", "DECIMAL", "FLOAT", "DOUBLE"]
}

# Field constraints
FIELD_CONSTRAINTS = {
    "id": {
        "required": True,
        "unique": True,
        "min_length": 1
    },
    "name": {
        "required": True,
        "min_length": 2
    },
    "email": {
        "required": False,
        "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    },
    "phone": {
        "required": False,
        "pattern": r"^\+?1?\d{9,15}$"
    },
    "date": {
        "required": True,
        "min_value": "1900-01-01",
        "max_value": "2100-12-31"
    }
}

def get_field_description(field_name: str) -> Optional[str]:
    """Get description for a healthcare field."""
    # Check exact match
    if field_name in HEALTHCARE_FIELDS:
        return HEALTHCARE_FIELDS[field_name]
    
    # Check partial matches
    for key, description in HEALTHCARE_FIELDS.items():
        if key in field_name or field_name in key:
            return description
    
    return None

def get_compatible_types(data_type: str) -> List[str]:
    """Get list of compatible data types."""
    data_type = data_type.upper()
    
    # Check direct compatibility
    if data_type in DATA_TYPE_COMPATIBILITY:
        return DATA_TYPE_COMPATIBILITY[data_type]
    
    # Check reverse compatibility
    for base_type, compatible_types in DATA_TYPE_COMPATIBILITY.items():
        if data_type in compatible_types:
            return [base_type] + compatible_types
    
    return [data_type]

def get_field_constraints(
    field_name: str,
    data_type: str
) -> Dict[str, Any]:
    """Get constraints for a field."""
    constraints = {}
    
    # Check field-specific constraints
    for pattern, rules in FIELD_CONSTRAINTS.items():
        if pattern in field_name:
            constraints.update(rules)
    
    # Add type-specific constraints
    if "INT" in data_type.upper():
        constraints.update({
            "min_value": -2147483648,
            "max_value": 2147483647
        })
    elif "VARCHAR" in data_type.upper():
        constraints.update({
            "max_length": 255
        })
    elif "TEXT" in data_type.upper():
        constraints.update({
            "max_length": 65535
        })
    
    return constraints

def is_type_compatible(
    source_type: str,
    target_type: str
) -> bool:
    """Check if two data types are compatible."""
    compatible_types = get_compatible_types(source_type)
    return target_type.upper() in [t.upper() for t in compatible_types] 