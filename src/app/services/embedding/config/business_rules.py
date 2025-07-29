"""
Business rules and data type compatibility configurations.
"""
from typing import List, Dict

# Field descriptions for common healthcare fields
HEALTHCARE_FIELDS = {
    "patient_id": "Unique identifier for patient records",
    "npi": "National Provider Identifier for healthcare providers",
    "icd_code": "International Classification of Diseases code",
    "diagnosis": "Medical diagnosis description",
    "medication": "Prescribed medication name",
    "dosage": "Medication dosage amount",
    "vital_signs": "Patient vital sign measurements",
    "lab_results": "Laboratory test results",
    "procedure_code": "Medical procedure identifier code",
    "insurance_id": "Health insurance identifier"
}

# Data type compatibility matrix
DATA_TYPE_COMPATIBILITY = {
    "VARCHAR": ["VARCHAR", "TEXT", "CHAR", "STRING"],
    "INTEGER": ["INTEGER", "BIGINT", "SMALLINT", "INT"],
    "FLOAT": ["FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"],
    "DATE": ["DATE", "TIMESTAMP", "DATETIME"],
    "BOOLEAN": ["BOOLEAN", "TINYINT", "BIT"],
    "TEXT": ["TEXT", "VARCHAR", "CHAR", "STRING"],
    "TIMESTAMP": ["TIMESTAMP", "DATETIME", "DATE"],
    "NUMERIC": ["NUMERIC", "DECIMAL", "FLOAT", "DOUBLE"]
}

def get_field_description(field_name: str) -> str:
    """Get description for a known field."""
    # Convert to lowercase for case-insensitive matching
    field_lower = field_name.lower()
    
    # Check exact matches
    if field_lower in HEALTHCARE_FIELDS:
        return HEALTHCARE_FIELDS[field_lower]
    
    # Check partial matches
    for key, desc in HEALTHCARE_FIELDS.items():
        if key in field_lower or field_lower in key:
            return desc
    
    return ""

def get_compatible_types(data_type: str) -> List[str]:
    """Get list of compatible data types."""
    data_type = data_type.upper()
    
    # Direct compatibility
    if data_type in DATA_TYPE_COMPATIBILITY:
        return DATA_TYPE_COMPATIBILITY[data_type]
    
    # Reverse lookup
    for primary_type, compatible_types in DATA_TYPE_COMPATIBILITY.items():
        if data_type in compatible_types:
            return [primary_type] + compatible_types
    
    return [data_type]  # Return original type if no compatibility found

def is_type_compatible(source_type: str, target_type: str) -> bool:
    """Check if two data types are compatible."""
    source_type = source_type.upper()
    target_type = target_type.upper()
    
    # Check direct compatibility
    compatible_types = get_compatible_types(source_type)
    return target_type in compatible_types

def get_field_constraints(field_name: str, data_type: str) -> Dict:
    """Get common constraints for a field."""
    field_lower = field_name.lower()
    constraints = {}
    
    # ID fields
    if "id" in field_lower or field_lower.endswith("_id"):
        constraints["unique"] = True
        constraints["nullable"] = False
    
    # Email fields
    if "email" in field_lower:
        constraints["format"] = "email"
        constraints["max_length"] = 255
    
    # Date fields
    if any(x in field_lower for x in ["date", "time", "timestamp"]):
        constraints["format"] = "datetime"
    
    # Code fields
    if field_lower.endswith("_code"):
        constraints["max_length"] = 50
        
    # Name fields
    if field_lower.endswith("_name"):
        constraints["max_length"] = 100
        
    return constraints 