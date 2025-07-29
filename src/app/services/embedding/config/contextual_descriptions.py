"""
Contextual descriptions and semantic relationships for schema mapping.
"""
from typing import Dict, List

# Healthcare-specific contextual information
HEALTHCARE_CONTEXTS = {
    "patient": {
        "domain": "healthcare",
        "entity_type": "person",
        "privacy": "PHI",
        "related_concepts": ["medical_record", "diagnosis", "treatment"],
        "description": "Individual receiving medical care"
    },
    "provider": {
        "domain": "healthcare",
        "entity_type": "person",
        "privacy": "public",
        "related_concepts": ["doctor", "nurse", "specialist"],
        "description": "Healthcare service provider"
    },
    "diagnosis": {
        "domain": "healthcare",
        "entity_type": "medical",
        "privacy": "PHI",
        "related_concepts": ["condition", "symptoms", "treatment"],
        "description": "Medical condition identification"
    },
    "medication": {
        "domain": "healthcare",
        "entity_type": "medical",
        "privacy": "PHI",
        "related_concepts": ["prescription", "dosage", "treatment"],
        "description": "Prescribed medical treatment"
    }
}

# Semantic relationships between concepts
SEMANTIC_RELATIONSHIPS = {
    "is_a": [
        ("patient_id", "identifier"),
        ("provider_id", "identifier"),
        ("diagnosis_code", "medical_code"),
        ("medication_name", "drug_name")
    ],
    "part_of": [
        ("vital_signs", "medical_record"),
        ("lab_results", "medical_record"),
        ("dosage", "medication"),
        ("symptoms", "diagnosis")
    ],
    "related_to": [
        ("patient", "medical_record"),
        ("provider", "treatment"),
        ("diagnosis", "treatment"),
        ("medication", "prescription")
    ]
}

def generate_contextual_description(
    field_name: str,
    data_type: str,
    sample_values: List[str] = None
) -> str:
    """Generate rich contextual description for a field."""
    context = []
    
    # Add field name context
    field_lower = field_name.lower()
    for concept, info in HEALTHCARE_CONTEXTS.items():
        if concept in field_lower:
            context.append(f"Domain: {info['domain']}")
            context.append(f"Entity Type: {info['entity_type']}")
            context.append(f"Privacy Level: {info['privacy']}")
            context.append(f"Description: {info['description']}")
            context.append(f"Related to: {', '.join(info['related_concepts'])}")
    
    # Add data type context
    context.append(f"Data Type: {data_type}")
    
    # Add sample value context if available
    if sample_values and len(sample_values) > 0:
        context.append(f"Example Values: {', '.join(sample_values[:3])}")
    
    return " | ".join(context)

def get_semantic_relationships(field_name: str) -> Dict[str, List[str]]:
    """Get semantic relationships for a field."""
    field_lower = field_name.lower()
    relationships = {
        "is_a": [],
        "part_of": [],
        "related_to": []
    }
    
    # Check each relationship type
    for rel_type, rel_pairs in SEMANTIC_RELATIONSHIPS.items():
        for source, target in rel_pairs:
            if source in field_lower:
                relationships[rel_type].append(target)
            elif target in field_lower:
                relationships[rel_type].append(source)
    
    return relationships

def enrich_field_context(
    field_name: str,
    data_type: str,
    sample_values: List[str] = None,
    additional_context: Dict = None
) -> Dict:
    """Enrich field with contextual information."""
    context = {
        "name": field_name,
        "data_type": data_type,
        "description": generate_contextual_description(
            field_name,
            data_type,
            sample_values
        ),
        "semantic_relationships": get_semantic_relationships(field_name)
    }
    
    # Add any additional context
    if additional_context:
        context.update(additional_context)
    
    return context 