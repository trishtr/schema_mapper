"""
Configuration for generating contextual descriptions.
"""
from typing import Dict, List, Any, Optional

# Healthcare contexts
HEALTHCARE_CONTEXTS = {
    "patient": {
        "domain": "healthcare",
        "entity_type": "person",
        "related_concepts": [
            "demographics",
            "medical_history",
            "insurance"
        ]
    },
    "provider": {
        "domain": "healthcare",
        "entity_type": "person",
        "related_concepts": [
            "credentials",
            "specialties",
            "practice"
        ]
    },
    "facility": {
        "domain": "healthcare",
        "entity_type": "organization",
        "related_concepts": [
            "location",
            "services",
            "capacity"
        ]
    },
    "visit": {
        "domain": "healthcare",
        "entity_type": "event",
        "related_concepts": [
            "appointment",
            "encounter",
            "treatment"
        ]
    },
    "diagnosis": {
        "domain": "healthcare",
        "entity_type": "condition",
        "related_concepts": [
            "symptoms",
            "icd_codes",
            "treatment"
        ]
    },
    "procedure": {
        "domain": "healthcare",
        "entity_type": "service",
        "related_concepts": [
            "cpt_codes",
            "treatment",
            "billing"
        ]
    },
    "medication": {
        "domain": "healthcare",
        "entity_type": "drug",
        "related_concepts": [
            "prescription",
            "dosage",
            "pharmacy"
        ]
    },
    "insurance": {
        "domain": "healthcare",
        "entity_type": "coverage",
        "related_concepts": [
            "policy",
            "claims",
            "benefits"
        ]
    }
}

# Semantic relationships
SEMANTIC_RELATIONSHIPS = {
    "patient": [
        "demographics",
        "medical_history",
        "visits",
        "insurance"
    ],
    "provider": [
        "credentials",
        "specialties",
        "patients",
        "facilities"
    ],
    "facility": [
        "providers",
        "services",
        "patients",
        "equipment"
    ],
    "visit": [
        "patient",
        "provider",
        "facility",
        "diagnosis"
    ],
    "diagnosis": [
        "patient",
        "visit",
        "procedures",
        "medications"
    ],
    "procedure": [
        "diagnosis",
        "provider",
        "facility",
        "billing"
    ],
    "medication": [
        "prescription",
        "provider",
        "pharmacy",
        "diagnosis"
    ],
    "insurance": [
        "patient",
        "policy",
        "claims",
        "coverage"
    ]
}

def generate_contextual_description(
    column_name: str,
    data_type: str,
    sample_values: List[str]
) -> str:
    """Generate rich contextual description."""
    components = []
    
    # Add domain context
    context = _get_domain_context(column_name)
    if context:
        components.append(
            f"Domain: {context['domain']} | "
            f"Entity Type: {context['entity_type']}"
        )
        
        if context["related_concepts"]:
            concepts = ", ".join(context["related_concepts"])
            components.append(f"Related: {concepts}")
    
    # Add data type context
    components.append(f"Type: {data_type}")
    
    # Add value patterns
    if sample_values:
        pattern = _detect_value_pattern(sample_values)
        if pattern:
            components.append(f"Pattern: {pattern}")
    
    return " | ".join(components)

def get_semantic_relationships(
    column_name: str
) -> List[str]:
    """Get semantic relationships for a column."""
    relationships = []
    
    # Check each context
    for context, related in SEMANTIC_RELATIONSHIPS.items():
        if context in column_name.lower():
            relationships.extend(related)
            
            # Add reverse relationships
            for rel in related:
                if rel in SEMANTIC_RELATIONSHIPS:
                    reverse = SEMANTIC_RELATIONSHIPS[rel]
                    if context in reverse:
                        relationships.extend(
                            [r for r in reverse if r != context]
                        )
    
    return list(set(relationships))

def enrich_field_context(
    column_name: str,
    data_type: str,
    sample_values: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Enrich field with contextual information."""
    context = {
        "name": column_name,
        "data_type": data_type,
        "domain_context": None,
        "semantic_relationships": [],
        "value_patterns": None
    }
    
    # Add domain context
    domain_context = _get_domain_context(column_name)
    if domain_context:
        context["domain_context"] = domain_context
    
    # Add semantic relationships
    relationships = get_semantic_relationships(column_name)
    if relationships:
        context["semantic_relationships"] = relationships
    
    # Add value patterns
    if sample_values:
        pattern = _detect_value_pattern(sample_values)
        if pattern:
            context["value_patterns"] = pattern
    
    return context

def _get_domain_context(
    column_name: str
) -> Optional[Dict[str, Any]]:
    """Get domain context for a column."""
    column_lower = column_name.lower()
    
    # Check each context
    for context_name, context_info in HEALTHCARE_CONTEXTS.items():
        if context_name in column_lower:
            return context_info
    
    return None

def _detect_value_pattern(
    sample_values: List[str]
) -> Optional[str]:
    """Detect pattern in sample values."""
    if not sample_values:
        return None
    
    # Convert to strings
    values = [str(v) for v in sample_values if v]
    if not values:
        return None
    
    # Check common patterns
    first_value = values[0]
    
    if all(v.isdigit() for v in values):
        return "numeric_sequence"
    elif all(len(v) == len(first_value) for v in values):
        return f"fixed_length_{len(first_value)}"
    elif all("@" in v for v in values):
        return "email_address"
    elif all(v.startswith(("P", "PAT")) for v in values):
        return "patient_identifier"
    elif all(v.startswith(("DR", "DOC")) for v in values):
        return "provider_identifier"
    elif all("-" in v and len(v.split("-")) == 3 for v in values):
        return "date_format"
    
    return None 