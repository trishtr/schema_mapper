"""
Prompt templates for LLM interactions.
"""
from typing import Dict, Any

def generate_mapping_prompt(
    source_column: Dict[str, Any],
    target_column: Dict[str, Any],
    current_mapping: Dict[str, Any]
) -> str:
    """Generate prompt for analyzing column mapping."""
    prompt = f"""
Analyze the following database column mapping:

Source Column:
- Name: {source_column['name']}
- Data Type: {source_column['data_type']}
- Sample Values: {', '.join(source_column.get('sample_values', [])[:3])}

Target Column:
- Name: {target_column['name']}
- Data Type: {target_column['data_type']}
- Sample Values: {', '.join(target_column.get('sample_values', [])[:3])}

Current Mapping Analysis:
- Confidence Score: {current_mapping['confidence_score']}
- Similarity Score: {current_mapping['similarity_score']}
- Rule Score: {current_mapping['rule_score']}
- Compatible Types: {', '.join(current_mapping['compatible_types'])}

Please analyze this mapping and provide:
1. Whether this mapping is valid
2. Confidence score (0.0-1.0)
3. Explanation of your reasoning
4. Any potential data transformation needed

Format your response as JSON:
{{
    "is_valid": boolean,
    "confidence": float,
    "explanation": string,
    "transformations": [string]
}}
"""
    return prompt

def generate_validation_prompt(mapping_result: Dict[str, Any]) -> str:
    """Generate prompt for validating mapping result."""
    prompt = f"""
Validate the following schema mapping result:

Source Column: {mapping_result['source_column']}
Target Column: {mapping_result['target_column']}
Confidence Score: {mapping_result['confidence_score']}
Compatible Types: {', '.join(mapping_result['compatible_types'])}

Additional Context:
- Semantic Relationships: {mapping_result.get('semantic_relationships', {})}
- Rule Analysis: {mapping_result.get('rule_score', 'N/A')}

Please validate this mapping and provide:
1. Whether the mapping is valid
2. Any warnings or potential issues
3. Any errors that should prevent this mapping
4. Suggested improvements

Format your response as JSON:
{{
    "is_valid": boolean,
    "warnings": [string],
    "errors": [string],
    "suggestions": [string]
}}
"""
    return prompt

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response into structured format."""
    try:
        # Basic cleanup
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
            
        # Parse JSON
        import json
        result = json.loads(response)
        
        return result
        
    except Exception as e:
        raise ValueError(f"Failed to parse LLM response: {str(e)}")

def validate_mapping_result(result: Dict[str, Any]) -> bool:
    """Validate mapping result structure."""
    required_fields = {
        "analyze": ["is_valid", "confidence", "explanation"],
        "validate": ["is_valid", "warnings", "errors"]
    }
    
    # Check result type
    if "explanation" in result:
        fields = required_fields["analyze"]
    else:
        fields = required_fields["validate"]
    
    # Validate fields
    for field in fields:
        if field not in result:
            return False
            
    return True 