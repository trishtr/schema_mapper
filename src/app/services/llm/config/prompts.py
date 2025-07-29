"""
Configuration for LLM prompts and response parsing.
"""
from typing import Dict, Any, List
import json
import re

def generate_mapping_prompt(
    source_column: Dict[str, Any],
    target_column: Dict[str, Any],
    current_mapping: Dict[str, Any]
) -> str:
    """Generate prompt for mapping analysis."""
    return f"""Analyze the following schema mapping:

Source Column:
- Name: {source_column['name']}
- Data Type: {source_column['data_type']}
- Sample Values: {source_column.get('sample_values', [])}

Target Column:
- Name: {target_column['column_name']}
- Data Type: {target_column['data_type']}
- Sample Values: {target_column.get('sample_values', [])}

Current Mapping:
- Rule Results: {current_mapping['rule_results']}
- Matches: {current_mapping['matches']}

Please analyze this mapping and provide:
1. Confidence score (0.0 to 1.0)
2. Explanation of the mapping
3. Any necessary transformations
4. Validation rules to apply

Respond in JSON format:
{
    "confidence": float,
    "explanation": string,
    "transformations": list,
    "validation_rules": list
}
"""

def generate_validation_prompt(
    mapping_result: Dict[str, Any]
) -> str:
    """Generate prompt for mapping validation."""
    return f"""Validate the following mapping result:

Mapping:
{json.dumps(mapping_result, indent=2)}

Please validate:
1. Data type compatibility
2. Value format consistency
3. Business rule compliance
4. Transformation feasibility

Respond in JSON format:
{
    "is_valid": boolean,
    "validation_score": float,
    "issues": list,
    "recommendations": list
}
"""

def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response into structured format."""
    try:
        # Clean response
        response = response.strip()
        
        # Extract JSON if wrapped in code blocks
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        
        # Parse JSON
        result = json.loads(response)
        
        # Validate required fields
        if "confidence" in result:
            result["confidence"] = float(result["confidence"])
            if not 0 <= result["confidence"] <= 1:
                raise ValueError("Confidence must be between 0 and 1")
        
        return result
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to parse response: {str(e)}")

def validate_mapping_result(
    result: Dict[str, Any]
) -> bool:
    """Validate mapping result structure."""
    required_fields = {
        "confidence": float,
        "explanation": str,
        "transformations": list
    }
    
    try:
        # Check required fields
        for field, field_type in required_fields.items():
            if field not in result:
                return False
            if not isinstance(result[field], field_type):
                return False
        
        # Validate confidence range
        if not 0 <= result["confidence"] <= 1:
            return False
        
        # Validate explanation
        if not result["explanation"].strip():
            return False
        
        return True
        
    except Exception:
        return False

def extract_transformations(
    explanation: str
) -> List[str]:
    """Extract transformations from explanation text."""
    transformations = []
    
    # Look for transformation keywords
    keywords = [
        "convert",
        "transform",
        "format",
        "change",
        "modify",
        "replace"
    ]
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', explanation)
    
    for sentence in sentences:
        sentence = sentence.strip().lower()
        if any(keyword in sentence for keyword in keywords):
            transformations.append(sentence)
    
    return transformations

def format_llm_result(
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """Format LLM result for API response."""
    return {
        "confidence": result.get("confidence", 0.0),
        "explanation": result.get("explanation", ""),
        "transformations": result.get("transformations", []),
        "validation": {
            "is_valid": result.get("is_valid", False),
            "issues": result.get("issues", []),
            "recommendations": result.get("recommendations", [])
        }
    } 