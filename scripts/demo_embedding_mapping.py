"""
Demo script for embedding-based schema mapping.
"""
import asyncio
import json
from datetime import datetime
from src.app.services.schema.schema_mapper import SchemaMapper
from src.app.models.models import ColumnInfo

async def run_demo():
    """Run schema mapping demo."""
    # Sample source schema
    source_schema = [
        ColumnInfo(
            name="patient_id",
            data_type="VARCHAR",
            sample_values=["P123", "P456", "P789"]
        ),
        ColumnInfo(
            name="email_address",
            data_type="VARCHAR",
            sample_values=["john@example.com", "jane@example.com"]
        ),
        ColumnInfo(
            name="birth_date",
            data_type="DATE",
            sample_values=["1990-01-01", "1985-12-31"]
        )
    ]

    # Sample target schema
    target_schema = [
        ColumnInfo(
            name="id",
            data_type="VARCHAR",
            sample_values=["PAT123", "PAT456"]
        ),
        ColumnInfo(
            name="email",
            data_type="VARCHAR",
            sample_values=["user@example.com"]
        ),
        ColumnInfo(
            name="dob",
            data_type="DATE",
            sample_values=["1992-05-15"]
        )
    ]

    try:
        # Initialize schema mapper
        mapper = SchemaMapper()

        # Map schemas
        print("Mapping schemas...")
        result = await mapper.map_schemas(
            source_schema,
            target_schema,
            confidence_threshold=0.8
        )

        # Save results
        output_file = f"mapping_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(
                {
                    "mappings": [
                        {
                            "source_column": m.source_column,
                            "target_column": m.target_column,
                            "confidence_score": m.confidence_score,
                            "confidence_level": m.confidence_level,
                            "compatible_types": m.compatible_types,
                            "transformations": m.transformations,
                            "semantic_relationships": m.semantic_relationships,
                            "llm_analysis": m.llm_analysis
                        }
                        for m in result["mappings"]
                    ],
                    "stats": result["stats"]
                },
                f,
                indent=2
            )

        print(f"\nResults saved to {output_file}")

        # Print summary
        print("\nMapping Summary:")
        print(f"Total mappings: {result['stats']['total_mappings']}")
        print(f"High confidence: {result['stats']['high_confidence']}")
        print(f"Low confidence: {result['stats']['low_confidence']}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_demo()) 