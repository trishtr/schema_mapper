"""
Database operations for schema mapping.
"""
from typing import List, Dict, Any, Optional
import sqlite3
import logging
from datetime import datetime

from src.app.config import settings
from src.app.models.models import MetadataMapping

class Database:
    """Database connection and operations."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.DATABASE_URL
        
    async def get_source_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a source table."""
        try:
            conn = sqlite3.connect(self.database_url)
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"""
                PRAGMA table_info({table_name})
            """)
            columns = cursor.fetchall()
            
            # Get sample data
            cursor.execute(f"""
                SELECT * FROM {table_name} LIMIT 5
            """)
            samples = cursor.fetchall()
            
            # Build schema info
            schema = {
                "table_name": table_name,
                "columns": {}
            }
            
            for col in columns:
                col_id, name, data_type, notnull, default_value, pk = col
                
                # Get sample values for this column
                sample_values = [
                    str(row[col_id])
                    for row in samples
                    if row[col_id] is not None
                ]
                
                schema["columns"][name] = {
                    "name": name,
                    "data_type": data_type.upper(),
                    "constraints": {
                        "primary_key": pk == 1,
                        "nullable": notnull == 0,
                        "default": default_value
                    },
                    "sample_values": sample_values
                }
            
            return schema
            
        except Exception as e:
            logging.error(f"Failed to get source schema: {str(e)}")
            raise
            
        finally:
            conn.close()
    
    async def get_target_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema information for a target table."""
        try:
            conn = sqlite3.connect(self.database_url)
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"""
                PRAGMA table_info({table_name})
            """)
            columns = cursor.fetchall()
            
            # Get sample data
            cursor.execute(f"""
                SELECT * FROM {table_name} LIMIT 5
            """)
            samples = cursor.fetchall()
            
            # Build schema info
            schema = {
                "table_name": table_name,
                "columns": {}
            }
            
            for col in columns:
                col_id, name, data_type, notnull, default_value, pk = col
                
                # Get sample values for this column
                sample_values = [
                    str(row[col_id])
                    for row in samples
                    if row[col_id] is not None
                ]
                
                # Get foreign key info
                cursor.execute(f"""
                    SELECT m.tbl_name, ii.name
                    FROM sqlite_master AS m,
                         pragma_table_info(m.name) AS ii
                    WHERE m.type = 'table'
                      AND ii.name = ?
                """, (name,))
                fk_info = cursor.fetchone()
                
                schema["columns"][name] = {
                    "name": name,
                    "data_type": data_type.upper(),
                    "constraints": {
                        "primary_key": pk == 1,
                        "nullable": notnull == 0,
                        "default": default_value,
                        "foreign_key": bool(fk_info)
                    },
                    "sample_values": sample_values
                }
            
            return schema
            
        except Exception as e:
            logging.error(f"Failed to get target schema: {str(e)}")
            raise
            
        finally:
            conn.close()
    
    async def get_source_column_sample(
        self,
        table_name: str,
        column_name: str,
        limit: int = 100
    ) -> List[str]:
        """Get sample values from a source column."""
        try:
            conn = sqlite3.connect(self.database_url)
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                LIMIT ?
            """, (limit,))
            
            samples = cursor.fetchall()
            return [str(row[0]) for row in samples]
            
        except Exception as e:
            logging.error(f"Failed to get column sample: {str(e)}")
            raise
            
        finally:
            conn.close()
    
    async def get_target_column_sample(
        self,
        table_name: str,
        column_name: str,
        limit: int = 100
    ) -> List[str]:
        """Get sample values from a target column."""
        try:
            conn = sqlite3.connect(self.database_url)
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT DISTINCT {column_name}
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                LIMIT ?
            """, (limit,))
            
            samples = cursor.fetchall()
            return [str(row[0]) for row in samples]
            
        except Exception as e:
            logging.error(f"Failed to get column sample: {str(e)}")
            raise
            
        finally:
            conn.close()
    
    async def save_metadata_mapping(
        self,
        mapping: MetadataMapping
    ) -> None:
        """Save metadata mapping to target database."""
        try:
            conn = sqlite3.connect(self.database_url)
            cursor = conn.cursor()
            
            # Create metadata_mapping table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata_mapping (
                    mapping_id INTEGER PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    source_table TEXT NOT NULL,
                    source_column TEXT NOT NULL,
                    target_table TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    mapping_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """)
            
            # Insert mapping
            cursor.execute("""
                INSERT INTO metadata_mapping (
                    mapping_id,
                    source_id,
                    source_table,
                    source_column,
                    target_table,
                    target_column,
                    confidence_score,
                    mapping_type,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mapping.mapping_id,
                mapping.source_id,
                mapping.source_table,
                mapping.source_column,
                mapping.target_table,
                mapping.target_column,
                mapping.confidence_score,
                mapping.mapping_type,
                mapping.created_at.isoformat(),
                mapping.updated_at.isoformat()
            ))
            
            conn.commit()
            
        except Exception as e:
            logging.error(f"Failed to save metadata mapping: {str(e)}")
            raise
            
        finally:
            conn.close()
    
    async def get_metadata_mappings(
        self,
        source_id: Optional[str] = None,
        confidence_threshold: Optional[float] = None
    ) -> List[MetadataMapping]:
        """Get metadata mappings from target database."""
        try:
            conn = sqlite3.connect(self.database_url)
            cursor = conn.cursor()
            
            query = "SELECT * FROM metadata_mapping WHERE 1=1"
            params = []
            
            if source_id:
                query += " AND source_id = ?"
                params.append(source_id)
            
            if confidence_threshold:
                query += " AND confidence_score >= ?"
                params.append(confidence_threshold)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            mappings = []
            for row in rows:
                mapping = MetadataMapping(
                    mapping_id=row[0],
                    source_id=row[1],
                    source_table=row[2],
                    source_column=row[3],
                    target_table=row[4],
                    target_column=row[5],
                    confidence_score=row[6],
                    mapping_type=row[7],
                    created_at=datetime.fromisoformat(row[8]),
                    updated_at=datetime.fromisoformat(row[9])
                )
                mappings.append(mapping)
            
            return mappings
            
        except Exception as e:
            logging.error(f"Failed to get metadata mappings: {str(e)}")
            raise
            
        finally:
            conn.close() 