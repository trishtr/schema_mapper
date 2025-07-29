"""
Service for analyzing and profiling data columns.
"""
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import re
import logging

from src.app.utils.error_handling import ProfilerError
from .config.profiling_rules import (
    get_quality_score,
    get_type_confidence,
    check_relationship_type,
    suggest_column_type,
    THRESHOLDS,
    QUALITY_WEIGHTS,
    PATTERN_THRESHOLDS
)

class ProfilerService:
    """Service for data profiling and analysis."""
    
    def __init__(self):
        self.profile_cache = {}
    
    async def profile_column(
        self,
        column_name: str,
        data_type: str,
        sample_values: List[Any]
    ) -> Dict[str, Any]:
        """Profile a data column."""
        try:
            # Convert to pandas series for analysis
            series = pd.Series(sample_values)
            
            profile = {
                "name": column_name,
                "data_type": data_type,
                "sample_size": len(sample_values),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Basic statistics
            profile.update(self._analyze_basic_stats(series))
            
            # Type-specific analysis
            if np.issubdtype(series.dtype, np.number):
                profile.update(await self._analyze_numeric(series))
            elif series.dtype == 'object':
                profile.update(await self._analyze_string(series))
            elif pd.api.types.is_datetime64_any_dtype(series):
                profile.update(await self._analyze_temporal(series))
            
            # Data quality assessment
            profile["quality"] = await self._assess_data_quality(series)
            
            # Pattern detection
            profile["patterns"] = await self._detect_value_patterns(series)
            
            # Relationship detection
            profile["relationships"] = await self._detect_relationships(
                column_name,
                series
            )
            
            # Cache profile
            self.profile_cache[column_name] = profile
            
            return profile
            
        except Exception as e:
            logging.error(f"Column profiling failed: {str(e)}")
            raise ProfilerError(
                f"Failed to profile column {column_name}",
                details={"error": str(e)}
            )
    
    def _analyze_basic_stats(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Analyze basic statistics."""
        return {
            "count": len(series),
            "unique_count": series.nunique(),
            "null_count": series.isnull().sum(),
            "unique_ratio": series.nunique() / len(series),
            "null_ratio": series.isnull().sum() / len(series)
        }
    
    async def _analyze_numeric(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Analyze numeric column."""
        stats = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "quartiles": {
                "q1": float(series.quantile(0.25)),
                "q2": float(series.quantile(0.50)),
                "q3": float(series.quantile(0.75))
            }
        }
        
        # Detect outliers
        q1 = stats["quartiles"]["q1"]
        q3 = stats["quartiles"]["q3"]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = series[
            (series < lower_bound) | (series > upper_bound)
        ]
        
        stats["outliers"] = {
            "count": len(outliers),
            "ratio": len(outliers) / len(series),
            "bounds": {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            }
        }
        
        return stats
    
    async def _analyze_string(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Analyze string column."""
        # Remove null values
        series = series.dropna()
        
        stats = {
            "length_stats": {
                "min": min(series.str.len()),
                "max": max(series.str.len()),
                "mean": float(series.str.len().mean()),
                "median": float(series.str.len().median())
            },
            "case_stats": {
                "upper": sum(series.str.isupper()) / len(series),
                "lower": sum(series.str.islower()) / len(series),
                "mixed": sum(~(series.str.isupper() | series.str.islower())) / len(series)
            }
        }
        
        # Character type analysis
        stats["char_types"] = {
            "alpha": sum(series.str.isalpha()) / len(series),
            "numeric": sum(series.str.isnumeric()) / len(series),
            "alphanumeric": sum(series.str.isalnum()) / len(series),
            "spaces": sum(series.str.contains(r"\s")) / len(series),
            "special": sum(series.str.contains(r"[^a-zA-Z0-9\s]")) / len(series)
        }
        
        return stats
    
    async def _analyze_temporal(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Analyze temporal column."""
        stats = {
            "min_date": series.min().isoformat(),
            "max_date": series.max().isoformat(),
            "range_days": (series.max() - series.min()).days
        }
        
        # Distribution analysis
        stats["distribution"] = {
            "year": series.dt.year.value_counts().to_dict(),
            "month": series.dt.month.value_counts().to_dict(),
            "weekday": series.dt.dayofweek.value_counts().to_dict()
        }
        
        # Format detection
        sample = series.iloc[0]
        stats["format"] = {
            "has_time": bool(sample.time()),
            "has_timezone": bool(sample.tzinfo),
            "resolution": "seconds" if sample.second else "minutes" if sample.minute else "hours" if sample.hour else "days"
        }
        
        return stats
    
    async def _assess_data_quality(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Assess data quality metrics."""
        quality = {
            "completeness": 1 - (series.isnull().sum() / len(series)),
            "validity": 0.0,
            "consistency": 0.0,
            "uniqueness": series.nunique() / len(series)
        }
        
        # Validity check
        if np.issubdtype(series.dtype, np.number):
            quality["validity"] = 1.0
        else:
            # Check string patterns
            valid_pattern = re.compile(r"^[\w\s\-\.@]+$")
            quality["validity"] = sum(
                series.dropna().astype(str).str.match(valid_pattern)
            ) / len(series)
        
        # Consistency check
        if len(series) > 1:
            # Check value patterns
            first_value = str(series.iloc[0])
            pattern = re.compile(
                r"[A-Za-z]" if first_value.isalpha()
                else r"\d" if first_value.isdigit()
                else r"."
            )
            quality["consistency"] = sum(
                series.dropna().astype(str).str.match(pattern)
            ) / len(series)
        
        # Overall score
        quality["overall_score"] = sum(
            score * QUALITY_WEIGHTS[metric]
            for metric, score in quality.items()
            if metric != "overall_score"
        )
        
        return quality
    
    async def _detect_value_patterns(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Detect patterns in values."""
        patterns = {
            "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?1?\d{9,15}$",
            "url": r"^https?://[\w\-\.]+(:\d+)?(/[\w\-\./?%&=]*)?$",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "time": r"^\d{2}:\d{2}(:\d{2})?$",
            "ip": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
            "uuid": r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        }
        
        results = {}
        series_str = series.dropna().astype(str)
        
        for pattern_name, pattern in patterns.items():
            matches = series_str.str.match(pattern)
            match_ratio = sum(matches) / len(series)
            
            if match_ratio >= PATTERN_THRESHOLDS[pattern_name]:
                results[pattern_name] = match_ratio
        
        return results
    
    async def _detect_relationships(
        self,
        column_name: str,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Detect potential relationships."""
        relationships = {
            "type": check_relationship_type(column_name),
            "cardinality": "unknown",
            "suggested_references": []
        }
        
        # Detect cardinality
        unique_ratio = series.nunique() / len(series)
        if unique_ratio == 1.0:
            relationships["cardinality"] = "one_to_one"
        elif unique_ratio >= 0.9:
            relationships["cardinality"] = "one_to_many"
        elif unique_ratio <= 0.1:
            relationships["cardinality"] = "many_to_one"
        else:
            relationships["cardinality"] = "many_to_many"
        
        # Suggest references based on name patterns
        if "_id" in column_name.lower():
            base_name = column_name.lower().replace("_id", "")
            relationships["suggested_references"].append(f"{base_name}s")
            relationships["suggested_references"].append(base_name)
        
        return relationships 