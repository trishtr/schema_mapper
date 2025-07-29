"""
Data profiling service for analyzing source and target data.
"""
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from collections import Counter

class DataProfiler:
    """Service for profiling database columns and tables."""
    
    def __init__(self):
        self.numeric_types = ["INTEGER", "FLOAT", "DECIMAL", "NUMERIC"]
        self.string_types = ["VARCHAR", "TEXT", "CHAR", "STRING"]
        self.date_types = ["DATE", "TIMESTAMP", "DATETIME"]
        
    async def profile_column(
        self,
        column_data: List[Any],
        column_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Profile a single database column."""
        try:
            # Convert to pandas series for analysis
            series = pd.Series(column_data)
            data_type = column_info["data_type"].upper()
            
            profile = {
                "name": column_info["name"],
                "data_type": data_type,
                "statistics": self._get_basic_stats(series, data_type),
                "patterns": self._analyze_patterns(series, data_type),
                "quality": self._assess_data_quality(series),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add type-specific analysis
            if data_type in self.numeric_types:
                profile.update(self._analyze_numeric(series))
            elif data_type in self.string_types:
                profile.update(self._analyze_string(series))
            elif data_type in self.date_types:
                profile.update(self._analyze_date(series))
            
            return profile
            
        except Exception as e:
            logging.error(f"Column profiling failed: {str(e)}")
            raise ValueError(f"Failed to profile column: {str(e)}")
    
    def _get_basic_stats(
        self,
        series: pd.Series,
        data_type: str
    ) -> Dict[str, Any]:
        """Get basic statistics for a column."""
        stats = {
            "count": len(series),
            "null_count": series.isnull().sum(),
            "unique_count": series.nunique(),
            "unique_ratio": series.nunique() / len(series) if len(series) > 0 else 0
        }
        
        # Add numeric statistics if applicable
        if data_type in self.numeric_types:
            numeric_series = pd.to_numeric(series, errors='coerce')
            stats.update({
                "min": numeric_series.min(),
                "max": numeric_series.max(),
                "mean": numeric_series.mean(),
                "median": numeric_series.median(),
                "std": numeric_series.std()
            })
        
        return stats
    
    def _analyze_patterns(
        self,
        series: pd.Series,
        data_type: str
    ) -> Dict[str, Any]:
        """Analyze data patterns in the column."""
        patterns = {
            "common_prefixes": self._get_common_patterns(series, "prefix"),
            "common_suffixes": self._get_common_patterns(series, "suffix"),
            "format_consistency": self._check_format_consistency(series, data_type)
        }
        
        # Add value patterns
        value_patterns = self._detect_value_patterns(series)
        if value_patterns:
            patterns["value_patterns"] = value_patterns
        
        return patterns
    
    def _assess_data_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Assess data quality metrics."""
        total_count = len(series)
        if total_count == 0:
            return {
                "completeness": 0,
                "validity": 0,
                "consistency": 0
            }
            
        # Calculate quality metrics
        null_count = series.isnull().sum()
        empty_count = series.astype(str).str.strip().eq('').sum()
        
        # Basic pattern consistency check
        consistent_format = self._check_value_consistency(series)
        
        return {
            "completeness": (total_count - null_count - empty_count) / total_count,
            "validity": self._check_data_validity(series),
            "consistency": consistent_format
        }
    
    def _analyze_numeric(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column characteristics."""
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        analysis = {
            "distribution": {
                "skewness": numeric_series.skew(),
                "kurtosis": numeric_series.kurtosis(),
                "is_normal": self._test_normality(numeric_series)
            },
            "ranges": self._get_numeric_ranges(numeric_series),
            "outliers": self._detect_outliers(numeric_series)
        }
        
        # Detect potential categorical numeric values
        if numeric_series.nunique() < 20:
            analysis["value_counts"] = numeric_series.value_counts().to_dict()
            
        return analysis
    
    def _analyze_string(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze string column characteristics."""
        str_series = series.astype(str)
        
        return {
            "length_stats": {
                "min_length": str_series.str.len().min(),
                "max_length": str_series.str.len().max(),
                "mean_length": str_series.str.len().mean()
            },
            "character_types": self._analyze_character_types(str_series),
            "common_patterns": self._detect_string_patterns(str_series),
            "potential_types": self._suggest_semantic_types(str_series)
        }
    
    def _analyze_date(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze date/timestamp column characteristics."""
        # Convert to datetime
        date_series = pd.to_datetime(series, errors='coerce')
        
        return {
            "temporal_range": {
                "min_date": date_series.min(),
                "max_date": date_series.max(),
                "range_days": (date_series.max() - date_series.min()).days
            },
            "patterns": {
                "weekday_distribution": date_series.dt.dayofweek.value_counts().to_dict(),
                "month_distribution": date_series.dt.month.value_counts().to_dict(),
                "year_distribution": date_series.dt.year.value_counts().to_dict()
            },
            "format_analysis": self._analyze_date_formats(series)
        }
    
    def _get_common_patterns(
        self,
        series: pd.Series,
        pattern_type: str,
        min_length: int = 2
    ) -> List[Dict[str, Any]]:
        """Get common prefixes or suffixes."""
        patterns = []
        str_series = series.astype(str)
        
        if pattern_type == "prefix":
            pattern_func = lambda x: x[:min_length]
        else:  # suffix
            pattern_func = lambda x: x[-min_length:]
            
        pattern_counts = Counter(str_series.apply(pattern_func))
        
        # Get patterns that appear more than once
        for pattern, count in pattern_counts.most_common(5):
            if count > 1:
                patterns.append({
                    "pattern": pattern,
                    "count": count,
                    "frequency": count / len(series)
                })
                
        return patterns
    
    def _check_format_consistency(
        self,
        series: pd.Series,
        data_type: str
    ) -> float:
        """Check format consistency of values."""
        if len(series) == 0:
            return 1.0
            
        if data_type in self.numeric_types:
            # Check decimal places consistency
            decimal_places = series.astype(str).str.extract(r'\.(\d+)')[0].str.len()
            return 1.0 - (decimal_places.nunique() / len(series))
            
        elif data_type in self.string_types:
            # Check character type consistency
            patterns = series.astype(str).apply(self._get_char_pattern)
            return len(patterns.unique()) / len(series)
            
        return 1.0
    
    def _detect_value_patterns(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Detect common value patterns."""
        patterns = []
        str_series = series.astype(str)
        
        # Common regex patterns
        pattern_checks = {
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "phone": r'^\+?[\d\-\(\)]+$',
            "url": r'^https?://\S+$',
            "zipcode": r'^\d{5}(-\d{4})?$',
            "date_iso": r'^\d{4}-\d{2}-\d{2}',
            "numeric": r'^-?\d*\.?\d+$'
        }
        
        for pattern_name, regex in pattern_checks.items():
            matches = str_series.str.match(regex, na=False)
            match_count = matches.sum()
            
            if match_count > 0:
                patterns.append({
                    "pattern_type": pattern_name,
                    "match_count": int(match_count),
                    "match_ratio": match_count / len(series)
                })
                
        return patterns
    
    def _check_value_consistency(self, series: pd.Series) -> float:
        """Check value format consistency."""
        if len(series) == 0:
            return 1.0
            
        # Get value patterns
        patterns = series.astype(str).apply(self._get_value_pattern)
        
        # Calculate consistency ratio
        dominant_pattern = patterns.mode()[0]
        consistency = (patterns == dominant_pattern).sum() / len(series)
        
        return consistency
    
    def _check_data_validity(self, series: pd.Series) -> float:
        """Check data validity based on type-specific rules."""
        if len(series) == 0:
            return 1.0
            
        valid_count = len(series)
        
        # Remove null values from count
        valid_count -= series.isnull().sum()
        
        # Remove empty strings
        valid_count -= series.astype(str).str.strip().eq('').sum()
        
        # Calculate validity ratio
        return valid_count / len(series)
    
    def _test_normality(self, series: pd.Series) -> bool:
        """Test if numeric data follows normal distribution."""
        if len(series) < 8:  # Not enough data
            return False
            
        # Use skewness and kurtosis test
        skewness = abs(series.skew())
        kurtosis = abs(series.kurtosis())
        
        # Values close to 0 indicate normal distribution
        return skewness < 0.5 and kurtosis < 0.5
    
    def _get_numeric_ranges(
        self,
        series: pd.Series
    ) -> List[Dict[str, Any]]:
        """Get value ranges for numeric data."""
        if len(series) == 0:
            return []
            
        # Calculate quartiles
        q1 = series.quantile(0.25)
        q2 = series.quantile(0.50)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        return [
            {
                "range": "low",
                "min": series.min(),
                "max": q1,
                "count": len(series[series <= q1])
            },
            {
                "range": "mid",
                "min": q1,
                "max": q3,
                "count": len(series[(series > q1) & (series <= q3)])
            },
            {
                "range": "high",
                "min": q3,
                "max": series.max(),
                "count": len(series[series > q3])
            }
        ]
    
    def _detect_outliers(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Detect outliers in numeric data."""
        if len(series) < 4:  # Not enough data
            return {"count": 0, "values": []}
            
        # Calculate IQR
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        # Find outliers
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        return {
            "count": len(outliers),
            "values": outliers.tolist()[:5],  # First 5 outliers
            "bounds": {
                "lower": lower_bound,
                "upper": upper_bound
            }
        }
    
    def _analyze_character_types(
        self,
        series: pd.Series
    ) -> Dict[str, float]:
        """Analyze character type distributions."""
        if len(series) == 0:
            return {}
            
        total_chars = series.str.len().sum()
        if total_chars == 0:
            return {}
            
        char_counts = {
            "alphabetic": series.str.count(r'[a-zA-Z]').sum(),
            "numeric": series.str.count(r'[0-9]').sum(),
            "special": series.str.count(r'[^a-zA-Z0-9\s]').sum(),
            "whitespace": series.str.count(r'\s').sum()
        }
        
        # Calculate proportions
        return {
            k: v / total_chars
            for k, v in char_counts.items()
        }
    
    def _detect_string_patterns(
        self,
        series: pd.Series
    ) -> List[Dict[str, Any]]:
        """Detect common string patterns."""
        patterns = []
        
        # Common string patterns
        pattern_checks = {
            "all_caps": lambda x: x.isupper(),
            "all_lower": lambda x: x.islower(),
            "title_case": lambda x: x.istitle(),
            "contains_spaces": lambda x: ' ' in x,
            "alphanumeric": lambda x: x.isalnum(),
            "numeric_only": lambda x: x.isdigit()
        }
        
        for pattern_name, check_func in pattern_checks.items():
            matches = series.apply(check_func)
            match_count = matches.sum()
            
            if match_count > 0:
                patterns.append({
                    "pattern": pattern_name,
                    "count": int(match_count),
                    "ratio": match_count / len(series)
                })
                
        return patterns
    
    def _suggest_semantic_types(
        self,
        series: pd.Series
    ) -> List[Dict[str, float]]:
        """Suggest potential semantic types."""
        suggestions = []
        
        # Common semantic patterns
        semantic_patterns = {
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "url": r'^https?://\S+$',
            "phone": r'^\+?[\d\-\(\)]+$',
            "date": r'^\d{4}-\d{2}-\d{2}',
            "time": r'^\d{2}:\d{2}(:\d{2})?$',
            "ip_address": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
            "credit_card": r'^\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}$'
        }
        
        for type_name, pattern in semantic_patterns.items():
            matches = series.str.match(pattern, na=False)
            match_ratio = matches.sum() / len(series)
            
            if match_ratio > 0.1:  # At least 10% match
                suggestions.append({
                    "type": type_name,
                    "confidence": match_ratio
                })
                
        return sorted(
            suggestions,
            key=lambda x: x["confidence"],
            reverse=True
        )
    
    def _analyze_date_formats(
        self,
        series: pd.Series
    ) -> Dict[str, Any]:
        """Analyze date string formats."""
        formats = Counter()
        
        # Common date formats to check
        date_patterns = {
            "ISO": r'^\d{4}-\d{2}-\d{2}',
            "US": r'^\d{2}/\d{2}/\d{4}',
            "EU": r'^\d{2}\.\d{2}\.\d{4}',
            "TIMESTAMP": r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
        }
        
        for format_name, pattern in date_patterns.items():
            matches = series.astype(str).str.match(pattern, na=False)
            match_count = matches.sum()
            
            if match_count > 0:
                formats[format_name] = match_count
                
        return {
            "detected_formats": {
                format_name: {
                    "count": count,
                    "ratio": count / len(series)
                }
                for format_name, count in formats.most_common()
            },
            "consistency": len(formats) == 1  # True if only one format detected
        }
    
    def _get_char_pattern(self, value: str) -> str:
        """Get character pattern for a string."""
        if pd.isna(value):
            return "NA"
            
        pattern = ""
        for char in str(value):
            if char.isupper():
                pattern += "A"
            elif char.islower():
                pattern += "a"
            elif char.isdigit():
                pattern += "9"
            else:
                pattern += char
                
        return pattern
    
    def _get_value_pattern(self, value: str) -> str:
        """Get general pattern for a value."""
        if pd.isna(value):
            return "NA"
            
        # Replace character sequences with pattern
        pattern = str(value)
        pattern = re.sub(r'[A-Z]+', 'A', pattern)
        pattern = re.sub(r'[a-z]+', 'a', pattern)
        pattern = re.sub(r'\d+', '9', pattern)
        
        return pattern 