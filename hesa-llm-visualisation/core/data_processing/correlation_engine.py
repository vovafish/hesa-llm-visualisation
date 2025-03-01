#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-file Correlation Engine for HESA Data

This module provides tools for correlating data across multiple HESA data files,
identifying relationships between different metrics, and generating correlation
reports and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import json
import os

# Set up logging
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy data types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

class DataCorrelator:
    """
    A class for correlating data across multiple HESA data files.
    
    This class provides methods to:
    - Load data from different file formats (CSV, Excel, JSON)
    - Identify common keys across datasets
    - Merge data from multiple files
    - Calculate correlations between variables
    - Identify the strongest correlations
    - Generate correlation reports
    """
    
    def __init__(self):
        """Initialize the DataCorrelator."""
        logger.info("Initializing DataCorrelator")
    
    def correlate_files(self, files: List[str]) -> pd.DataFrame:
        """
        Load and merge multiple data files based on common keys.
        
        """
        if len(files) < 2:
            raise ValueError("At least two files are required for correlation")
        
        logger.info(f"Correlating {len(files)} files")
        
        # Load all dataframes
        dfs = []
        for file_path in files:
            try:
                df = self.load_file(file_path)
                # Add a source column to track which file the data came from
                df['data_source'] = Path(file_path).stem
                dfs.append(df)
                logger.info(f"Loaded {file_path} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
                raise
        
        # Find common keys across all dataframes
        common_keys = self.identify_common_keys(dfs)
        
        # If no common keys found, try to suggest potential join keys
        if not common_keys:
            logger.warning("No common columns found across all dataframes")
            suggested_keys = self.suggest_join_keys(dfs)
            logger.info(f"Suggested join keys: {suggested_keys}")
            
            # For now, use the first dataframe as the base and merge others with an index join
            merged_df = dfs[0]
            for i, df in enumerate(dfs[1:], 1):
                # Use suffix to avoid column name conflicts
                suffix = f"_file{i}"
                merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True, suffixes=('', suffix))
                
            logger.info(f"Merged dataframes using index join. Shape: {merged_df.shape}")
        else:
            # Merge using common keys
            logger.info(f"Found common keys: {common_keys}")
            merged_df = self.merge_dataframes(dfs, common_keys)
            logger.info(f"Merged dataframes using common keys. Shape: {merged_df.shape}")
        
        return merged_df
    
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a data file into a DataFrame based on its extension.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            DataFrame containing the loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load based on file extension
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == '.pkl':
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def identify_common_keys(self, dfs: List[pd.DataFrame]) -> List[str]:
        """
        Find column names that are common across all provided DataFrames.
        
        Args:
            dfs: List of DataFrames to analyze
            
        Returns:
            List of common column names
        """
        if not dfs:
            return []
        
        # Get sets of column names for each DataFrame
        column_sets = [set(df.columns) for df in dfs]
        
        # Find intersection of all column sets
        common_columns = set.intersection(*column_sets)
        
        # Convert to list and sort
        return sorted(list(common_columns))
    
    def suggest_join_keys(self, dfs: List[pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Suggest potential join keys when no common columns exist.
        
        This method looks for columns with similar names or common patterns
        that might be joinable with some transformation.
        
        Args:
            dfs: List of DataFrames to analyze
            
        Returns:
            Dictionary mapping each DataFrame index to potential join key columns
        """
        if not dfs:
            return {}
        
        # Common patterns to look for
        patterns = [
            'id', 'code', 'key', 'institution', 'year', 'academic_year',
            'date', 'time', 'period', 'university'
        ]
        
        suggestions = {}
        
        for i, df in enumerate(dfs):
            # Look for columns matching common patterns
            matching_cols = []
            for col in df.columns:
                col_lower = col.lower()
                for pattern in patterns:
                    if pattern in col_lower:
                        matching_cols.append(col)
                        break
            
            suggestions[f"df_{i}"] = matching_cols
        
        return suggestions
    
    def merge_dataframes(self, dfs: List[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
        """
        Merge multiple DataFrames using specified key columns.
        
        Args:
            dfs: List of DataFrames to merge
            keys: List of column names to use as join keys
            
        Returns:
            Merged DataFrame
        """
        if not dfs:
            raise ValueError("No DataFrames provided for merging")
        
        if not keys:
            raise ValueError("No keys provided for merging")
        
        # Start with the first DataFrame
        merged_df = dfs[0]
        
        # Merge with each subsequent DataFrame
        for i, df in enumerate(dfs[1:], 1):
            # Use suffix to avoid column name conflicts
            suffix = f"_file{i}"
            
            # Check if all keys exist in both DataFrames
            missing_keys_left = [k for k in keys if k not in merged_df.columns]
            missing_keys_right = [k for k in keys if k not in df.columns]
            
            if missing_keys_left or missing_keys_right:
                logger.warning(f"Some keys missing. Left: {missing_keys_left}, Right: {missing_keys_right}")
                # Use only keys that exist in both
                valid_keys = [k for k in keys if k in merged_df.columns and k in df.columns]
                
                if not valid_keys:
                    logger.warning(f"No valid keys for merge with DataFrame {i}. Using outer join on index.")
                    merged_df = pd.merge(merged_df, df, how='outer', left_index=True, right_index=True, suffixes=('', suffix))
                else:
                    # Handle type mismatches in join keys
                    try:
                        # First, attempt to standardize types for join keys
                        for key in valid_keys:
                            # Check if types don't match
                            if merged_df[key].dtype != df[key].dtype:
                                logger.info(f"Type mismatch for key '{key}'. Left: {merged_df[key].dtype}, Right: {df[key].dtype}")
                                
                                # If one is numeric and one is string, convert both to string
                                if (pd.api.types.is_numeric_dtype(merged_df[key]) and pd.api.types.is_string_dtype(df[key])) or \
                                   (pd.api.types.is_string_dtype(merged_df[key]) and pd.api.types.is_numeric_dtype(df[key])):
                                    logger.info(f"Converting '{key}' to string type for consistent joining")
                                    merged_df[key] = merged_df[key].astype(str)
                                    df[key] = df[key].astype(str)
                                # If both are numeric but different types, convert to float
                                elif pd.api.types.is_numeric_dtype(merged_df[key]) and pd.api.types.is_numeric_dtype(df[key]):
                                    logger.info(f"Converting '{key}' to float type for consistent joining")
                                    merged_df[key] = merged_df[key].astype(float)
                                    df[key] = df[key].astype(float)
                        
                        logger.info(f"Merging on valid keys: {valid_keys}")
                        merged_df = pd.merge(merged_df, df, how='outer', on=valid_keys, suffixes=('', suffix))
                    except Exception as e:
                        logger.warning(f"Error during merge with standardized types: {str(e)}")
                        logger.warning("Falling back to concatenation")
                        # Fallback to concatenation
                        merged_df = pd.concat([merged_df, df], axis=1)
            else:
                # All keys exist in both DataFrames, but may have type mismatches
                try:
                    # Handle type mismatches in join keys
                    for key in keys:
                        # Check if types don't match
                        if merged_df[key].dtype != df[key].dtype:
                            logger.info(f"Type mismatch for key '{key}'. Left: {merged_df[key].dtype}, Right: {df[key].dtype}")
                            
                            # If one is numeric and one is string, convert both to string
                            if (pd.api.types.is_numeric_dtype(merged_df[key]) and pd.api.types.is_string_dtype(df[key])) or \
                               (pd.api.types.is_string_dtype(merged_df[key]) and pd.api.types.is_numeric_dtype(df[key])):
                                logger.info(f"Converting '{key}' to string type for consistent joining")
                                merged_df[key] = merged_df[key].astype(str)
                                df[key] = df[key].astype(str)
                            # If both are numeric but different types, convert to float
                            elif pd.api.types.is_numeric_dtype(merged_df[key]) and pd.api.types.is_numeric_dtype(df[key]):
                                logger.info(f"Converting '{key}' to float type for consistent joining")
                                merged_df[key] = merged_df[key].astype(float)
                                df[key] = df[key].astype(float)
                    
                    merged_df = pd.merge(merged_df, df, how='outer', on=keys, suffixes=('', suffix))
                except Exception as e:
                    logger.warning(f"Error during merge: {str(e)}")
                    logger.warning("Falling back to concatenation")
                    # Fallback to concatenation
                    merged_df = pd.concat([merged_df, df], axis=1)
        
        return merged_df
    
    def calculate_correlations(self, df: pd.DataFrame, methods: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlation matrices using specified methods.
        
        Args:
            df: DataFrame containing data to correlate
            methods: List of correlation methods to use (default: ['pearson', 'spearman'])
            
        Returns:
            Dictionary mapping method names to correlation matrices
        """
        if methods is None:
            methods = ['pearson', 'spearman']
        
        # Ensure we only use numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValueError("No numeric columns found for correlation")
        
        # Calculate correlations for each method
        correlations = {}
        
        for method in methods:
            try:
                corr_matrix = numeric_df.corr(method=method)
                correlations[method] = corr_matrix
                logger.info(f"Calculated {method} correlation matrix with shape {corr_matrix.shape}")
            except Exception as e:
                logger.error(f"Error calculating {method} correlation: {str(e)}")
        
        return correlations
    
    def identify_strongest_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Tuple]:
        """
        Identify pairs of variables with the strongest correlations.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Minimum absolute correlation value to include
            
        Returns:
            List of tuples (variable1, variable2, correlation_value)
        """
        if corr_matrix.empty:
            return []
        
        # Create a copy of the correlation matrix
        corr = corr_matrix.copy()
        
        # Set self-correlations (diagonal) to NaN
        np.fill_diagonal(corr.values, np.nan)
        
        # Get pairs with strong correlations
        strong_pairs = []
        
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                col1 = corr.columns[i]
                col2 = corr.columns[j]
                corr_value = corr.iloc[i, j]
                
                # Check if correlation exceeds threshold
                if pd.notna(corr_value) and abs(corr_value) >= threshold:
                    strong_pairs.append((col1, col2, corr_value))
        
        # Sort by absolute correlation value (descending)
        strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        logger.info(f"Identified {len(strong_pairs)} strong correlations (threshold: {threshold})")
        return strong_pairs
    
    def generate_correlation_report(self, merged_df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive report of correlations and dataset information.
        
        Args:
            merged_df: Merged DataFrame containing data from multiple sources
            
        Returns:
            Dictionary containing correlation analysis and dataset information
        """
        report = {
            'dataset_info': {
                'rows': int(len(merged_df)),
                'columns': int(len(merged_df.columns)),
                'column_names': list(merged_df.columns),
                'data_sources': merged_df['data_source'].unique().tolist() if 'data_source' in merged_df.columns else [],
                'missing_values': int(merged_df.isna().sum().sum()),
                'memory_usage_bytes': int(merged_df.memory_usage(deep=True).sum()),
            },
            'column_types': {
                'numeric': list(merged_df.select_dtypes(include=[np.number]).columns),
                'categorical': list(merged_df.select_dtypes(include=['category', 'object']).columns),
                'datetime': list(merged_df.select_dtypes(include=['datetime']).columns),
                'boolean': list(merged_df.select_dtypes(include=['bool']).columns),
            }
        }
        
        # Only perform correlation analysis on numeric columns
        numeric_df = merged_df.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            # Calculate correlations
            correlations = self.calculate_correlations(numeric_df)
            
            # Add correlation summary to report
            report['correlation_analysis'] = {}
            
            for method, corr_matrix in correlations.items():
                # Find strongest correlations
                strong_correlations = self.identify_strongest_correlations(corr_matrix, threshold=0.7)
                
                # Convert strong correlations to JSON-serializable format
                strong_pairs_json = []
                for var1, var2, corr in strong_correlations[:10]:  # Top 10
                    strong_pairs_json.append({
                        'var1': str(var1), 
                        'var2': str(var2), 
                        'correlation': float(corr)
                    })
                
                # Add to report
                report['correlation_analysis'][method] = {
                    'strongest_pairs': strong_pairs_json,
                    'matrix_shape': [int(dim) for dim in corr_matrix.shape],
                    'average_abs_correlation': float(np.nanmean(np.abs(corr_matrix.values))),
                    'max_abs_correlation': float(np.nanmax(np.abs(corr_matrix.values - np.eye(len(corr_matrix))))),
                }
        
        # Add summary statistics for numeric columns in a JSON-serializable format
        if not numeric_df.empty:
            # Convert numpy types to Python types for JSON serialization
            stats_dict = {}
            for col, stats in numeric_df.describe().to_dict().items():
                stats_dict[col] = {
                    stat: float(value) if isinstance(value, (np.floating, np.integer)) else value
                    for stat, value in stats.items()
                }
            report['numeric_statistics'] = stats_dict
        
        # Add data quality metrics in a JSON-serializable format
        missing_by_col = {}
        for col, count in merged_df.isna().sum().to_dict().items():
            missing_by_col[col] = int(count)
        
        report['data_quality'] = {
            'missing_values_by_column': missing_by_col,
            'complete_rows': float((merged_df.notna().all(axis=1).sum() / len(merged_df)) * 100),
            'complete_columns': float((merged_df.notna().all(axis=0).sum() / len(merged_df.columns)) * 100),
        }
        
        logger.info(f"Generated correlation report with {len(report)} sections")
        return report


# Example usage if run as script
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example correlation workflow
    correlator = DataCorrelator()
    
    # Get current directory
    current_dir = Path(__file__).resolve().parent.parent.parent
    
    # Example files (should be adjusted to actual file paths)
    file1 = current_dir / "data" / "raw_files" / "enrollment.csv"
    file2 = current_dir / "data" / "raw_files" / "performance.csv"
    
    if file1.exists() and file2.exists():
        print(f"Correlating {file1.name} and {file2.name}...")
        
        # Correlate files
        merged_data = correlator.correlate_files([str(file1), str(file2)])
        
        # Calculate correlations
        correlations = correlator.calculate_correlations(merged_data)
        
        # Find strongest correlations
        strong_correlations = correlator.identify_strongest_correlations(
            correlations['pearson'], threshold=0.7
        )
        
        print("Strongest correlations:")
        for col1, col2, value in strong_correlations[:5]:
            print(f"{col1} and {col2}: {value:.2f}")
            
        # Generate report and save to JSON
        report = correlator.generate_correlation_report(merged_data)
        report_path = current_dir / "data" / "reports" / "correlation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        print(f"Saved correlation report to {report_path}")
    else:
        print("Example files not found. Please adjust file paths.") 