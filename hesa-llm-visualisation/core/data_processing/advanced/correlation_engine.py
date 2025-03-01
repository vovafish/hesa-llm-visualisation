import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set, Union
from pathlib import Path
import os
import json

logger = logging.getLogger(__name__)

class DataCorrelator:
    """
    Handles cross-file correlation of HESA data sources, allowing data from multiple
    files to be integrated and analyzed together.
    """
    
    def __init__(self, max_file_size_mb: int = 100):
        self.max_file_size_mb = max_file_size_mb
        
    def correlate_files(self, files: List[str]) -> pd.DataFrame:
        
        #Load and correlate multiple data files into a single DataFrame.
       
    
        if not files:
            raise ValueError("No files provided for correlation")
            
        logger.info(f"Correlating {len(files)} files")
        
        # Load all dataframes
        dfs = []
        for file_path in files:
            try:
                df = self.load_file(file_path)
                dfs.append(df)
                logger.info(f"Loaded file: {file_path} with shape {df.shape}")
            except Exception as e:
                logger.error(f"Failed to load file {file_path}: {str(e)}")
                raise ValueError(f"Failed to load file {file_path}: {str(e)}")
        
        # Identify and validate common keys
        common_keys = self.identify_common_keys(dfs)
        if not common_keys:
            # Try to find potential join keys
            potential_keys = self.suggest_join_keys(dfs)
            msg = f"No common columns found across all files. Potential join keys: {potential_keys}"
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info(f"Identified common keys: {common_keys}")
        
        # Merge dataframes
        return self.merge_dataframes(dfs, common_keys)
    
    def load_file(self, file_path: str) -> pd.DataFrame:
       
        #Load a file into a pandas DataFrame based on file extension.
        
        file_path = Path(file_path)
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File {file_path} exceeds maximum size limit ({file_size_mb:.2f}MB > {self.max_file_size_mb}MB)")
        
        # Load based on extension
        ext = file_path.suffix.lower()
        
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        elif ext == '.json':
            return pd.json_normalize(json.load(open(file_path)))
        elif ext == '.pickle' or ext == '.pkl':
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def identify_common_keys(self, dfs: List[pd.DataFrame]) -> List[str]:
        # Find common columns across all dataframes.
    
        if not dfs:
            return []
            
        # Get the set of columns from each dataframe
        column_sets = [set(df.columns) for df in dfs]
        
        # Find intersection of all sets
        common_columns = set.intersection(*column_sets)
        
        return list(common_columns)
    
    def suggest_join_keys(self, dfs: List[pd.DataFrame]) -> Dict[str, List[str]]:
        """
        Suggest potential join keys when no common columns exist.
        
        This method looks for:
        1. Columns with similar names (e.g., 'student_id' and 'student-id')
        2. Columns with unique values that could serve as keys
        """
        potential_keys = {}
        
        for i, df in enumerate(dfs):
            # Look for columns that might be primary keys (high cardinality)
            possible_keys = []
            
            for col in df.columns:
                # Check if column name suggests it's an ID
                if any(id_term in col.lower() for id_term in ['id', 'key', 'code']):
                    possible_keys.append(col)
                    continue
                    
                # Check for unique values (at least 80% unique for larger datasets)
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
                if unique_ratio > 0.8 and df[col].nunique() > 10:
                    possible_keys.append(col)
            
            potential_keys[f"file_{i}"] = possible_keys
            
        return potential_keys
    
    def merge_dataframes(self, dfs: List[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
       
        #Merge multiple dataframes based on common keys.
     
        if not dfs:
            raise ValueError("No dataframes to merge")
        
        if len(dfs) == 1:
            return dfs[0]
        
        # Start with the first dataframe
        result = dfs[0]
        
        # Merge with each subsequent dataframe
        for i, df in enumerate(dfs[1:], 1):
            try:
                # Create suffixes for overlapping columns
                suffixes = (f"_file1", f"_file{i+1}")
                
                # Merge using common keys
                result = pd.merge(
                    result, 
                    df, 
                    on=keys, 
                    how='outer',  # Use outer join to keep all data
                    suffixes=suffixes,
                    indicator=True  # Add indicator column showing merge source
                )
                
                # Rename indicator for clarity
                result = result.rename(columns={'_merge': f'_merge_step{i}'})
                
                logger.info(f"Merged dataframe {i+1}, resulting shape: {result.shape}")
                
            except Exception as e:
                logger.error(f"Error merging dataframe {i+1}: {str(e)}")
                raise ValueError(f"Failed to merge dataframe {i+1}: {str(e)}")
        
        return result
    
    def calculate_correlations(self, df: pd.DataFrame, methods: List[str] = ['pearson']) -> Dict[str, pd.DataFrame]:
        
        #Calculate correlations between numeric columns in the merged dataframe.
        
        
        # Filter numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns found for correlation analysis")
            return {}
        
        results = {}
        
        for method in methods:
            try:
                if method.lower() not in ['pearson', 'kendall', 'spearman']:
                    logger.warning(f"Unsupported correlation method: {method}")
                    continue
                    
                corr_matrix = numeric_df.corr(method=method.lower())
                results[method] = corr_matrix
                logger.info(f"Calculated {method} correlation matrix with shape {corr_matrix.shape}")
                
            except Exception as e:
                logger.error(f"Error calculating {method} correlation: {str(e)}")
        
        return results

    def identify_strongest_correlations(self, 
                                       corr_matrix: pd.DataFrame, 
                                       threshold: float = 0.7,
                                       exclude_self: bool = True) -> List[Tuple[str, str, float]]:
       
        #Identify strongest correlations from a correlation matrix.
        
     
        strong_correlations = []
        
        # Create a copy and set upper triangle to NaN to avoid duplicate pairs
        mask = np.triu(np.ones(corr_matrix.shape), k=0 if not exclude_self else 1).astype(bool)
        corr_matrix_masked = corr_matrix.mask(mask)
        
        # Find correlations above threshold
        for col in corr_matrix.columns:
            for idx, value in corr_matrix_masked[col].items():
                if pd.notnull(value) and abs(value) >= threshold:
                    strong_correlations.append((col, idx, value))
        
        # Sort by absolute correlation value (descending)
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return strong_correlations
        
    def generate_correlation_report(self, merged_df: pd.DataFrame) -> Dict:
        
        #Generate a comprehensive correlation report.
        
        
        report = {
            "dataset_info": {
                "shape": merged_df.shape,
                "columns": list(merged_df.columns),
                "numeric_columns": list(merged_df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(merged_df.select_dtypes(include=['object', 'category']).columns)
            },
            "correlations": {},
            "strongest_correlations": {},
            "summary": {}
        }
        
        # Calculate correlations
        correlation_matrices = self.calculate_correlations(merged_df, methods=['pearson', 'spearman'])
        
        # For each method, get the strongest correlations
        for method, matrix in correlation_matrices.items():
            # Store the full correlation matrix (as dictionary for serialization)
            report["correlations"][method] = matrix.to_dict()
            
            # Find and store strongest correlations
            strongest = self.identify_strongest_correlations(matrix, threshold=0.6)
            report["strongest_correlations"][method] = [
                {"column1": col1, "column2": col2, "value": float(val)}
                for col1, col2, val in strongest[:10]  # Top 10
            ]
            
            # Summary statistics for this correlation method
            if strongest:
                avg_correlation = sum(abs(val) for _, _, val in strongest) / len(strongest)
                report["summary"][method] = {
                    "count_strong_correlations": len(strongest),
                    "average_correlation": float(avg_correlation),
                    "max_correlation": float(max(abs(val) for _, _, val in strongest))
                }
        
        return report 