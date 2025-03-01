import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
import tempfile
import random

def generate_test_datasets(num_files: int = 3, 
                          rows_per_file: int = 100,
                          common_columns: int = 2,
                          unique_columns: int = 3,
                          correlation_strength: float = 0.7) -> Tuple[List[str], List[pd.DataFrame]]:

    
    
    np.random.seed(42)  # For reproducibility
    
    # Generate common column names
    common_col_names = [f"common_col_{i}" for i in range(common_columns)]
    
    # Generate common data - we'll use this as a base for all files
    # One column will be the key, others will have some correlation
    common_data = {}
    common_data[common_col_names[0]] = np.arange(rows_per_file)  # Key column
    
    # Generate correlated data for other common columns
    for i in range(1, common_columns):
        # Base value plus random noise
        common_data[common_col_names[i]] = (
            common_data[common_col_names[0]] * correlation_strength + 
            np.random.normal(0, 1-correlation_strength, rows_per_file)
        )
    
    # Create dataframes with common + unique columns
    dfs = []
    file_paths = []
    
    for i in range(num_files):
        # Start with common data
        df_data = common_data.copy()
        
        # Add unique columns
        for j in range(unique_columns):
            col_name = f"file{i}_unique_{j}"
            if j == 0:
                # First unique column has some correlation with the key
                df_data[col_name] = (
                    common_data[common_col_names[0]] * (correlation_strength - 0.1*i) + 
                    np.random.normal(0, 1, rows_per_file)
                )
            else:
                # Other columns are random
                df_data[col_name] = np.random.normal(i, 1, rows_per_file)
        
        # Create DataFrame
        df = pd.DataFrame(df_data)
        
        # Shuffle rows to make it more realistic
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Add a few categorical columns with correlations to numeric data
        df['category'] = pd.cut(df[common_col_names[0]], bins=5, labels=['A', 'B', 'C', 'D', 'E'])
        df['binary'] = np.where(df[common_col_names[0]] > rows_per_file/2, 'Yes', 'No')
        
        # Add a date column that correlates with the key
        start_date = pd.Timestamp('2020-01-01')
        df['date'] = start_date + pd.to_timedelta(df[common_col_names[0]], unit='D')
        
        dfs.append(df)
        
        # Save to temporary CSV file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
        os.close(temp_fd)
        df.to_csv(temp_path, index=False)
        file_paths.append(temp_path)
    
    return file_paths, dfs

def cleanup_test_files(file_paths: List[str]) -> None:
   
    #Clean up temporary test files.
    
  
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing file {file_path}: {str(e)}") 