from typing import Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataAggregator:
    def __init__(self):
        """Initialize data aggregator with common aggregation functions."""
        self.aggregation_functions = {
            'sum': np.sum,
            'mean': np.mean,
            'median': np.median,
            'min': np.min,
            'max': np.max,
            'count': len,
            'std': np.std,
            'var': np.var,
            'first': lambda x: x.iloc[0],
            'last': lambda x: x.iloc[-1]
        }
        
        self.time_periods = {
            'yearly': 'Y',
            'monthly': 'M',
            'weekly': 'W',
            'daily': 'D'
        }

    def aggregate_data(self,
                      df: pd.DataFrame,
                      group_by: Union[str, List[str]],
                      agg_columns: Dict[str, Union[str, List[str]]],
                      include_count: bool = True) -> pd.DataFrame:
        """Aggregate data based on specified grouping and aggregation functions."""
        try:
            # Validate inputs
            if not isinstance(group_by, (str, list)):
                raise ValueError("group_by must be a string or list of strings")
            
            if isinstance(group_by, str):
                group_by = [group_by]
            
            # Check if all group_by columns exist
            missing_cols = [col for col in group_by if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing grouping columns: {missing_cols}")
            
            # Prepare aggregation dictionary
            agg_dict = {}
            for col, funcs in agg_columns.items():
                if col not in df.columns:
                    logger.warning(f"Column {col} not found in dataframe")
                    continue
                
                if isinstance(funcs, str):
                    funcs = [funcs]
                
                # Validate and convert function names
                valid_funcs = []
                for func in funcs:
                    if func in self.aggregation_functions:
                        valid_funcs.append(self.aggregation_functions[func])
                    else:
                        logger.warning(f"Unknown aggregation function: {func}")
                
                if valid_funcs:
                    agg_dict[col] = valid_funcs
            
            # Perform grouping and aggregation
            result = df.groupby(group_by).agg(agg_dict)
            
            # Add count if requested
            if include_count:
                count_col = df.groupby(group_by).size()
                result['count'] = count_col
            
            # Flatten column names if multiple aggregations per column
            result.columns = ['_'.join(col).strip() 
                            if isinstance(col, tuple) else col 
                            for col in result.columns]
            
            return result.reset_index()
            
        except Exception as e:
            logger.error(f"Error in aggregate_data: {str(e)}")
            raise

    def time_based_aggregation(self,
                             df: pd.DataFrame,
                             time_column: str,
                             period: str,
                             agg_columns: Dict[str, Union[str, List[str]]],
                             include_count: bool = True) -> pd.DataFrame:
        """Aggregate data based on time periods."""
        try:
            # Validate time column
            if time_column not in df.columns:
                raise ValueError(f"Time column {time_column} not found in dataframe")
            
            # Convert time column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                df[time_column] = pd.to_datetime(df[time_column])
            
            # Validate period
            if period not in self.time_periods:
                raise ValueError(f"Invalid time period: {period}. "
                              f"Valid periods are: {list(self.time_periods.keys())}")
            
            # Create time-based groups
            time_groups = df.groupby(pd.Grouper(
                key=time_column, 
                freq=self.time_periods[period]))
            
            # Prepare aggregation dictionary
            agg_dict = {}
            for col, funcs in agg_columns.items():
                if col not in df.columns:
                    logger.warning(f"Column {col} not found in dataframe")
                    continue
                
                if isinstance(funcs, str):
                    funcs = [funcs]
                
                valid_funcs = []
                for func in funcs:
                    if func in self.aggregation_functions:
                        valid_funcs.append(self.aggregation_functions[func])
                    else:
                        logger.warning(f"Unknown aggregation function: {func}")
                
                if valid_funcs:
                    agg_dict[col] = valid_funcs
            
            # Perform aggregation
            result = time_groups.agg(agg_dict)
            
            # Add count if requested
            if include_count:
                count_col = time_groups.size()
                result['count'] = count_col
            
            # Flatten column names
            result.columns = ['_'.join(col).strip() 
                            if isinstance(col, tuple) else col 
                            for col in result.columns]
            
            return result.reset_index()
            
        except Exception as e:
            logger.error(f"Error in time_based_aggregation: {str(e)}")
            raise

    def rolling_aggregation(self,
                          df: pd.DataFrame,
                          window_size: int,
                          agg_columns: Dict[str, Union[str, List[str]]],
                          min_periods: Optional[int] = None) -> pd.DataFrame:
        """Calculate rolling aggregations with specified window size."""
        try:
            result = df.copy()
            
            # Validate window size
            if window_size < 1:
                raise ValueError("Window size must be positive")
            
            # Set minimum periods if not specified
            if min_periods is None:
                min_periods = window_size
            
            # Process each column
            for col, funcs in agg_columns.items():
                if col not in df.columns:
                    logger.warning(f"Column {col} not found in dataframe")
                    continue
                
                if isinstance(funcs, str):
                    funcs = [funcs]
                
                # Apply rolling calculations
                rolling = df[col].rolling(
                    window=window_size,
                    min_periods=min_periods)
                
                for func in funcs:
                    if func not in self.aggregation_functions:
                        logger.warning(f"Unknown aggregation function: {func}")
                        continue
                    
                    try:
                        result[f"{col}_rolling_{func}_{window_size}"] = rolling.apply(
                            self.aggregation_functions[func])
                    except Exception as e:
                        logger.error(f"Error calculating rolling {func} for {col}: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in rolling_aggregation: {str(e)}")
            raise

    def add_custom_aggregation(self,
                             name: str,
                             agg_func: Callable) -> None:
        """Add a custom aggregation function."""
        if name in self.aggregation_functions:
            logger.warning(f"Overwriting existing aggregation function: {name}")
        self.aggregation_functions[name] = agg_func

    def get_available_aggregations(self) -> Dict[str, str]:
        """Get list of available aggregation functions with descriptions."""
        return {
            'sum': 'Calculate the sum of values',
            'mean': 'Calculate the arithmetic mean',
            'median': 'Calculate the median value',
            'min': 'Find the minimum value',
            'max': 'Find the maximum value',
            'count': 'Count the number of records',
            'std': 'Calculate the standard deviation',
            'var': 'Calculate the variance',
            'first': 'Get the first value in the group',
            'last': 'Get the last value in the group'
        }

    def validate_aggregation_config(self,
                                  df: pd.DataFrame,
                                  group_by: Union[str, List[str]],
                                  agg_columns: Dict[str, Union[str, List[str]]]) -> Dict:
        """Validate aggregation configuration."""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate group_by
            if isinstance(group_by, str):
                group_by = [group_by]
            
            if not isinstance(group_by, list):
                validation['is_valid'] = False
                validation['errors'].append("group_by must be a string or list of strings")
            
            # Check grouping columns exist
            missing_group_cols = [col for col in group_by if col not in df.columns]
            if missing_group_cols:
                validation['is_valid'] = False
                validation['errors'].append(
                    f"Missing grouping columns: {missing_group_cols}")
            
            # Validate aggregation columns and functions
            for col, funcs in agg_columns.items():
                if col not in df.columns:
                    validation['warnings'].append(
                        f"Column {col} not found in dataframe")
                    continue
                
                if isinstance(funcs, str):
                    funcs = [funcs]
                
                if not isinstance(funcs, list):
                    validation['is_valid'] = False
                    validation['errors'].append(
                        f"Aggregation functions for {col} must be string or list")
                    continue
                
                invalid_funcs = [f for f in funcs 
                               if f not in self.aggregation_functions]
                if invalid_funcs:
                    validation['warnings'].append(
                        f"Unknown aggregation functions for {col}: {invalid_funcs}")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error in validate_aggregation_config: {str(e)}")
            validation['is_valid'] = False
            validation['errors'].append(str(e))
            return validation

    def generate_summary_statistics(self,
                                  df: pd.DataFrame,
                                  group_by: Optional[Union[str, List[str]]] = None) -> pd.DataFrame:
        """Generate comprehensive summary statistics for the dataframe."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if not numeric_cols.empty:
                if group_by:
                    if isinstance(group_by, str):
                        group_by = [group_by]
                    
                    # Group by specified columns
                    summary = df.groupby(group_by)[numeric_cols].agg([
                        'count', 'mean', 'std', 'min',
                        lambda x: x.quantile(0.25),
                        'median',
                        lambda x: x.quantile(0.75),
                        'max'
                    ])
                    
                    # Rename quantile columns
                    summary = summary.rename(columns={
                        '<lambda_0>': 'q25',
                        '<lambda_1>': 'q75'
                    })
                else:
                    # Calculate overall summary
                    summary = df[numeric_cols].agg([
                        'count', 'mean', 'std', 'min',
                        lambda x: x.quantile(0.25),
                        'median',
                        lambda x: x.quantile(0.75),
                        'max'
                    ]).T
                    
                    summary = summary.rename(columns={
                        '<lambda_0>': 'q25',
                        '<lambda_1>': 'q75'
                    })
                
                return summary
            else:
                logger.warning("No numeric columns found in dataframe")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in generate_summary_statistics: {str(e)}")
            raise 