from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataMerger:
    def __init__(self):
        """Initialize data merger with merge strategies and rules."""
        self.merge_strategies = {
            'inner': pd.merge,
            'outer': lambda left, right, **kwargs: pd.merge(
                left, right, how='outer', **kwargs),
            'left': lambda left, right, **kwargs: pd.merge(
                left, right, how='left', **kwargs),
            'right': lambda left, right, **kwargs: pd.merge(
                left, right, how='right', **kwargs)
        }
        
        self.common_keys = [
            'institution_id',
            'year',
            'metric_type'
        ]

    def merge_dataframes(self, 
                        dfs: List[pd.DataFrame], 
                        merge_on: Optional[Union[str, List[str]]] = None,
                        strategy: str = 'inner') -> Tuple[Optional[pd.DataFrame], Dict]:
        """Merge multiple dataframes using specified strategy."""
        try:
            if not dfs:
                raise ValueError("No dataframes provided")
            
            if len(dfs) == 1:
                return dfs[0], {'status': 'success', 'message': 'Single dataframe provided'}
            
            # Determine merge keys if not specified
            if merge_on is None:
                merge_on = self._identify_common_keys(dfs)
                if not merge_on:
                    raise ValueError("No common keys found between dataframes")
            
            if isinstance(merge_on, str):
                merge_on = [merge_on]
            
            # Validate merge keys
            for df in dfs:
                missing_keys = [key for key in merge_on if key not in df.columns]
                if missing_keys:
                    raise ValueError(f"Missing merge keys in dataframe: {missing_keys}")
            
            # Perform merge
            if strategy not in self.merge_strategies:
                raise ValueError(f"Invalid merge strategy: {strategy}")
            
            result = dfs[0]
            for i, df in enumerate(dfs[1:], 1):
                result = self.merge_strategies[strategy](
                    result, df, on=merge_on)
                
                logger.info(f"Merged dataframe {i+1}/{len(dfs)}")
            
            merge_info = {
                'status': 'success',
                'merged_count': len(dfs),
                'result_shape': result.shape,
                'merge_keys': merge_on,
                'strategy': strategy
            }
            
            return result, merge_info
            
        except Exception as e:
            logger.error(f"Error merging dataframes: {str(e)}")
            return None, {'status': 'error', 'message': str(e)}

    def _identify_common_keys(self, dfs: List[pd.DataFrame]) -> List[str]:
        """Identify common columns that could be used as merge keys."""
        if not dfs:
            return []
        
        # Get common columns across all dataframes
        common_columns = set(dfs[0].columns)
        for df in dfs[1:]:
            common_columns = common_columns.intersection(df.columns)
        
        # Prioritize known common keys
        merge_keys = [key for key in self.common_keys if key in common_columns]
        
        return merge_keys if merge_keys else list(common_columns)

    def validate_merge_compatibility(self, 
                                  dfs: List[pd.DataFrame], 
                                  merge_on: Optional[Union[str, List[str]]] = None) -> Dict:
        """Validate if dataframes can be merged successfully."""
        try:
            validation = {
                'can_merge': True,
                'warnings': [],
                'suggested_keys': [],
                'potential_issues': []
            }
            
            if not dfs:
                validation['can_merge'] = False
                validation['potential_issues'].append("No dataframes provided")
                return validation
            
            if len(dfs) == 1:
                validation['warnings'].append("Single dataframe provided, no merge needed")
                return validation
            
            # Identify merge keys if not specified
            if merge_on is None:
                suggested_keys = self._identify_common_keys(dfs)
                if not suggested_keys:
                    validation['can_merge'] = False
                    validation['potential_issues'].append("No common columns found")
                    return validation
                validation['suggested_keys'] = suggested_keys
                merge_on = suggested_keys
            
            if isinstance(merge_on, str):
                merge_on = [merge_on]
            
            # Check for missing keys
            for i, df in enumerate(dfs):
                missing_keys = [key for key in merge_on if key not in df.columns]
                if missing_keys:
                    validation['can_merge'] = False
                    validation['potential_issues'].append(
                        f"Dataframe {i} missing keys: {missing_keys}")
            
            # Check for duplicate values in merge keys
            for i, df in enumerate(dfs):
                for key in merge_on:
                    if key in df.columns and df[key].duplicated().any():
                        validation['warnings'].append(
                            f"Dataframe {i} has duplicate values in {key}")
            
            # Check for data type compatibility
            for key in merge_on:
                dtypes = [df[key].dtype for df in dfs if key in df.columns]
                if len(set(dtypes)) > 1:
                    validation['warnings'].append(
                        f"Inconsistent data types for {key}: {dtypes}")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating merge compatibility: {str(e)}")
            return {
                'can_merge': False,
                'warnings': [],
                'suggested_keys': [],
                'potential_issues': [str(e)]
            }

    def suggest_merge_strategy(self, dfs: List[pd.DataFrame]) -> Dict:
        """Suggest appropriate merge strategy based on data characteristics."""
        try:
            suggestion = {
                'strategy': 'inner',
                'merge_keys': [],
                'rationale': []
            }
            
            if not dfs or len(dfs) == 1:
                return suggestion
            
            # Identify common keys
            common_keys = self._identify_common_keys(dfs)
            suggestion['merge_keys'] = common_keys
            
            # Check for missing values
            missing_values = [df.isnull().sum().sum() for df in dfs]
            has_missing = any(mv > 0 for mv in missing_values)
            
            # Check for size differences
            sizes = [len(df) for df in dfs]
            size_ratio = max(sizes) / min(sizes)
            
            # Make strategy suggestion
            if has_missing and size_ratio > 2:
                suggestion['strategy'] = 'outer'
                suggestion['rationale'].append(
                    "Significant missing values and size differences detected")
            elif has_missing:
                suggestion['strategy'] = 'left'
                suggestion['rationale'].append(
                    "Missing values detected, preserving primary dataset")
            elif size_ratio > 2:
                suggestion['strategy'] = 'left'
                suggestion['rationale'].append(
                    "Significant size differences between datasets")
            else:
                suggestion['strategy'] = 'inner'
                suggestion['rationale'].append(
                    "Datasets appear compatible for inner join")
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error suggesting merge strategy: {str(e)}")
            return {
                'strategy': 'inner',
                'merge_keys': [],
                'rationale': [f"Error: {str(e)}"]
            }

    def generate_merge_report(self, 
                            original_dfs: List[pd.DataFrame],
                            merged_df: pd.DataFrame,
                            merge_info: Dict) -> Dict:
        """Generate a detailed report about the merge operation."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'merge_info': merge_info,
                'data_summary': {
                    'original': [
                        {
                            'shape': df.shape,
                            'columns': list(df.columns),
                            'missing_values': df.isnull().sum().to_dict()
                        }
                        for df in original_dfs
                    ],
                    'merged': {
                        'shape': merged_df.shape,
                        'columns': list(merged_df.columns),
                        'missing_values': merged_df.isnull().sum().to_dict()
                    }
                },
                'validation': self.validate_merge_compatibility(
                    original_dfs, merge_info.get('merge_keys'))
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating merge report: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            } 