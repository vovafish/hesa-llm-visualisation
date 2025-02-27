from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

from .data_merger import DataMerger
from .data_transformer import DataTransformer
from .data_aggregator import DataAggregator

logger = logging.getLogger(__name__)

class IntegrationManager:
    def __init__(self):
        """Initialize integration manager with its component classes."""
        self.merger = DataMerger()
        self.transformer = DataTransformer()
        self.aggregator = DataAggregator()
        
        self.integration_history = []

    def integrate_datasets(self,
                         datasets: List[pd.DataFrame],
                         merge_config: Dict[str, Any],
                         transform_config: Optional[List[Dict[str, Any]]] = None,
                         aggregation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Integrate multiple datasets using specified configurations.
        
        Args:
            datasets: List of pandas DataFrames to integrate
            merge_config: Configuration for merging datasets
            transform_config: Optional configuration for data transformations
            aggregation_config: Optional configuration for data aggregation
            
        Returns:
            Dictionary containing:
                - result: The integrated DataFrame
                - merge_info: Information about the merge operation
                - transform_info: Information about transformations
                - aggregation_info: Information about aggregations
                - validation: Validation results
                - summary: Summary statistics
        """
        try:
            result = {
                'status': 'success',
                'result': None,
                'merge_info': None,
                'transform_info': None,
                'aggregation_info': None,
                'validation': None,
                'summary': None,
                'errors': []
            }
            
            # Validate inputs
            if not datasets:
                raise ValueError("No datasets provided")
            
            # Step 1: Merge datasets
            try:
                merged_df, merge_info = self.merger.merge_dataframes(
                    datasets,
                    merge_on=merge_config.get('merge_on'),
                    strategy=merge_config.get('strategy', 'inner')
                )
                result['merge_info'] = merge_info
                
                if merged_df is None:
                    raise ValueError("Merge operation failed")
                
            except Exception as e:
                logger.error(f"Error during merge: {str(e)}")
                result['status'] = 'error'
                result['errors'].append(f"Merge error: {str(e)}")
                return result
            
            # Step 2: Apply transformations if specified
            transform_info = []
            if transform_config:
                try:
                    # Validate transformation config
                    validation = self.transformer.validate_transformation_config(
                        transform_config)
                    
                    if not validation['is_valid']:
                        logger.warning("Invalid transformation configuration")
                        result['errors'].extend(validation['errors'])
                    
                    # Apply transformations
                    merged_df = self.transformer.transform_data(
                        merged_df, transform_config)
                    
                    result['transform_info'] = {
                        'applied_transformations': transform_config,
                        'validation': validation
                    }
                    
                except Exception as e:
                    logger.error(f"Error during transformation: {str(e)}")
                    result['errors'].append(f"Transform error: {str(e)}")
            
            # Step 3: Apply aggregations if specified
            if aggregation_config:
                try:
                    # Validate aggregation config
                    validation = self.aggregator.validate_aggregation_config(
                        merged_df,
                        aggregation_config.get('group_by', []),
                        aggregation_config.get('agg_columns', {})
                    )
                    
                    if not validation['is_valid']:
                        logger.warning("Invalid aggregation configuration")
                        result['errors'].extend(validation['errors'])
                    
                    # Apply aggregation
                    merged_df = self.aggregator.aggregate_data(
                        merged_df,
                        group_by=aggregation_config.get('group_by', []),
                        agg_columns=aggregation_config.get('agg_columns', {}),
                        include_count=aggregation_config.get('include_count', True)
                    )
                    
                    result['aggregation_info'] = {
                        'config': aggregation_config,
                        'validation': validation
                    }
                    
                except Exception as e:
                    logger.error(f"Error during aggregation: {str(e)}")
                    result['errors'].append(f"Aggregation error: {str(e)}")
            
            # Generate summary statistics
            try:
                summary = self.aggregator.generate_summary_statistics(merged_df)
                result['summary'] = summary.to_dict()
            except Exception as e:
                logger.error(f"Error generating summary: {str(e)}")
                result['errors'].append(f"Summary error: {str(e)}")
            
            # Store result and update history
            result['result'] = merged_df
            self._update_integration_history(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in integrate_datasets: {str(e)}")
            return {
                'status': 'error',
                'result': None,
                'errors': [str(e)]
            }

    def _update_integration_history(self, integration_result: Dict[str, Any]) -> None:
        """Update the integration history with latest operation."""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'status': integration_result['status'],
            'dataset_shape': integration_result['result'].shape if integration_result['result'] is not None else None,
            'merge_info': integration_result.get('merge_info'),
            'transform_info': integration_result.get('transform_info'),
            'aggregation_info': integration_result.get('aggregation_info'),
            'errors': integration_result.get('errors', [])
        }
        
        self.integration_history.append(history_entry)

    def get_integration_history(self) -> List[Dict[str, Any]]:
        """Get the history of integration operations."""
        return self.integration_history

    def validate_integration_config(self,
                                 datasets: List[pd.DataFrame],
                                 merge_config: Dict[str, Any],
                                 transform_config: Optional[List[Dict[str, Any]]] = None,
                                 aggregation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate all integration configurations before execution."""
        validation = {
            'is_valid': True,
            'merge_validation': None,
            'transform_validation': None,
            'aggregation_validation': None,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate merge configuration
            if not datasets:
                validation['is_valid'] = False
                validation['errors'].append("No datasets provided")
            else:
                merge_validation = self.merger.validate_merge_compatibility(
                    datasets, merge_config.get('merge_on'))
                validation['merge_validation'] = merge_validation
                
                if not merge_validation['can_merge']:
                    validation['is_valid'] = False
                    validation['errors'].extend(merge_validation['potential_issues'])
            
            # Validate transformation configuration
            if transform_config:
                transform_validation = self.transformer.validate_transformation_config(
                    transform_config)
                validation['transform_validation'] = transform_validation
                
                if not transform_validation['is_valid']:
                    validation['is_valid'] = False
                    validation['errors'].extend(transform_validation['errors'])
                validation['warnings'].extend(transform_validation['warnings'])
            
            # Validate aggregation configuration
            if aggregation_config and len(datasets) > 0:
                # Use first dataset for validation as we can't access merged result yet
                aggregation_validation = self.aggregator.validate_aggregation_config(
                    datasets[0],
                    aggregation_config.get('group_by', []),
                    aggregation_config.get('agg_columns', {})
                )
                validation['aggregation_validation'] = aggregation_validation
                
                if not aggregation_validation['is_valid']:
                    validation['is_valid'] = False
                    validation['errors'].extend(aggregation_validation['errors'])
                validation['warnings'].extend(aggregation_validation['warnings'])
            
            return validation
            
        except Exception as e:
            logger.error(f"Error in validate_integration_config: {str(e)}")
            return {
                'is_valid': False,
                'errors': [str(e)],
                'warnings': []
            }

    def suggest_integration_strategy(self,
                                  datasets: List[pd.DataFrame]) -> Dict[str, Any]:
        """Suggest integration strategy based on dataset characteristics."""
        try:
            suggestion = {
                'merge_strategy': None,
                'transformations': [],
                'aggregations': None,
                'rationale': []
            }
            
            if not datasets:
                return suggestion
            
            # Analyze datasets for merge strategy
            merge_suggestion = self.merger.suggest_merge_strategy(datasets)
            suggestion['merge_strategy'] = {
                'strategy': merge_suggestion['strategy'],
                'merge_keys': merge_suggestion['merge_keys']
            }
            suggestion['rationale'].extend(merge_suggestion['rationale'])
            
            # Suggest transformations
            for df in datasets:
                # Check for date columns that might need standardization
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) > 0:
                    suggestion['transformations'].append({
                        'type': 'standardize_dates',
                        'columns': list(date_cols),
                        'params': {'target_format': '%Y-%m-%d'}
                    })
                    suggestion['rationale'].append(
                        f"Found {len(date_cols)} date columns that may need standardization")
                
                # Check for numeric columns that might need normalization
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    suggestion['transformations'].append({
                        'type': 'standardize_numbers',
                        'columns': list(numeric_cols),
                        'params': {'decimal_places': 2}
                    })
                    suggestion['rationale'].append(
                        f"Found {len(numeric_cols)} numeric columns that may need standardization")
                
                # Check for categorical columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                if len(categorical_cols) > 0:
                    suggestion['transformations'].append({
                        'type': 'standardize_text',
                        'columns': list(categorical_cols),
                        'params': {'case': 'lower', 'strip_whitespace': True}
                    })
                    suggestion['rationale'].append(
                        f"Found {len(categorical_cols)} text columns that may need standardization")
            
            # Suggest aggregations if multiple records per key
            if merge_suggestion['merge_keys']:
                key_counts = datasets[0].groupby(
                    merge_suggestion['merge_keys']).size()
                if (key_counts > 1).any():
                    suggestion['aggregations'] = {
                        'group_by': merge_suggestion['merge_keys'],
                        'agg_columns': {
                            col: ['mean', 'sum'] for col in datasets[0].select_dtypes(
                                include=[np.number]).columns
                        },
                        'include_count': True
                    }
                    suggestion['rationale'].append(
                        "Multiple records per key detected, suggesting aggregation")
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error in suggest_integration_strategy: {str(e)}")
            return {
                'merge_strategy': None,
                'transformations': [],
                'aggregations': None,
                'rationale': [f"Error generating suggestions: {str(e)}"]
            } 