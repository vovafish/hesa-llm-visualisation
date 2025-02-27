from typing import Dict, List, Optional, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataTransformer:
    def __init__(self):
        """Initialize data transformer with common transformation functions."""
        self.transformations = {
            'standardize_dates': self._standardize_dates,
            'standardize_numbers': self._standardize_numbers,
            'standardize_text': self._standardize_text,
            'handle_missing': self._handle_missing_values,
            'encode_categorical': self._encode_categorical,
            'normalize': self._normalize_data
        }
        
        self.date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%Y'
        ]

    def transform_data(self, 
                      df: pd.DataFrame,
                      transformations: List[Dict[str, Union[str, Dict]]]) -> pd.DataFrame:
        """Apply a series of transformations to the dataframe."""
        try:
            result_df = df.copy()
            transformation_log = []
            
            for transform in transformations:
                transform_type = transform.get('type')
                if transform_type not in self.transformations:
                    logger.warning(f"Unknown transformation type: {transform_type}")
                    continue
                
                columns = transform.get('columns', df.columns)
                params = transform.get('params', {})
                
                try:
                    result_df = self.transformations[transform_type](
                        result_df, columns, **params)
                    
                    transformation_log.append({
                        'type': transform_type,
                        'columns': columns,
                        'params': params,
                        'status': 'success'
                    })
                except Exception as e:
                    logger.error(f"Error applying {transform_type}: {str(e)}")
                    transformation_log.append({
                        'type': transform_type,
                        'columns': columns,
                        'params': params,
                        'status': 'error',
                        'error': str(e)
                    })
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in transform_data: {str(e)}")
            raise

    def _standardize_dates(self, 
                          df: pd.DataFrame,
                          columns: List[str],
                          target_format: str = '%Y-%m-%d') -> pd.DataFrame:
        """Standardize date columns to a consistent format."""
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            try:
                # Try parsing dates with different formats
                for date_format in self.date_formats:
                    try:
                        result[col] = pd.to_datetime(
                            df[col], format=date_format)
                        break
                    except:
                        continue
                
                # Convert to target format
                result[col] = result[col].dt.strftime(target_format)
                
            except Exception as e:
                logger.error(f"Error standardizing dates in column {col}: {str(e)}")
                
        return result

    def _standardize_numbers(self, 
                           df: pd.DataFrame,
                           columns: List[str],
                           decimal_places: int = 2) -> pd.DataFrame:
        """Standardize numeric columns to consistent format."""
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            try:
                # Convert to numeric, coercing errors to NaN
                result[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Round to specified decimal places
                result[col] = result[col].round(decimal_places)
                
            except Exception as e:
                logger.error(f"Error standardizing numbers in column {col}: {str(e)}")
                
        return result

    def _standardize_text(self, 
                         df: pd.DataFrame,
                         columns: List[str],
                         case: str = 'lower',
                         strip_whitespace: bool = True) -> pd.DataFrame:
        """Standardize text columns for consistency."""
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            try:
                # Convert to string
                result[col] = result[col].astype(str)
                
                # Apply case transformation
                if case == 'lower':
                    result[col] = result[col].str.lower()
                elif case == 'upper':
                    result[col] = result[col].str.upper()
                elif case == 'title':
                    result[col] = result[col].str.title()
                
                # Strip whitespace if requested
                if strip_whitespace:
                    result[col] = result[col].str.strip()
                
            except Exception as e:
                logger.error(f"Error standardizing text in column {col}: {str(e)}")
                
        return result

    def _handle_missing_values(self, 
                             df: pd.DataFrame,
                             columns: List[str],
                             strategy: str = 'mean',
                             fill_value: Optional[Union[str, float]] = None) -> pd.DataFrame:
        """Handle missing values in specified columns."""
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            try:
                if strategy == 'drop':
                    result = result.dropna(subset=[col])
                elif strategy == 'mean':
                    result[col] = result[col].fillna(result[col].mean())
                elif strategy == 'median':
                    result[col] = result[col].fillna(result[col].median())
                elif strategy == 'mode':
                    result[col] = result[col].fillna(result[col].mode()[0])
                elif strategy == 'constant':
                    result[col] = result[col].fillna(fill_value)
                elif strategy == 'ffill':
                    result[col] = result[col].ffill()
                elif strategy == 'bfill':
                    result[col] = result[col].bfill()
                
            except Exception as e:
                logger.error(f"Error handling missing values in column {col}: {str(e)}")
                
        return result

    def _encode_categorical(self, 
                          df: pd.DataFrame,
                          columns: List[str],
                          method: str = 'label') -> pd.DataFrame:
        """Encode categorical variables using specified method."""
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            try:
                if method == 'label':
                    result[col] = pd.Categorical(result[col]).codes
                elif method == 'one_hot':
                    # Create one-hot encoded columns
                    one_hot = pd.get_dummies(result[col], prefix=col)
                    # Drop original column and join encoded columns
                    result = result.drop(col, axis=1).join(one_hot)
                
            except Exception as e:
                logger.error(f"Error encoding categorical column {col}: {str(e)}")
                
        return result

    def _normalize_data(self, 
                       df: pd.DataFrame,
                       columns: List[str],
                       method: str = 'minmax') -> pd.DataFrame:
        """Normalize numeric columns using specified method."""
        result = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            try:
                if method == 'minmax':
                    min_val = result[col].min()
                    max_val = result[col].max()
                    result[col] = (result[col] - min_val) / (max_val - min_val)
                elif method == 'zscore':
                    mean = result[col].mean()
                    std = result[col].std()
                    result[col] = (result[col] - mean) / std
                
            except Exception as e:
                logger.error(f"Error normalizing column {col}: {str(e)}")
                
        return result

    def add_custom_transformation(self, 
                                name: str,
                                transform_func: Callable) -> None:
        """Add a custom transformation function to the transformer."""
        if name in self.transformations:
            logger.warning(f"Overwriting existing transformation: {name}")
        self.transformations[name] = transform_func

    def get_available_transformations(self) -> Dict[str, str]:
        """Get list of available transformations with descriptions."""
        return {
            'standardize_dates': 'Standardize date formats across columns',
            'standardize_numbers': 'Standardize numeric values with consistent decimals',
            'standardize_text': 'Standardize text case and whitespace',
            'handle_missing': 'Handle missing values using various strategies',
            'encode_categorical': 'Encode categorical variables using label or one-hot encoding',
            'normalize': 'Normalize numeric columns using min-max or z-score scaling'
        }

    def validate_transformation_config(self, 
                                    transformations: List[Dict[str, Union[str, Dict]]]) -> Dict:
        """Validate transformation configuration before applying."""
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        for transform in transformations:
            # Check required fields
            if 'type' not in transform:
                validation['is_valid'] = False
                validation['errors'].append("Missing 'type' in transformation config")
                continue
            
            # Check transformation type exists
            if transform['type'] not in self.transformations:
                validation['is_valid'] = False
                validation['errors'].append(
                    f"Unknown transformation type: {transform['type']}")
                continue
            
            # Validate parameters if provided
            if 'params' in transform:
                if not isinstance(transform['params'], dict):
                    validation['is_valid'] = False
                    validation['errors'].append(
                        f"Parameters for {transform['type']} must be a dictionary")
            
            # Check columns format if provided
            if 'columns' in transform:
                if not isinstance(transform['columns'], list):
                    validation['warnings'].append(
                        f"'columns' should be a list for {transform['type']}")
        
        return validation 