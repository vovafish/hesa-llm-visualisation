from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        """Initialize data cleaner with cleaning rules and mappings."""
        self.string_replacements = {
            'n/a': np.nan,
            'N/A': np.nan,
            'null': np.nan,
            'NULL': np.nan,
            '': np.nan
        }
        
        self.column_name_mappings = {
            'institution': 'institution_id',
            'inst_id': 'institution_id',
            'academic_year': 'year',
            'year_academic': 'year',
            'metric': 'metric_type'
        }
        
        self.metric_type_mappings = {
            'pg': 'postgraduate',
            'ug': 'undergraduate',
            'ft': 'full_time',
            'pt': 'part_time',
            'intl': 'international',
            'dom': 'domestic'
        }

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows while preserving the most recent/complete data."""
        try:
            # Sort by completeness (fewer null values first)
            completeness = df.isnull().sum(axis=1)
            df['_completeness'] = completeness
            
            # Remove exact duplicates, keeping most complete row
            df = df.sort_values('_completeness').drop_duplicates(
                subset=[col for col in df.columns if col != '_completeness'],
                keep='first'
            )
            
            df = df.drop('_completeness', axis=1)
            return df
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}")
            return df

    def standardize_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and data formats."""
        try:
            # Standardize column names
            df = df.rename(columns=self.column_name_mappings)
            
            # Standardize metric types
            if 'metric_type' in df.columns:
                df['metric_type'] = df['metric_type'].str.lower().map(
                    self.metric_type_mappings).fillna(df['metric_type'])
            
            # Standardize year format
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'].astype(str).str[:4], errors='coerce')
            
            # Standardize institution IDs (remove spaces, convert to uppercase)
            if 'institution_id' in df.columns:
                df['institution_id'] = df['institution_id'].str.strip().str.upper()
            
            return df
            
        except Exception as e:
            logger.error(f"Error standardizing formats: {str(e)}")
            return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type and context."""
        try:
            # Replace string variants of null values
            df = df.replace(self.string_replacements)
            
            # Handle missing values based on column type
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            string_columns = df.select_dtypes(include=['object']).columns
            
            # Fill numeric missing values with 0
            df[numeric_columns] = df[numeric_columns].fillna(0)
            
            # Fill string missing values with 'Unknown'
            df[string_columns] = df[string_columns].fillna('Unknown')
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df

    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and validate data values."""
        try:
            # Ensure numeric columns are properly typed
            if 'value' in df.columns:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce')
            
            # Remove rows with invalid numeric values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df = df.dropna(subset=numeric_columns)
            
            # Ensure no negative values in appropriate columns
            if 'value' in df.columns:
                df.loc[df['value'] < 0, 'value'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps to the dataframe."""
        try:
            df = self.remove_duplicates(df)
            df = self.standardize_formats(df)
            df = self.handle_missing_values(df)
            df = self.normalize_data(df)
            return df
            
        except Exception as e:
            logger.error(f"Error in cleaning pipeline: {str(e)}")
            return df

    def generate_cleaning_report(self, df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        """Generate a report of the cleaning process and its effects."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'original_shape': df.shape,
                'cleaned_shape': cleaned_df.shape,
                'rows_removed': len(df) - len(cleaned_df),
                'null_values_handled': {
                    'before': df.isnull().sum().to_dict(),
                    'after': cleaned_df.isnull().sum().to_dict()
                },
                'duplicates_removed': len(df) - len(df.drop_duplicates()),
                'memory_usage': {
                    'before': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                    'after': cleaned_df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                }
            }
            return report
            
        except Exception as e:
            logger.error(f"Error generating cleaning report: {str(e)}")
            return {} 