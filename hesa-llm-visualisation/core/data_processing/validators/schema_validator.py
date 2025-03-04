from typing import Dict, List, Optional, Union
import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SchemaValidator:
    def __init__(self):
        """Initialize schema validator with validation rules."""
        self.schema = {
            'institution_id': {
                'type': str,
                'required': True,
                'unique': False
            },
            'year': {
                'type': int,
                'required': True,
                'range': (1900, 2100),
                'unique': False
            },
            'metric_type': {
                'type': str,
                'required': True,
                'allowed_values': [
                    'postgraduate',
                    'undergraduate',
                    'full_time',
                    'part_time',
                    'international',
                    'domestic',
                    'research',
                    'teaching'
                ],
                'unique': False
            },
            'value': {
                'type': float,
                'required': True,
                'range': (0, float('inf')),
                'unique': False
            }
        }
        
        self.relationships = {
            'primary_key': ['institution_id', 'year', 'metric_type'],
            'foreign_keys': {}
        }

    def validate_dataframe(self, df: pd.DataFrame, file_name: str = None) -> Dict[str, any]:
        """
        Validate a DataFrame against the schema and return validation results.
        
        Args:
            df (pd.DataFrame): The DataFrame to validate
            file_name (str, optional): Name of the file being validated
            
        Returns:
            Dict[str, any]: Validation results containing:
                - is_valid (bool): Whether the DataFrame is valid
                - errors (List[str]): List of validation errors
                - warnings (List[str]): List of validation warnings
                - summary (Dict): Summary of validation results
        """
        # Initialize validation results
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {
                'total_issues': 0,
                'critical_issues': 0,
                'warnings': 0,
                'fixed_issues': 0
            }
        }
        
        try:
            # Add file name to results if provided
            if file_name:
                validation_results['file_name'] = file_name
            
            # Validate schema
            schema_results = self.validate_schema(df)
            if not schema_results.get('is_valid', False):
                validation_results['errors'].extend(schema_results.get('errors', []))
                validation_results['is_valid'] = False
            
            # Update summary
            validation_results['summary']['total_issues'] = len(validation_results['errors']) + len(validation_results['warnings'])
            validation_results['summary']['critical_issues'] = len(validation_results['errors'])
            validation_results['summary']['warnings'] = len(validation_results['warnings'])
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Unexpected error during validation: {str(e)}")
            validation_results['summary']['critical_issues'] += 1
            validation_results['summary']['total_issues'] += 1
            
        return validation_results

    def validate_schema(self, df: pd.DataFrame) -> Dict:
        """Validate dataframe against defined schema."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        for column, rules in self.schema.items():
            if rules['required'] and column not in df.columns:
                results['valid'] = False
                results['errors'].append(f"Required column '{column}' is missing")
                continue
            
            if column not in df.columns:
                continue
            
            # Validate data type
            try:
                if rules['type'] == int:
                    pd.to_numeric(df[column], downcast='integer')
                elif rules['type'] == float:
                    pd.to_numeric(df[column], downcast='float')
                elif rules['type'] == str:
                    df[column].astype(str)
            except Exception as e:
                results['valid'] = False
                results['errors'].append(f"Column '{column}' has invalid data type: {str(e)}")
            
            # Validate value range if specified
            if 'range' in rules and column in df.columns:
                min_val, max_val = rules['range']
                try:
                    values = pd.to_numeric(df[column])
                    if (values < min_val).any():
                        results['errors'].append(
                            f"Column '{column}' contains values below minimum {min_val}")
                    if (values > max_val).any():
                        results['errors'].append(
                            f"Column '{column}' contains values above maximum {max_val}")
                except Exception as e:
                    results['errors'].append(
                        f"Error validating range for column '{column}': {str(e)}")
            
            # Validate allowed values if specified
            if 'allowed_values' in rules and column in df.columns:
                invalid_values = df[df[column].notna()][
                    ~df[column].isin(rules['allowed_values'])][column].unique()
                if len(invalid_values) > 0:
                    results['errors'].append(
                        f"Column '{column}' contains invalid values: {invalid_values}")
            
            # Check uniqueness if required
            if rules.get('unique', False) and column in df.columns:
                if df[column].duplicated().any():
                    results['errors'].append(f"Column '{column}' contains duplicate values")
        
        results['valid'] = len(results['errors']) == 0
        return results

    def validate_relationships(self, df: pd.DataFrame, related_dfs: Dict[str, pd.DataFrame]) -> Dict:
        """Validate relationships between dataframes."""
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate primary key
        if self.relationships['primary_key']:
            try:
                pk_columns = self.relationships['primary_key']
                if not all(col in df.columns for col in pk_columns):
                    missing_cols = [col for col in pk_columns if col not in df.columns]
                    results['errors'].append(
                        f"Missing primary key columns: {missing_cols}")
                else:
                    duplicates = df.duplicated(subset=pk_columns)
                    if duplicates.any():
                        results['errors'].append(
                            f"Duplicate entries found for primary key: {pk_columns}")
            except Exception as e:
                results['errors'].append(f"Error validating primary key: {str(e)}")
        
        # Validate foreign keys
        for fk_column, reference in self.relationships['foreign_keys'].items():
            try:
                if fk_column not in df.columns:
                    results['errors'].append(f"Foreign key column '{fk_column}' not found")
                    continue
                
                ref_table, ref_column = reference.split('.')
                if ref_table not in related_dfs:
                    results['warnings'].append(
                        f"Referenced table '{ref_table}' not provided for validation")
                    continue
                
                ref_df = related_dfs[ref_table]
                if ref_column not in ref_df.columns:
                    results['errors'].append(
                        f"Referenced column '{ref_column}' not found in '{ref_table}'")
                    continue
                
                invalid_refs = df[~df[fk_column].isin(ref_df[ref_column])][fk_column].unique()
                if len(invalid_refs) > 0:
                    results['errors'].append(
                        f"Invalid foreign key values in '{fk_column}': {invalid_refs}")
            
            except Exception as e:
                results['errors'].append(
                    f"Error validating foreign key '{fk_column}': {str(e)}")
        
        results['valid'] = len(results['errors']) == 0
        return results

    def generate_schema_report(self, df: pd.DataFrame, 
                             related_dfs: Optional[Dict[str, pd.DataFrame]] = None) -> str:
        """Generate a comprehensive schema validation report."""
        schema_validation = self.validate_schema(df)
        relationship_validation = self.validate_relationships(
            df, related_dfs or {}) if related_dfs else {'valid': True, 'errors': [], 'warnings': []}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'file_info': {
                'columns': list(df.columns),
                'row_count': len(df),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'schema_validation': schema_validation,
            'relationship_validation': relationship_validation,
            'overall_valid': schema_validation['valid'] and relationship_validation['valid']
        }
        
        return json.dumps(report, indent=2)

    def export_schema(self, file_path: Path) -> None:
        """Export the current schema definition to a JSON file."""
        try:
            schema_export = {
                'schema': self.schema,
                'relationships': self.relationships,
                'export_time': datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(schema_export, f, indent=2)
                
            logger.info(f"Schema exported successfully to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting schema: {str(e)}")
            raise

    def import_schema(self, file_path: Path) -> None:
        """Import schema definition from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                schema_import = json.load(f)
            
            self.schema = schema_import['schema']
            self.relationships = schema_import['relationships']
            
            logger.info(f"Schema imported successfully from {file_path}")
            
        except Exception as e:
            logger.error(f"Error importing schema: {str(e)}")
            raise 