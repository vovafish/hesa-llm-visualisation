import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import logging
from datetime import datetime
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationRules:
    """
    A comprehensive data validation system for HESA datasets.
    
    This class provides configurable validation rules that can be applied to
    dataframes containing HESA data. It supports various types of validation
    including:
    - Data type validation
    - Range validation
    - Format validation
    - Cross-field validations
    - Consistency checks
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ValidationRules with optional configuration.
        
        """
        # Default data type rules
        self.data_type_rules = {
            'institution': str,
            'institution_id': str,
            'year': int,
            'academic_year': str,
            'subject': str,
            'subject_code': str,
            'metric_type': str,
            'count': int,
            'value': float,
            'percentage': float,
            'ratio': float,
            'rank': int,
            'date': 'datetime'
        }
        
        # Default range rules
        self.range_rules = {
            'year': (1900, 2100),
            'count': (0, None),  # None means no upper bound
            'value': (0, None),
            'percentage': (0, 100),
            'ratio': (0, None),
            'rank': (1, None)
        }
        
        # Format validation rules using regex patterns
        self.format_rules = {
            'academic_year': r'^\d{4}/\d{2,4}$',  # e.g., 2021/22 or 2021/2022
            'institution_id': r'^[A-Z0-9]+$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'subject_code': r'^[A-Z0-9]{4,6}$',
        }
        
        # Consistency rules for cross-field validation
        self.consistency_rules = [
            {
                'name': 'year_academic_year_consistency',
                'columns': ['year', 'academic_year'],
                'check': lambda row: str(row['year']) in row['academic_year'].replace('/', '') if pd.notna(row['year']) and pd.notna(row['academic_year']) else True
            },
            {
                'name': 'count_percentage_consistency',
                'columns': ['count', 'total', 'percentage'],
                'check': lambda row: abs((row['count'] / row['total'] * 100) - row['percentage']) < 0.1 if pd.notna(row['count']) and pd.notna(row['total']) and pd.notna(row['percentage']) and row['total'] > 0 else True
            }
        ]
        
        # Required columns for different dataset types
        self.required_columns = {
            'enrollment': ['institution', 'year', 'count'],
            'performance': ['institution', 'metric_type', 'value', 'year'],
            'demographic': ['institution', 'year', 'demographic_type', 'category', 'count'],
            'general': ['institution']  # Fallback for unspecified dataset types
        }
        
        # Override defaults with provided config
        if config:
            if 'data_type_rules' in config:
                self.data_type_rules.update(config['data_type_rules'])
            if 'range_rules' in config:
                self.range_rules.update(config['range_rules'])
            if 'format_rules' in config:
                self.format_rules.update(config['format_rules'])
            if 'consistency_rules' in config:
                self.consistency_rules.extend(config['consistency_rules'])
            if 'required_columns' in config:
                self.required_columns.update(config['required_columns'])
    
    def apply_rules(self, df: pd.DataFrame, dataset_type: str = 'general') -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Apply all validation rules to the dataframe.
       
        """
        errors = []
        
        # Check required columns
        required_cols = self.required_columns.get(dataset_type, self.required_columns['general'])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append({
                'rule': 'required_columns',
                'details': f"Missing required columns for {dataset_type} dataset: {', '.join(missing_cols)}",
                'severity': 'error'
            })
        
        # Proceed with other validations if we have minimum required data
        if not missing_cols or dataset_type == 'general':
            df, type_errors = self.check_data_types(df)
            errors.extend(type_errors)
            
            df, range_errors = self.validate_ranges(df)
            errors.extend(range_errors)
            
            df, format_errors = self.validate_formats(df)
            errors.extend(format_errors)
            
            consistency_errors = self.check_relationships(df)
            errors.extend(consistency_errors)
        
        return df, errors
    
    def check_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Check and attempt to convert columns to their expected data types.
        
        """
        errors = []
        
        for column, expected_type in self.data_type_rules.items():
            if column not in df.columns:
                continue
                
            try:
                # Skip columns that are entirely NaN
                if df[column].isna().all():
                    continue
                    
                if expected_type == int:
                    df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                    invalid_mask = df[column].isna() & df.iloc[:, df.columns.get_loc(column)].notna()
                    if invalid_mask.any():
                        errors.append({
                            'rule': 'data_type',
                            'column': column,
                            'expected': 'integer',
                            'rows': invalid_mask.sum(),
                            'sample': str(df.iloc[:, df.columns.get_loc(column)][invalid_mask].head(3).tolist()),
                            'severity': 'warning'
                        })
                        
                elif expected_type == float:
                    df[column] = pd.to_numeric(df[column], errors='coerce', downcast='float')
                    invalid_mask = df[column].isna() & df.iloc[:, df.columns.get_loc(column)].notna()
                    if invalid_mask.any():
                        errors.append({
                            'rule': 'data_type',
                            'column': column,
                            'expected': 'float',
                            'rows': invalid_mask.sum(),
                            'sample': str(df.iloc[:, df.columns.get_loc(column)][invalid_mask].head(3).tolist()),
                            'severity': 'warning'
                        })
                        
                elif expected_type == str:
                    # Convert to string but handle NaN values
                    df[column] = df[column].astype(str).replace('nan', np.nan)
                    
                elif expected_type == 'datetime':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    invalid_mask = df[column].isna() & df.iloc[:, df.columns.get_loc(column)].notna()
                    if invalid_mask.any():
                        errors.append({
                            'rule': 'data_type',
                            'column': column,
                            'expected': 'datetime',
                            'rows': invalid_mask.sum(),
                            'sample': str(df.iloc[:, df.columns.get_loc(column)][invalid_mask].head(3).tolist()),
                            'severity': 'warning'
                        })
                
            except Exception as e:
                errors.append({
                    'rule': 'data_type',
                    'column': column,
                    'expected': str(expected_type),
                    'error': str(e),
                    'severity': 'error'
                })
        
        return df, errors
    
    def validate_ranges(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Validate that numeric values fall within acceptable ranges.
   
        """
        errors = []
        
        for column, (min_val, max_val) in self.range_rules.items():
            if column not in df.columns:
                continue
            
            try:
                # Skip columns that aren't numeric or are entirely NaN
                if not pd.api.types.is_numeric_dtype(df[column]) or df[column].isna().all():
                    continue
                
                # Check minimum value
                if min_val is not None:
                    below_min = df[column] < min_val
                    if below_min.any():
                        errors.append({
                            'rule': 'range',
                            'column': column,
                            'constraint': f'>= {min_val}',
                            'rows': below_min.sum(),
                            'sample': str(df.loc[below_min, column].head(3).tolist()),
                            'severity': 'warning'
                        })
                        
                        # Optionally, fix values (commented out as this would modify the data)
                        # df.loc[below_min, column] = min_val
                
                # Check maximum value
                if max_val is not None:
                    above_max = df[column] > max_val
                    if above_max.any():
                        errors.append({
                            'rule': 'range',
                            'column': column,
                            'constraint': f'<= {max_val}',
                            'rows': above_max.sum(),
                            'sample': str(df.loc[above_max, column].head(3).tolist()),
                            'severity': 'warning'
                        })
                        
                        # Optionally, fix values (commented out as this would modify the data)
                        # df.loc[above_max, column] = max_val
                
            except Exception as e:
                errors.append({
                    'rule': 'range',
                    'column': column,
                    'error': str(e),
                    'severity': 'error'
                })
        
        return df, errors
    
    def validate_formats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Validate text formats using regex patterns.
        
        """
        errors = []
        
        for column, pattern in self.format_rules.items():
            if column not in df.columns:
                continue
            
            try:
                # Skip columns that are entirely NaN
                if df[column].isna().all():
                    continue
                
                # Only check string columns
                if not pd.api.types.is_string_dtype(df[column]):
                    try:
                        # Try converting to string for validation
                        string_values = df[column].astype(str)
                    except:
                        continue
                else:
                    string_values = df[column]
                
                # Check non-NaN values against pattern
                mask = string_values.notna()
                invalid_format = mask & ~string_values.str.match(pattern)
                
                if invalid_format.any():
                    errors.append({
                        'rule': 'format',
                        'column': column,
                        'pattern': pattern,
                        'rows': invalid_format.sum(),
                        'sample': str(df.loc[invalid_format, column].head(3).tolist()),
                        'severity': 'warning'
                    })
                
            except Exception as e:
                errors.append({
                    'rule': 'format',
                    'column': column,
                    'error': str(e),
                    'severity': 'error'
                })
        
        return df, errors
    
    def check_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """
        Check cross-field relationships and consistency rules.
        """
        errors = []
        
        for rule in self.consistency_rules:
            # Skip if required columns are missing
            if not all(col in df.columns for col in rule['columns']):
                continue
                
            try:
                # Apply the check function to each row
                invalid_rows = ~df.apply(rule['check'], axis=1)
                
                if invalid_rows.any():
                    # Get sample of invalid data
                    sample_rows = df.loc[invalid_rows].head(3)
                    sample_data = {col: sample_rows[col].tolist() for col in rule['columns']}
                    
                    errors.append({
                        'rule': 'consistency',
                        'name': rule['name'],
                        'columns': rule['columns'],
                        'rows': invalid_rows.sum(),
                        'sample': str(sample_data),
                        'severity': 'warning'
                    })
                    
            except Exception as e:
                errors.append({
                    'rule': 'consistency',
                    'name': rule['name'],
                    'columns': rule['columns'],
                    'error': str(e),
                    'severity': 'error'
                })
        
        return errors
    
    def categorize_errors(self, errors: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize errors by severity and rule type.
        
        """
        categorized = {
            'error': [],
            'warning': [],
            'info': [],
            'by_rule': {},
            'by_column': {}
        }
        
        for error in errors:
            severity = error.get('severity', 'error')
            rule = error.get('rule', 'unknown')
            column = error.get('column', 'multiple')
            
            # Add to severity category
            categorized[severity].append(error)
            
            # Add to rule category
            if rule not in categorized['by_rule']:
                categorized['by_rule'][rule] = []
            categorized['by_rule'][rule].append(error)
            
            # Add to column category if applicable
            if column != 'multiple':
                if column not in categorized['by_column']:
                    categorized['by_column'][column] = []
                categorized['by_column'][column].append(error)
        
        return categorized
    
    def generate_validation_report(self, df: pd.DataFrame, dataset_type: str = 'general') -> Dict:
        """
        Generate a comprehensive validation report.
        
        """
        validated_df, errors = self.apply_rules(df, dataset_type)
        categorized_errors = self.categorize_errors(errors)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'rows': len(df),
                'columns': list(df.columns),
                'dataset_type': dataset_type
            },
            'validation_summary': {
                'total_issues': len(errors),
                'errors': len(categorized_errors['error']),
                'warnings': len(categorized_errors['warning']),
                'info': len(categorized_errors['info']),
                'has_critical_issues': len(categorized_errors['error']) > 0
            },
            'issues_by_rule': {
                rule: len(issues) for rule, issues in categorized_errors['by_rule'].items()
            },
            'issues_by_column': {
                col: len(issues) for col, issues in categorized_errors['by_column'].items()
            },
            'all_issues': errors
        }
        
        return report
    
    def validate_and_clean_file(self, file_path: Union[str, Path], dataset_type: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Load a CSV file, validate it, and return cleaned data with validation report.
        
        """
        file_path = Path(file_path) if isinstance(file_path, str) else file_path
        
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Try to infer dataset type if not provided
            if dataset_type is None:
                if 'demographic' in file_path.stem.lower() or 'demographic_type' in df.columns:
                    dataset_type = 'demographic'
                elif 'enrollment' in file_path.stem.lower() or 'count' in df.columns:
                    dataset_type = 'enrollment'
                elif 'performance' in file_path.stem.lower() or 'metric_type' in df.columns:
                    dataset_type = 'performance'
                else:
                    dataset_type = 'general'
            
            # Validate the data
            validated_df, errors = self.apply_rules(df, dataset_type)
            
            # Generate report
            report = self.generate_validation_report(df, dataset_type)
            report['file_info'] = {
                'name': file_path.name,
                'path': str(file_path),
                'size_bytes': file_path.stat().st_size
            }
            
            return validated_df, report
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {str(e)}")
            report = {
                'timestamp': datetime.now().isoformat(),
                'file_info': {
                    'name': file_path.name if isinstance(file_path, Path) else str(file_path),
                    'path': str(file_path)
                },
                'validation_summary': {
                    'has_critical_issues': True
                },
                'error': str(e)
            }
            return pd.DataFrame(), report
            
    def save_validation_report(self, report: Dict, output_path: Union[str, Path]) -> None:
        """
        Save validation report to a JSON file.
        
        """
        import json
        
        output_path = Path(output_path) if isinstance(output_path, str) else output_path
        
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving validation report: {str(e)}") 