from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class CSVValidator:
    def __init__(self):
        """Initialize CSV validator with validation rules."""
        self.required_columns = [
            'institution_id',
            'year',
            'metric_type',
            'value'
        ]
        
        self.data_types = {
            'institution_id': str,
            'year': int,
            'metric_type': str,
            'value': float
        }
        
        self.value_ranges = {
            'year': (1900, 2100),
            'value': (0, float('inf'))
        }

    def validate(self, file_path: str) -> List[Dict]:
        """
        Validate a CSV file and return a list of issues.
        This is a wrapper for validate_file method to maintain compatibility.
        
        Args:
            file_path: Path to the CSV file to validate
            
        Returns:
            List of dictionaries describing validation issues
        """
        validation_result = self.validate_file(Path(file_path))
        if 'issues' in validation_result:
            return validation_result['issues']
        return []

    def check_format(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Check if file is a valid CSV and can be read."""
        try:
            if not file_path.exists():
                return False, "File does not exist"
            
            if file_path.suffix.lower() != '.csv':
                return False, "File is not a CSV"
            
            # Try reading the file
            pd.read_csv(file_path, nrows=1)
            return True, None
            
        except Exception as e:
            return False, f"Error reading CSV: {str(e)}"

    def validate_headers(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate that all required columns are present."""
        try:
            df = pd.read_csv(file_path, nrows=0)
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            
            if missing_columns:
                return False, f"Missing required columns: {', '.join(missing_columns)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Error validating headers: {str(e)}"

    def validate_data_types(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data types of columns."""
        errors = []
        
        for column, expected_type in self.data_types.items():
            if column not in df.columns:
                continue
                
            try:
                if expected_type == int:
                    pd.to_numeric(df[column], downcast='integer')
                elif expected_type == float:
                    pd.to_numeric(df[column], downcast='float')
                elif expected_type == str:
                    df[column].astype(str)
            except Exception as e:
                errors.append(f"Column {column} has invalid data type: {str(e)}")
        
        return len(errors) == 0, errors

    def validate_value_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that values are within acceptable ranges."""
        errors = []
        
        for column, (min_val, max_val) in self.value_ranges.items():
            if column not in df.columns:
                continue
                
            try:
                values = pd.to_numeric(df[column])
                if (values < min_val).any():
                    errors.append(f"Column {column} contains values below minimum {min_val}")
                if (values > max_val).any():
                    errors.append(f"Column {column} contains values above maximum {max_val}")
            except Exception as e:
                errors.append(f"Error validating range for column {column}: {str(e)}")
        
        return len(errors) == 0, errors

    def validate_file(self, file_path: Path) -> Dict:
        """Validate entire CSV file and return validation results."""
        results = {
            'valid': False,
            'errors': [],
            'warnings': []
        }
        
        # Check file format
        format_valid, format_error = self.check_format(file_path)
        if not format_valid:
            results['errors'].append(format_error)
            return results
        
        # Validate headers
        headers_valid, headers_error = self.validate_headers(file_path)
        if not headers_valid:
            results['errors'].append(headers_error)
            return results
        
        # Read the full file for data validation
        try:
            df = pd.read_csv(file_path)
            
            # Validate data types
            types_valid, type_errors = self.validate_data_types(df)
            if not types_valid:
                results['errors'].extend(type_errors)
            
            # Validate value ranges
            ranges_valid, range_errors = self.validate_value_ranges(df)
            if not ranges_valid:
                results['errors'].extend(range_errors)
            
            # Set overall validation result
            results['valid'] = len(results['errors']) == 0
            
        except Exception as e:
            results['errors'].append(f"Error during validation: {str(e)}")
        
        return results

    def generate_validation_report(self, file_path: Path) -> str:
        """Generate a detailed validation report in JSON format."""
        validation_results = self.validate_file(file_path)
        
        report = {
            'file_name': file_path.name,
            'validation_time': pd.Timestamp.now().isoformat(),
            'validation_results': validation_results,
            'file_stats': self._get_file_stats(file_path) if validation_results['valid'] else None
        }
        
        return json.dumps(report, indent=2)

    def _get_file_stats(self, file_path: Path) -> Dict:
        """Get basic statistics about the CSV file."""
        try:
            df = pd.read_csv(file_path)
            return {
                'row_count': len(df),
                'column_count': len(df.columns),
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # in MB
            }
        except Exception as e:
            logger.error(f"Error getting file stats: {str(e)}")
            return {} 