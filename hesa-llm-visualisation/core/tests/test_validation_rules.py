import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import json
import shutil

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from core.data_processing.validators.validation_rules import ValidationRules

class TestValidationRules(unittest.TestCase):
    """Test cases for the ValidationRules class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple test dataframe with some validation issues
        self.test_data = pd.DataFrame({
            'institution': ['University A', 'University B', None],
            'year': [2021, '2022', 2023],  # Mixed types
            'count': [100, -5, 200],  # Negative value
            'percentage': [50.5, 80.2, 110.5],  # Value over 100
            'academic_year': ['2021/22', '2022-23', '2023/24'],  # Invalid format
            'total': [200, 100, 180]
        })
        
        # Create a test file
        self.test_file_path = Path(self.temp_dir) / 'test_enrollment.csv'
        self.test_data.to_csv(self.test_file_path, index=False)
        
        # Initialize the validator
        self.validator = ValidationRules()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove temporary directory and files
        shutil.rmtree(self.temp_dir)
        
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration"""
        custom_config = {
            'data_type_rules': {'custom_field': float},
            'range_rules': {'custom_field': (0, 10)},
            'format_rules': {'custom_id': r'^ID\d+$'},
            'required_columns': {'custom_type': ['custom_field']}
        }
        
        validator = ValidationRules(config=custom_config)
        
        # Check if the custom config was applied
        self.assertEqual(validator.data_type_rules['custom_field'], float)
        self.assertEqual(validator.range_rules['custom_field'], (0, 10))
        self.assertEqual(validator.format_rules['custom_id'], r'^ID\d+$')
        self.assertEqual(validator.required_columns['custom_type'], ['custom_field'])
        
    def test_check_data_types(self):
        """Test data type validation"""
        df = pd.DataFrame({
            'institution': ['University A', 'University B', 'University C'],
            'year': ['2021', 2022, 'not a year'],
            'count': [100, 200, 'three hundred']
        })
        
        result_df, errors = self.validator.check_data_types(df)
        
        # Check if the function correctly identified type issues
        year_error = next((e for e in errors if e['column'] == 'year'), None)
        count_error = next((e for e in errors if e['column'] == 'count'), None)
        
        # There should be type errors
        self.assertIsNotNone(year_error)
        self.assertIsNotNone(count_error)
        
        # Check if conversion worked where possible
        self.assertEqual(result_df['year'].iloc[0], 2021)  # Should convert '2021' to 2021
        self.assertEqual(result_df['count'].iloc[0], 100)  # Should keep numeric values
        
    def test_validate_ranges(self):
        """Test range validation"""
        df = pd.DataFrame({
            'year': [1800, 2022, 2200],  # Out of range (1900-2100)
            'percentage': [-10, 50, 120],  # Out of range (0-100)
            'count': [-5, 0, 100]  # Out of range (0-None)
        })
        
        result_df, errors = self.validator.validate_ranges(df)
        
        # Check that all range violations were detected
        year_min_error = next((e for e in errors if e['column'] == 'year' and '1800' in e['sample']), None)
        year_max_error = next((e for e in errors if e['column'] == 'year' and '2200' in e['sample']), None)
        percentage_min_error = next((e for e in errors if e['column'] == 'percentage' and '-10' in e['sample']), None)
        percentage_max_error = next((e for e in errors if e['column'] == 'percentage' and '120' in e['sample']), None)
        count_min_error = next((e for e in errors if e['column'] == 'count' and '-5' in e['sample']), None)
        
        self.assertIsNotNone(year_min_error)
        self.assertIsNotNone(year_max_error)
        self.assertIsNotNone(percentage_min_error)
        self.assertIsNotNone(percentage_max_error)
        self.assertIsNotNone(count_min_error)
        
    def test_validate_formats(self):
        """Test format validation with regex patterns"""
        df = pd.DataFrame({
            'academic_year': ['2021/22', '2022-23', '2023/2024'],  # One invalid format
            'institution_id': ['AB123', 'cd456', 'XY-789']  # Mixed case and invalid character
        })
        
        result_df, errors = self.validator.validate_formats(df)
        
        # Check that format violations were detected
        academic_year_error = next((e for e in errors if e['column'] == 'academic_year'), None)
        institution_id_error = next((e for e in errors if e['column'] == 'institution_id'), None)
        
        self.assertIsNotNone(academic_year_error)
        self.assertIsNotNone(institution_id_error)
        self.assertIn('2022-23', academic_year_error['sample'])
        self.assertTrue(any(id in institution_id_error['sample'] for id in ['cd456', 'XY-789']))
        
    def test_check_relationships(self):
        """Test relationship validation between fields"""
        # Test academic year and year consistency
        df = pd.DataFrame({
            'year': [2021, 2022, 2023],
            'academic_year': ['2021/22', '2020/21', '2023/24'],  # Second row is inconsistent
            'count': [100, 40, 60],
            'total': [200, 50, 100],
            'percentage': [50, 80, 50]  # First and third rows are correct, second row is not
        })
        
        errors = self.validator.check_relationships(df)
        
        # Should find at least 2 consistency errors
        self.assertGreaterEqual(len(errors), 2)
        
        # Check specific errors
        year_consistency_error = next((e for e in errors if e['name'] == 'year_academic_year_consistency'), None)
        count_consistency_error = next((e for e in errors if e['name'] == 'count_percentage_consistency'), None)
        
        self.assertIsNotNone(year_consistency_error)
        self.assertIsNotNone(count_consistency_error)
        
    def test_apply_rules(self):
        """Test the complete validation pipeline"""
        # Create a test DataFrame with multiple issues
        df = pd.DataFrame({
            'institution': ['University A', 'University B', None],
            'year': [2021, '2022', 2200],  # Mixed types and out of range
            'count': [100, -5, 'invalid'],  # Negative and invalid type
            'percentage': [50, 110, 80],  # Value over 100
            'metric_type': ['enrollment', 'satisfaction', 'graduation']
        })
        
        result_df, errors = self.validator.apply_rules(df, 'performance')
        
        # Check that all types of validation were applied
        error_types = set(error['rule'] for error in errors)
        
        self.assertIn('required_columns', error_types)  # Missing 'value' for performance
        self.assertIn('data_type', error_types)  # Type issues
        self.assertIn('range', error_types)  # Range issues
        
    def test_generate_validation_report(self):
        """Test report generation"""
        report = self.validator.generate_validation_report(self.test_data, 'enrollment')
        
        # Check report structure
        self.assertIn('timestamp', report)
        self.assertIn('validation_summary', report)
        self.assertIn('dataset_info', report)
        self.assertIn('issues_by_rule', report)
        self.assertIn('issues_by_column', report)
        self.assertIn('all_issues', report)
        
        # Check issue counts
        self.assertTrue(report['validation_summary']['total_issues'] > 0)
        
    def test_validate_and_clean_file(self):
        """Test file-based validation"""
        df, report = self.validator.validate_and_clean_file(self.test_file_path, 'enrollment')
        
        # Check that the file was loaded and validated
        self.assertEqual(len(df), 3)  # Should have 3 rows
        self.assertIn('file_info', report)
        self.assertEqual(report['file_info']['name'], 'test_enrollment.csv')
        
    def test_save_validation_report(self):
        """Test saving validation report to a file"""
        # Generate a report
        report = self.validator.generate_validation_report(self.test_data, 'enrollment')
        
        # Save it to a file
        report_path = Path(self.temp_dir) / 'test_report.json'
        self.validator.save_validation_report(report, report_path)
        
        # Check if the file exists and contains valid JSON
        self.assertTrue(report_path.exists())
        with open(report_path, 'r') as f:
            loaded_report = json.load(f)
            
        # Check that the loaded report matches the original
        self.assertEqual(loaded_report['validation_summary']['total_issues'], 
                         report['validation_summary']['total_issues'])

    def test_categorize_errors(self):
        """Test error categorization by severity and type"""
        errors = [
            {'rule': 'data_type', 'column': 'year', 'severity': 'warning'},
            {'rule': 'range', 'column': 'percentage', 'severity': 'error'},
            {'rule': 'format', 'column': 'academic_year', 'severity': 'warning'},
            {'rule': 'consistency', 'columns': ['year', 'academic_year'], 'severity': 'error'}
        ]
        
        categorized = self.validator.categorize_errors(errors)
        
        # Check categorization by severity
        self.assertEqual(len(categorized['warning']), 2)
        self.assertEqual(len(categorized['error']), 2)
        
        # Check categorization by rule
        self.assertEqual(len(categorized['by_rule']['data_type']), 1)
        self.assertEqual(len(categorized['by_rule']['range']), 1)
        self.assertEqual(len(categorized['by_rule']['format']), 1)
        self.assertEqual(len(categorized['by_rule']['consistency']), 1)
        
        # Check categorization by column
        self.assertEqual(len(categorized['by_column']['year']), 1)
        self.assertEqual(len(categorized['by_column']['percentage']), 1)
        self.assertEqual(len(categorized['by_column']['academic_year']), 1)
        # consistency rule doesn't have a single column

if __name__ == '__main__':
    unittest.main() 