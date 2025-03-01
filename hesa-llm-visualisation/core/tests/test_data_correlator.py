import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from pathlib import Path

# Ensure we can import from core
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.data_processing.advanced.correlation_engine import DataCorrelator
from core.data_processing.advanced.test_utils import generate_test_datasets, cleanup_test_files


class TestDataCorrelator(unittest.TestCase):
    """Test cases for the DataCorrelator class."""
    
    def setUp(self):
        """Set up test environment."""
        self.correlator = DataCorrelator()
        self.temp_files = []
        
    def tearDown(self):
        """Clean up test environment."""
        cleanup_test_files(self.temp_files)
    
    def test_load_file(self):
        """Test loading files of different formats."""
        # Create a simple DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        
        # Save in different formats
        temp_csv = tempfile.mktemp(suffix='.csv')
        temp_excel = tempfile.mktemp(suffix='.xlsx')
        
        df.to_csv(temp_csv, index=False)
        df.to_excel(temp_excel, index=False)
        
        self.temp_files.extend([temp_csv, temp_excel])
        
        # Test loading
        csv_df = self.correlator.load_file(temp_csv)
        excel_df = self.correlator.load_file(temp_excel)
        
        # Check data is loaded correctly
        pd.testing.assert_frame_equal(csv_df, df)
        pd.testing.assert_frame_equal(excel_df, df)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.correlator.load_file(tempfile.mktemp(suffix='.txt'))
        
    def test_identify_common_keys(self):
        """Test identification of common keys."""
        df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        df2 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'D': [7, 8]})
        df3 = pd.DataFrame({'A': [1, 2], 'E': [9, 10], 'F': [11, 12]})
        
        # Test with all dataframes
        common_keys = self.correlator.identify_common_keys([df1, df2, df3])
        self.assertEqual(common_keys, ['A'])
        
        # Test with subset of dataframes
        common_keys = self.correlator.identify_common_keys([df1, df2])
        self.assertEqual(set(common_keys), {'A', 'B'})
        
        # Test with empty list
        common_keys = self.correlator.identify_common_keys([])
        self.assertEqual(common_keys, [])
        
    def test_suggest_join_keys(self):
        """Test suggestion of join keys."""
        # DataFrame with obvious ID column
        df1 = pd.DataFrame({
            'student_id': range(100),
            'name': ['Student' + str(i) for i in range(100)],
            'score': np.random.normal(70, 10, 100)
        })
        
        # DataFrame with unique values but no clear ID
        df2 = pd.DataFrame({
            'unique_column': range(100),
            'name': ['Student' + str(i) for i in range(100)],
            'grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 100)
        })
        
        suggestions = self.correlator.suggest_join_keys([df1, df2])
        
        # Check that student_id was identified in first dataframe
        self.assertIn('student_id', suggestions['file_0'])
        
        # Check that unique_column was identified in second dataframe
        self.assertIn('unique_column', suggestions['file_1'])
        
    def test_merge_dataframes(self):
        """Test merging dataframes."""
        df1 = pd.DataFrame({
            'key': [1, 2, 3, 4],
            'value1': ['a', 'b', 'c', 'd']
        })
        
        df2 = pd.DataFrame({
            'key': [1, 2, 3, 5],  # Note: 5 is not in df1
            'value2': ['w', 'x', 'y', 'z']
        })
        
        df3 = pd.DataFrame({
            'key': [1, 2, 6, 7],  # Note: 6, 7 are not in df1 or df2
            'value3': [True, False, True, False]
        })
        
        # Test with all dataframes
        merged = self.correlator.merge_dataframes([df1, df2, df3], ['key'])
        
        # Check shape
        self.assertEqual(merged.shape[0], 7)  # 7 unique keys total
        self.assertEqual(merged.shape[1], 6)  # key + 3 values + 2 indicators
        
        # Check that all keys are present
        self.assertSetEqual(set(merged['key']), {1, 2, 3, 4, 5, 6, 7})
        
        # Test with single dataframe
        single = self.correlator.merge_dataframes([df1], ['key'])
        pd.testing.assert_frame_equal(single, df1)
        
        # Test with empty list
        with self.assertRaises(ValueError):
            self.correlator.merge_dataframes([], ['key'])
            
    def test_correlate_files(self):
        """Test correlating files end-to-end."""
        # Generate test datasets
        file_paths, _ = generate_test_datasets(
            num_files=3,
            rows_per_file=50,
            common_columns=2,
            unique_columns=2
        )
        self.temp_files.extend(file_paths)
        
        # Correlate files
        correlated_df = self.correlator.correlate_files(file_paths)
        
        # Check results
        self.assertIsInstance(correlated_df, pd.DataFrame)
        self.assertEqual(correlated_df.shape[0], 50)  # All rows preserved
        
        # Should have common columns + unique columns per file + indicators
        expected_cols = 2 + (2 * 3) + 3 + 2  # common + unique per file + category/binary/date + indicators
        self.assertEqual(correlated_df.shape[1], expected_cols)
        
    def test_calculate_correlations(self):
        """Test correlation calculation."""
        # Create dataframe with known correlations
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        y = x * 0.8 + np.random.normal(0, 0.2, 100)  # Strong correlation
        z = np.random.normal(0, 1, 100)  # No correlation
        
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})
        
        # Calculate correlations
        correlations = self.correlator.calculate_correlations(df, methods=['pearson', 'spearman'])
        
        # Check results
        self.assertIn('pearson', correlations)
        self.assertIn('spearman', correlations)
        
        # Check correlation values
        self.assertGreater(correlations['pearson'].loc['x', 'y'], 0.7)  # Strong positive correlation
        self.assertLess(abs(correlations['pearson'].loc['x', 'z']), 0.3)  # Weak/no correlation
        
    def test_identify_strongest_correlations(self):
        """Test identification of strongest correlations."""
        # Create correlation matrix
        corr_matrix = pd.DataFrame({
            'A': [1.0, 0.8, 0.3],
            'B': [0.8, 1.0, 0.1],
            'C': [0.3, 0.1, 1.0]
        }, index=['A', 'B', 'C'])
        
        # Get strongest correlations
        strong = self.correlator.identify_strongest_correlations(
            corr_matrix, threshold=0.5, exclude_self=True)
        
        # Check results
        self.assertEqual(len(strong), 1)  # Only A-B correlation exceeds threshold
        self.assertEqual(strong[0][0], 'A')
        self.assertEqual(strong[0][1], 'B')
        self.assertEqual(strong[0][2], 0.8)
        
    def test_generate_correlation_report(self):
        """Test generation of correlation report."""
        # Generate test data
        file_paths, dfs = generate_test_datasets(
            num_files=2,
            rows_per_file=50,
            common_columns=2,
            unique_columns=2,
            correlation_strength=0.8
        )
        self.temp_files.extend(file_paths)
        
        # Merge dataframes
        merged_df = self.correlator.merge_dataframes(dfs, ['common_col_0', 'common_col_1'])
        
        # Generate report
        report = self.correlator.generate_correlation_report(merged_df)
        
        # Check report structure
        self.assertIn('dataset_info', report)
        self.assertIn('correlations', report)
        self.assertIn('strongest_correlations', report)
        self.assertIn('summary', report)
        
        # Check dataset info
        self.assertEqual(report['dataset_info']['shape'], merged_df.shape)
        
        # Check correlations
        self.assertIn('pearson', report['correlations'])
        self.assertIn('spearman', report['correlations'])
        
        # Check strongest correlations
        self.assertIn('pearson', report['strongest_correlations'])
        self.assertTrue(len(report['strongest_correlations']['pearson']) > 0)


if __name__ == '__main__':
    unittest.main() 