#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Validation and Processing Pipeline

This script provides utilities for validating, cleaning, and processing
HESA data files as part of a data pipeline. It works with the ValidationRules
class to ensure data integrity before further processing.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
from datetime import datetime
import argparse

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from core.data_processing.validators.validation_rules import ValidationRules

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(project_root / "logs" / "data_pipeline.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """
    A pipeline for validating and processing HESA data files.
    
    This class provides methods to:
    1. Validate raw data files against predefined rules
    2. Clean and fix common issues
    3. Process and transform data for analysis
    4. Export cleaned data for visualization
    """
    
    def __init__(self, raw_dir: Optional[Path] = None, cleaned_dir: Optional[Path] = None, 
                 reports_dir: Optional[Path] = None, validator: Optional[ValidationRules] = None):
        """
        Initialize the data pipeline.
        
        Args:
            raw_dir: Directory containing raw data files
            cleaned_dir: Directory to store cleaned data files
            reports_dir: Directory to store validation reports
            validator: Optional ValidationRules instance (creates a default one if None)
        """
        # Set default directories
        self.raw_dir = raw_dir or project_root / "data" / "raw_files"
        self.cleaned_dir = cleaned_dir or project_root / "data" / "cleaned_files"
        self.reports_dir = reports_dir or project_root / "data" / "reports"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a logs directory if it doesn't exist
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the validator
        self.validator = validator or ValidationRules()
        
        logger.info(f"Data pipeline initialized with directories:")
        logger.info(f"  Raw data: {self.raw_dir}")
        logger.info(f"  Cleaned data: {self.cleaned_dir}")
        logger.info(f"  Reports: {self.reports_dir}")
    
    def discover_raw_files(self, file_pattern: str = "*.csv") -> List[Path]:
        """
        Discover raw data files matching the specified pattern.
        
        Args:
            file_pattern: Glob pattern for files to discover
            
        Returns:
            List of file paths
        """
        files = list(self.raw_dir.glob(file_pattern))
        logger.info(f"Discovered {len(files)} files matching pattern '{file_pattern}'")
        return files
    
    def infer_dataset_type(self, file_path: Path) -> str:
        """
        Infer the dataset type from the file name or content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dataset type string ('enrollment', 'performance', 'demographic', or 'general')
        """
        file_name = file_path.name.lower()
        
        # Try to infer from file name
        if "enrol" in file_name or "student" in file_name or "count" in file_name:
            return "enrollment"
        elif "performance" in file_name or "satisfaction" in file_name or "outcome" in file_name:
            return "performance"
        elif "demographic" in file_name or "diversity" in file_name or "ethnicity" in file_name or "gender" in file_name:
            return "demographic"
        
        # If couldn't determine from name, try to look at content
        try:
            # Read just a few rows to check columns
            df_sample = pd.read_csv(file_path, nrows=5)
            columns = [col.lower() for col in df_sample.columns]
            
            if "count" in columns or "enrollment" in columns or "student" in columns:
                return "enrollment"
            elif "performance" in columns or "score" in columns or "metric" in columns or "satisfaction" in columns:
                return "performance"
            elif "demographic" in columns or "ethnicity" in columns or "gender" in columns or "category" in columns:
                return "demographic"
            
        except Exception as e:
            logger.warning(f"Could not read file {file_path} to infer type: {str(e)}")
        
        # Default to general if couldn't determine
        return "general"
    
    def process_file(self, file_path: Path, auto_fix: bool = True) -> Tuple[Path, Dict]:
        """
        Process a single data file through the pipeline.
        
        Args:
            file_path: Path to the raw data file
            auto_fix: Whether to automatically fix common issues
            
        Returns:
            Tuple containing:
                - Path to the cleaned data file
                - Validation report dictionary
        """
        logger.info(f"Processing file: {file_path}")
        
        # Infer dataset type
        dataset_type = self.infer_dataset_type(file_path)
        logger.info(f"Inferred dataset type: {dataset_type}")
        
        # Validate the file
        df, report = self.validator.validate_and_clean_file(file_path, dataset_type)
        
        # Save validation report
        report_path = self.reports_dir / f"{file_path.stem}_validation_report.json"
        self.validator.save_validation_report(report, report_path)
        
        # Check for critical issues
        if report['validation_summary']['has_critical_issues']:
            if not auto_fix:
                logger.warning(f"File {file_path} has critical issues. Skipping auto-fix.")
                return None, report
        
        # Apply auto-fixes if requested
        if auto_fix:
            logger.info(f"Applying automatic fixes to {file_path}")
            df = self.apply_auto_fixes(df, report, dataset_type)
            
            # Revalidate after fixes
            fixed_df, fixed_report = self.validator.apply_rules(df, dataset_type)
            
            # Update the report with post-fix validation
            report['post_fix_validation'] = {
                'total_issues': len(fixed_report),
                'errors': sum(1 for e in fixed_report if e.get('severity') == 'error'),
                'warnings': sum(1 for e in fixed_report if e.get('severity') == 'warning'),
                'fixed_issues': report['validation_summary']['total_issues'] - len(fixed_report)
            }
            
            df = fixed_df
        
        # Save cleaned data
        cleaned_file_path = self.cleaned_dir / f"{file_path.stem}_cleaned.csv"
        df.to_csv(cleaned_file_path, index=False)
        logger.info(f"Saved cleaned data to {cleaned_file_path}")
        
        return cleaned_file_path, report
    
    def apply_auto_fixes(self, df: pd.DataFrame, report: Dict, dataset_type: str) -> pd.DataFrame:
        """
        Apply automatic fixes to common data issues.
        
        Args:
            df: DataFrame to fix
            report: Validation report containing issues
            dataset_type: Type of dataset
            
        Returns:
            Fixed DataFrame
        """
        # Group issues by rule and column for easier processing
        issues_by_column = {}
        for issue in report.get('all_issues', []):
            column = issue.get('column')
            if column:
                if column not in issues_by_column:
                    issues_by_column[column] = []
                issues_by_column[column].append(issue)
        
        # Fix missing values based on dataset type
        for column in df.columns:
            if df[column].isna().any():
                if column == 'institution':
                    df[column].fillna('Unknown Institution', inplace=True)
                elif column == 'year' and 'academic_year' in df.columns:
                    # Try to infer year from academic_year
                    for idx in df[df[column].isna()].index:
                        if pd.notna(df.loc[idx, 'academic_year']):
                            year_match = re.search(r'(\d{4})', df.loc[idx, 'academic_year'])
                            if year_match:
                                df.loc[idx, column] = int(year_match.group(1))
                elif column in ['count', 'value', 'rank']:
                    # Fill missing numeric values with 0
                    df[column].fillna(0, inplace=True)
                elif column == 'percentage':
                    # Fill missing percentages with 0
                    df[column].fillna(0, inplace=True)
                elif column == 'metric_type' and dataset_type == 'performance':
                    # Fill missing metric type with default
                    df[column].fillna('general', inplace=True)
        
        # Fix data type issues
        for column, issues in issues_by_column.items():
            data_type_issues = [i for i in issues if i.get('rule') == 'data_type']
            if data_type_issues and column in df.columns:
                # Convert to proper types where possible
                if any(i.get('expected') == 'integer' for i in data_type_issues):
                    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
                elif any(i.get('expected') == 'float' for i in data_type_issues):
                    df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
        
        # Fix range issues
        for column, issues in issues_by_column.items():
            range_issues = [i for i in issues if i.get('rule') == 'range']
            if range_issues and column in df.columns:
                if column == 'year':
                    # Fix year range issues
                    min_val, max_val = self.validator.range_rules.get('year', (1900, 2100))
                    df.loc[df[column] < min_val, column] = min_val
                    df.loc[df[column] > max_val, column] = max_val
                elif column == 'percentage':
                    # Fix percentage range issues
                    df.loc[df[column] < 0, column] = 0
                    df.loc[df[column] > 100, column] = 100
                elif column in ['count', 'value', 'rank']:
                    # Fix negative values
                    df.loc[df[column] < 0, column] = 0
        
        # Fix format issues
        for column, issues in issues_by_column.items():
            format_issues = [i for i in issues if i.get('rule') == 'format']
            if format_issues and column in df.columns:
                if column == 'academic_year' and 'year' in df.columns:
                    # Fix academic_year format based on year
                    pattern = self.validator.format_rules.get('academic_year', r'^\d{4}/\d{2,4}$')
                    invalid_format = ~df[column].astype(str).str.match(pattern)
                    for idx in df[invalid_format].index:
                        if pd.notna(df.loc[idx, 'year']):
                            year = int(df.loc[idx, 'year'])
                            df.loc[idx, column] = f"{year}/{str(year+1)[-2:]}"
                elif column == 'institution_id':
                    # Standardize institution IDs to uppercase
                    df[column] = df[column].astype(str).str.upper()
                    # Remove any non-alphanumeric characters
                    df[column] = df[column].str.replace(r'[^A-Z0-9]', '', regex=True)
        
        return df
    
    def run_pipeline(self, file_pattern: str = "*.csv", auto_fix: bool = True) -> Dict:
        """
        Run the complete data pipeline on all matching files.
        
        Args:
            file_pattern: Glob pattern for files to process
            auto_fix: Whether to automatically fix common issues
            
        Returns:
            Summary report of the pipeline run
        """
        start_time = datetime.now()
        
        # Discover files
        files = self.discover_raw_files(file_pattern)
        
        # Process each file
        results = {
            'processed_files': 0,
            'cleaned_files': 0,
            'error_files': 0,
            'total_issues_found': 0,
            'total_issues_fixed': 0,
            'file_details': []
        }
        
        for file_path in files:
            try:
                cleaned_path, report = self.process_file(file_path, auto_fix)
                
                file_result = {
                    'raw_file': str(file_path),
                    'cleaned_file': str(cleaned_path) if cleaned_path else None,
                    'report_file': str(self.reports_dir / f"{file_path.stem}_validation_report.json"),
                    'dataset_type': self.infer_dataset_type(file_path),
                    'issues_found': report['validation_summary']['total_issues'],
                    'critical_issues': report['validation_summary']['errors'],
                    'warnings': report['validation_summary']['warnings']
                }
                
                if auto_fix and 'post_fix_validation' in report:
                    file_result['issues_fixed'] = report['post_fix_validation']['fixed_issues']
                    file_result['remaining_issues'] = report['post_fix_validation']['total_issues']
                
                results['processed_files'] += 1
                results['total_issues_found'] += report['validation_summary']['total_issues']
                
                if cleaned_path:
                    results['cleaned_files'] += 1
                    if auto_fix and 'post_fix_validation' in report:
                        results['total_issues_fixed'] += report['post_fix_validation']['fixed_issues']
                else:
                    results['error_files'] += 1
                
                results['file_details'].append(file_result)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                results['error_files'] += 1
                results['file_details'].append({
                    'raw_file': str(file_path),
                    'error': str(e)
                })
        
        # Calculate runtime
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()
        
        # Compile summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': runtime,
            'files_processed': results['processed_files'],
            'files_cleaned': results['cleaned_files'],
            'files_with_errors': results['error_files'],
            'total_issues_found': results['total_issues_found'],
            'total_issues_fixed': results['total_issues_fixed'],
            'auto_fix_enabled': auto_fix,
            'file_details': results['file_details']
        }
        
        # Save summary report
        summary_path = self.reports_dir / f"pipeline_summary_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Pipeline completed. Processed {results['processed_files']} files.")
        logger.info(f"Summary report saved to {summary_path}")
        
        return summary

def main():
    """Command line interface for the data pipeline"""
    parser = argparse.ArgumentParser(description='HESA Data Validation and Processing Pipeline')
    parser.add_argument('--raw-dir', type=str, help='Directory containing raw data files')
    parser.add_argument('--cleaned-dir', type=str, help='Directory to store cleaned data files')
    parser.add_argument('--reports-dir', type=str, help='Directory to store validation reports')
    parser.add_argument('--pattern', type=str, default='*.csv', help='File pattern to process')
    parser.add_argument('--no-auto-fix', action='store_true', help='Disable automatic fixing of issues')
    
    args = parser.parse_args()
    
    # Set up directories
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    cleaned_dir = Path(args.cleaned_dir) if args.cleaned_dir else None
    reports_dir = Path(args.reports_dir) if args.reports_dir else None
    
    # Create and run pipeline
    pipeline = DataPipeline(raw_dir, cleaned_dir, reports_dir)
    summary = pipeline.run_pipeline(args.pattern, not args.no_auto_fix)
    
    # Print summary
    print("\nPipeline Summary:")
    print(f"Files processed: {summary['files_processed']}")
    print(f"Files cleaned: {summary['files_cleaned']}")
    print(f"Files with errors: {summary['files_with_errors']}")
    print(f"Total issues found: {summary['total_issues_found']}")
    print(f"Issues fixed: {summary['total_issues_fixed']}")
    print(f"Runtime: {summary['runtime_seconds']:.2f} seconds")
    
    # Return exit code based on success
    return 0 if summary['files_with_errors'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 