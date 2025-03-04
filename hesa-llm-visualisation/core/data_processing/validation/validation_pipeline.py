#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HESA Data Validation Pipeline

This script runs a complete validation pipeline on all data files in the raw_files directory.
It validates, cleans, and reports on data quality issues across multiple HESA datasets.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Import validators
try:
    from core.data_processing.validators.csv_validator import CSVValidator
    from core.data_processing.validators.schema_validator import SchemaValidator
    from core.data_processing.validators.data_cleaner import DataCleaner
    from core.data_processing.validators.validation_rules import ValidationRules
except ImportError as e:
    print(f"Error importing validators: {str(e)}")
    print("Please make sure all validator modules are available.")
    sys.exit(1)

# Try to import dataset generator for sample data generation
try:
    from core.data_processing.test_utils.dataset_generator import generate_all_datasets
except ImportError:
    print("Warning: Dataset generator not found. Sample data generation will not be available.")
    generate_all_datasets = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "logs" / "validation_pipeline.log", mode='a')
    ]
)
logger = logging.getLogger("validation_pipeline")

# Create logs directory if it doesn't exist
(project_root / "logs").mkdir(parents=True, exist_ok=True)


class ValidationPipeline:
    """
    A pipeline for validating and cleaning HESA data files.
    
    This class orchestrates the validation, cleaning, and reporting process
    for multiple HESA data files.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the validation pipeline.
        
        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        
        # Initialize validators
        self.csv_validator = CSVValidator()
        self.schema_validator = SchemaValidator()
        self.data_cleaner = DataCleaner()
        self.validation_rules = ValidationRules()
        
        # Set up directories
        self.raw_dir = project_root / "data" / "raw_files"
        self.reports_dir = project_root / "data" / "reports"
        self.cleaned_dir = project_root / "data" / "cleaned_files"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize validation summary
        self.validation_summary = {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "files_processed": 0,
            "files_with_issues": 0,
            "total_issues": 0,
            "total_critical_issues": 0,
            "total_warnings": 0,
            "total_fixed_issues": 0,
            "file_summaries": []
        }
        
        if self.verbose:
            logger.info("ValidationPipeline initialized")
    
    def generate_sample_data(self, rows_per_dataset: int = 100, error_rate: float = 0.05):
        """
        Generate sample HESA data files with controlled errors.
        
        Args:
            rows_per_dataset: Number of rows per file
            error_rate: Proportion of cells to contain errors
        """
        logger.info("Generating sample HESA data with controlled errors...")
        logger.info(f"Generating sample HESA data files with {rows_per_dataset} rows each")
        logger.info(f"Introducing errors at a rate of {error_rate * 100}%")
        
        try:
            if generate_all_datasets:
                generate_all_datasets(
                    output_dir=self.raw_dir,
                    rows_per_dataset=rows_per_dataset,
                    error_rate=error_rate
                )
            else:
                logger.warning("Sample data generation not available - dataset generator not found")
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
    
    def run_pipeline(self, generate_samples: bool = False):
        """
        Run the validation pipeline on all files in the raw_files directory.
        
        Args:
            generate_samples: Whether to generate sample data before validation
            
        Returns:
            Validation summary report
        """
        # Start timer
        start_time = datetime.now()
        
        # Generate sample data if requested
        if generate_samples:
            logger.info("Generating sample HESA data with controlled errors...")
            self.generate_sample_data()
        
        # Get list of CSV files
        csv_files = list(self.raw_dir.glob("*.csv"))
        
        if not csv_files:
            logger.warning("No CSV files found in the raw_files directory.")
            logger.info("Use the --generate-samples flag to create sample data.")
            return {
                "error": "No CSV files found in the raw_files directory."
            }
        
        logger.info(f"Found {len(csv_files)} CSV files to validate")
        
        # Process each file
        for i, file_path in enumerate(csv_files, 1):
            logger.info(f"Processing file {i}/{len(csv_files)}: {file_path.name}")
            
            # Validate file
            validation_report = self.validate_file(file_path)
            
            # Add to summary
            self.validation_summary["files_processed"] += 1
            
            if validation_report["summary"]["total_issues"] > 0:
                self.validation_summary["files_with_issues"] += 1
            
            self.validation_summary["total_issues"] += validation_report["summary"]["total_issues"]
            self.validation_summary["total_critical_issues"] += validation_report["summary"]["critical_issues"]
            self.validation_summary["total_warnings"] += validation_report["summary"]["warnings"]
            self.validation_summary["total_fixed_issues"] += validation_report["summary"]["fixed_issues"]
            
            # Add file summary to overall summary
            self.validation_summary["file_summaries"].append({
                "file_name": file_path.name,
                "total_issues": validation_report["summary"]["total_issues"],
                "critical_issues": validation_report["summary"]["critical_issues"],
                "warnings": validation_report["summary"]["warnings"],
                "fixed_issues": validation_report["summary"]["fixed_issues"],
                "has_cleaned_version": "cleaned_file" in validation_report
            })
        
        # End timer
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Update summary
        self.validation_summary["end_time"] = end_time.isoformat()
        self.validation_summary["duration_seconds"] = duration
        
        # Save summary report
        summary_path = self.reports_dir / "validation_summary.json"
        
        # Custom JSON encoder for NumPy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)
        
        with open(summary_path, 'w') as f:
            json.dump(self.validation_summary, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Validation pipeline completed in {duration:.2f} seconds")
        logger.info(f"Processed {self.validation_summary['files_processed']} files")
        logger.info(f"Found {self.validation_summary['total_issues']} issues")
        logger.info(f"Fixed {self.validation_summary['total_fixed_issues']} issues")
        
        return self.validation_summary
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a single file and generate a report.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary containing validation results
        """
        file_name = file_path.name
        
        logger.info(f"Validating file: {file_name}")
        
        # Initialize validation report
        validation_report = {
            "file_name": file_name,
            "file_path": str(file_path),
            "validation_time": datetime.now().isoformat(),
            "file_size_bytes": file_path.stat().st_size,
            "issues": [],
            "summary": {
                "total_issues": 0,
                "critical_issues": 0,
                "warnings": 0,
                "fixed_issues": 0,
            }
        }
        
        try:
            # Step 1: CSV validation
            logger.info("Step 1: Validating CSV structure")
            csv_issues = self.csv_validator.validate(str(file_path))
            
            for issue in csv_issues:
                validation_report["issues"].append(issue)
                if issue["severity"] == "critical":
                    validation_report["summary"]["critical_issues"] += 1
                else:
                    validation_report["summary"]["warnings"] += 1
            
            validation_report["summary"]["total_issues"] += len(csv_issues)
            
            if any(issue["severity"] == "critical" for issue in csv_issues):
                logger.error("Critical CSV structure issues found. Stopping validation.")
                
                # Save report for this file even though we're stopping
                report_file_name = f"validation_report_{file_name.replace('.', '_')}.json"
                report_path = self.reports_dir / report_file_name
                
                with open(report_path, 'w') as f:
                    json.dump(validation_report, f, indent=2)
                
                return validation_report
            
            # Step 2: Try to load the data
            logger.info("Step 2: Loading data")
            try:
                # Attempt to load with appropriate settings based on CSV validation
                df = pd.read_csv(
                    file_path,
                    encoding='utf-8',
                    low_memory=False,
                    dtype=str  # Load everything as strings initially
                )
                
                # Add basic data profile to the report
                validation_report["data_profile"] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "missing_values": int(df.isna().sum().sum()),
                }
                
            except Exception as e:
                logger.error(f"Error loading file: {str(e)}")
                validation_report["issues"].append({
                    "type": "data_loading_error",
                    "severity": "critical",
                    "details": str(e),
                    "fixable": False
                })
                validation_report["summary"]["critical_issues"] += 1
                validation_report["summary"]["total_issues"] += 1
                
                # Save report for this file even though we're stopping
                report_file_name = f"validation_report_{file_name.replace('.', '_')}.json"
                report_path = self.reports_dir / report_file_name
                
                with open(report_path, 'w') as f:
                    json.dump(validation_report, f, indent=2)
                
                return validation_report
            
            # Step 3: Schema validation
            logger.info("Step 3: Validating schema")
            schema_issues = self.schema_validator.validate_dataframe(df, file_name)
            
            for issue in schema_issues:
                validation_report["issues"].append(issue)
                if issue["severity"] == "critical":
                    validation_report["summary"]["critical_issues"] += 1
                else:
                    validation_report["summary"]["warnings"] += 1
            
            validation_report["summary"]["total_issues"] += len(schema_issues)
            
            # Step 4: Data validation rules
            logger.info("Step 4: Validating data against business rules")
            rule_issues = self.validation_rules.validate_dataframe(df, file_name)
            
            for issue in rule_issues:
                validation_report["issues"].append(issue)
                if issue["severity"] == "critical":
                    validation_report["summary"]["critical_issues"] += 1
                else:
                    validation_report["summary"]["warnings"] += 1
            
            validation_report["summary"]["total_issues"] += len(rule_issues)
            
            # Step 5: Data cleaning
            if any(issue["fixable"] for issue in validation_report["issues"]):
                logger.info("Step 5: Cleaning data")
                
                # Get all fixable issues
                fixable_issues = [issue for issue in validation_report["issues"] if issue["fixable"]]
                
                # Apply fixes
                cleaned_df, fix_report = self.data_cleaner.clean_dataframe(df, fixable_issues)
                
                # Update the validation report with fixes
                validation_report["fixes"] = fix_report
                validation_report["summary"]["fixed_issues"] = len(fix_report)
                
                # Save cleaned file
                cleaned_file_path = self.cleaned_dir / f"cleaned_{file_name}"
                cleaned_df.to_csv(cleaned_file_path, index=False)
                
                validation_report["cleaned_file"] = str(cleaned_file_path)
                logger.info(f"Cleaned file saved to: {cleaned_file_path}")
            else:
                logger.info("No fixable issues found.")
            
            # Save validation report
            report_file_name = f"validation_report_{file_name.replace('.', '_')}.json"
            report_path = self.reports_dir / report_file_name
            
            # Custom JSON encoder for NumPy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float64)):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2, cls=NumpyEncoder)
            
            logger.info(f"Validation report saved to: {report_path}")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Unexpected error during validation: {str(e)}")
            validation_report["issues"].append({
                "type": "validation_error",
                "severity": "critical",
                "details": str(e),
                "fixable": False
            })
            validation_report["summary"]["critical_issues"] += 1
            validation_report["summary"]["total_issues"] += 1
            
            # Save report for this file
            report_file_name = f"validation_report_{file_name.replace('.', '_')}.json"
            report_path = self.reports_dir / report_file_name
            
            with open(report_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
            
            return validation_report


def main():
    """
    Main function to run the validation pipeline from the command line.
    """
    parser = argparse.ArgumentParser(description="HESA Data Validation Pipeline")
    parser.add_argument("--generate-samples", action="store_true", help="Generate sample data before validation")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ValidationPipeline(verbose=args.verbose)
    
    # Run the pipeline
    print("\nRunning HESA Data Validation Pipeline")
    print("=" * 60)
    
    if args.generate_samples:
        print("\nGenerating sample HESA data with controlled errors...")
    
    print("\nValidating data files...")
    summary = pipeline.run_pipeline(generate_samples=args.generate_samples)
    
    # Print summary
    print("\nValidation Pipeline Summary:")
    print("-" * 60)
    print(f"Files processed: {summary['files_processed']}")
    print(f"Files with issues: {summary['files_with_issues']}")
    print(f"Total issues found: {summary['total_issues']}")
    print(f"Critical issues: {summary['total_critical_issues']}")
    print(f"Warnings: {summary['total_warnings']}")
    print(f"Fixed issues: {summary['total_fixed_issues']}")
    
    # Print file summaries
    if summary["file_summaries"]:
        print("\nFile Summaries:")
        print("-" * 60)
        
        for i, file_summary in enumerate(summary["file_summaries"], 1):
            print(f"{i}. {file_summary['file_name']}")
            print(f"   Issues: {file_summary['total_issues']} "
                  f"(Critical: {file_summary['critical_issues']}, "
                  f"Warnings: {file_summary['warnings']})")
            print(f"   Fixed: {file_summary['fixed_issues']}")
            print(f"   Cleaned version: {'Yes' if file_summary['has_cleaned_version'] else 'No'}")
            print()
    
    # Print locations
    print("\nReports and cleaned files:")
    print("-" * 60)
    print(f"Raw files: {os.path.abspath(pipeline.raw_dir)}")
    print(f"Validation reports: {os.path.abspath(pipeline.reports_dir)}")
    print(f"Cleaned files: {os.path.abspath(pipeline.cleaned_dir)}")
    
    print("\nValidation pipeline completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 