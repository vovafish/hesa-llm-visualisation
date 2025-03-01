#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual Data Validator for HESA Data

This script provides a command-line interface for validating individual HESA data files.
It allows users to specify a file to validate and apply cleaning operations.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "logs" / "validation.log", mode='a')
    ]
)
logger = logging.getLogger("manual_validator")

# Create logs directory if it doesn't exist
(project_root / "logs").mkdir(parents=True, exist_ok=True)


class ManualValidator:
    """
    A class for manually validating and cleaning individual HESA data files.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the manual validator.

        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        
        # Initialize validators
        self.csv_validator = CSVValidator()
        self.schema_validator = SchemaValidator()
        self.data_cleaner = DataCleaner()
        self.validation_rules = ValidationRules()
        
        # Create necessary directories
        self.reports_dir = project_root / "data" / "reports"
        self.cleaned_dir = project_root / "data" / "cleaned_files"
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            logger.info("ManualValidator initialized")

    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a single file and generate a report.

        Args:
            file_path: Path to the file to validate

        Returns:
            Dictionary containing validation results
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return {"error": f"File does not exist: {file_path}"}
        
        file_name = file_path.name
        file_extension = file_path.suffix.lower()
        
        logger.info(f"Validating file: {file_name}")
        
        # Check file type
        if file_extension != '.csv':
            logger.error(f"Unsupported file type: {file_extension}. Only CSV files are supported.")
            return {"error": f"Unsupported file type: {file_extension}. Only CSV files are supported."}
        
        # Initialize validation report
        validation_report = {
            "file_name": file_name,
            "file_path": str(file_path),
            "validation_time": pd.Timestamp.now().isoformat(),
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
            return validation_report


def main():
    """
    Main function to run the manual validator from the command line.
    """
    parser = argparse.ArgumentParser(description="HESA Manual Data Validator")
    parser.add_argument("--file", type=str, required=True, help="Path to the file to validate")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")
    args = parser.parse_args()
    
    # Create validator
    validator = ManualValidator(verbose=args.verbose)
    
    # Validate file
    print(f"\nValidating file: {args.file}")
    print("=" * 60)
    
    validation_report = validator.validate_file(args.file)
    
    # Print summary
    print("\nValidation Summary:")
    print("-" * 60)
    print(f"Total issues found: {validation_report['summary']['total_issues']}")
    print(f"Critical issues: {validation_report['summary']['critical_issues']}")
    print(f"Warnings: {validation_report['summary']['warnings']}")
    print(f"Fixed issues: {validation_report['summary']['fixed_issues']}")
    
    # Print sample issues
    if validation_report["issues"]:
        print("\nSample Issues:")
        print("-" * 60)
        
        # Display up to 5 issues as example
        for i, issue in enumerate(validation_report["issues"][:5], 1):
            print(f"{i}. {issue['type']} ({issue['severity']})")
            print(f"   Details: {issue['details']}")
            print(f"   Fixable: {'Yes' if issue['fixable'] else 'No'}")
            print()
    
    # Print cleaned file path if available
    if "cleaned_file" in validation_report:
        print("\nCleaned file saved to:")
        print(validation_report["cleaned_file"])
    
    print("\nValidation report saved to:")
    print(f"{os.path.join(os.path.dirname(args.file), 'reports', f'validation_report_{os.path.basename(args.file).replace('.', '_')}.json')}")
    
    print("\nValidation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 