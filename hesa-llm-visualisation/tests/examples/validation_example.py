#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script demonstrating the use of ValidationRules for HESA data validation.
This script shows how to:
1. Validate a sample HESA dataset against predefined rules
2. Generate and save a validation report
3. Fix common issues in the data
4. Clean and export validated data
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from core.data_processing.validators.validation_rules import ValidationRules

def create_sample_data():
    """Create a sample HESA dataset with some validation issues for demonstration"""
    
    data = {
        'institution': ['University A', 'University B', 'University C', None, 'University E'],
        'institution_id': ['UA123', 'UB456', 'not valid', 'UD789', 'UE101'],
        'year': [2021, 2022, '2023', 2020, -1],  # One string, one negative
        'academic_year': ['2021/22', '2022/23', '2023-24', '2020/21', '2019/20'],  # One invalid format
        'metric_type': ['enrollment', 'satisfaction', 'graduation', 'enrollment', 'enrollment'],
        'count': [1500, 2300, -50, 1800, 2100],  # One negative
        'percentage': [75.5, 88.2, 92.7, 68.4, 115.2],  # One over 100%
        'total': [2000, 2500, 1000, 2500, 2000],  # For checking consistency
        'subject': ['Computer Science', 'Engineering', 'Medicine', 'Arts', 'Business']
    }
    
    df = pd.DataFrame(data)
    
    # Save the sample data to CSV
    raw_files_dir = project_root / 'data' / 'raw_files'
    raw_files_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file_path = raw_files_dir / 'enrollment_sample.csv'
    df.to_csv(sample_file_path, index=False)
    
    logger.info(f"Sample data created and saved to {sample_file_path}")
    return sample_file_path

def main():
    """Main execution function"""
    # 1. Create sample data
    sample_file_path = create_sample_data()
    
    # 2. Initialize ValidationRules with default settings
    validator = ValidationRules()
    
    # 3. Validate and get a report for the sample data
    logger.info("Validating sample data file...")
    df, report = validator.validate_and_clean_file(sample_file_path, dataset_type='enrollment')
    
    # 4. Save the validation report
    reports_dir = project_root / 'data' / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = reports_dir / 'validation_report.json'
    validator.save_validation_report(report, report_path)
    
    # Print validation summary
    logger.info(f"Validation complete. Found {report['validation_summary']['total_issues']} issues:")
    logger.info(f" - Errors: {report['validation_summary']['errors']}")
    logger.info(f" - Warnings: {report['validation_summary']['warnings']}")
    
    # 5. Automatically fix common issues and revalidate
    logger.info("Fixing common issues...")
    # Fix missing values
    df['institution'].fillna('Unknown Institution', inplace=True)
    
    # Fix negative counts
    if 'count' in df.columns:
        df.loc[df['count'] < 0, 'count'] = 0
    
    # Fix percentages over 100
    if 'percentage' in df.columns:
        df.loc[df['percentage'] > 100, 'percentage'] = 100
    
    # Fix invalid academic_year format
    if 'academic_year' in df.columns:
        # Replace any non-standard format with a proper one based on year
        invalid_format = ~df['academic_year'].str.match(r'^\d{4}/\d{2,4}$')
        for idx in df[invalid_format].index:
            if pd.notna(df.loc[idx, 'year']):
                year = int(df.loc[idx, 'year'])
                df.loc[idx, 'academic_year'] = f"{year}/{str(year+1)[-2:]}"
    
    # 6. Revalidate after fixing
    logger.info("Revalidating after fixes...")
    _, revalidation_report = validator.apply_rules(df, 'enrollment')
    
    # 7. Save the cleaned data if validation passes
    if not any(error['severity'] == 'error' for error in revalidation_report):
        cleaned_files_dir = project_root / 'data' / 'cleaned_files'
        cleaned_files_dir.mkdir(parents=True, exist_ok=True)
        
        cleaned_file_path = cleaned_files_dir / 'enrollment_cleaned.csv'
        df.to_csv(cleaned_file_path, index=False)
        logger.info(f"Cleaned data saved to {cleaned_file_path}")
    else:
        logger.warning("Critical issues remain. Data was not saved.")
    
    # 8. Compare before and after validation
    logger.info("\nValidation Report Comparison:")
    logger.info(f"Before fixes: {report['validation_summary']['total_issues']} issues")
    revalidation_categorized = validator.categorize_errors(revalidation_report)
    logger.info(f"After fixes: {len(revalidation_report)} issues")
    
    logger.info("\nDone.")

if __name__ == "__main__":
    main() 