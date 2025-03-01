#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HESA Dataset Generator

This script generates realistic HESA-like datasets for testing and demonstration
purposes. It creates various types of datasets that mimic real HESA data structures
and introduces controlled errors for testing validation rules.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import argparse

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define constants for data generation
UK_UNIVERSITIES = [
    ('University of Oxford', 'O10'),
    ('University of Cambridge', 'C05'),
    ('Imperial College London', 'I10'),
    ('University College London', 'U80'),
    ('University of Edinburgh', 'E56'),
    ('University of Manchester', 'M20'),
    ('King\'s College London', 'K60'),
    ('London School of Economics', 'L72'),
    ('University of Bristol', 'B78'),
    ('University of Warwick', 'W20'),
    ('University of Glasgow', 'G28'),
    ('Durham University', 'D86'),
    ('University of Sheffield', 'S18'),
    ('University of Southampton', 'S27'),
    ('University of Birmingham', 'B32'),
    ('University of Leeds', 'L23'),
    ('University of Nottingham', 'N84'),
    ('University of Exeter', 'E84'),
    ('Queen Mary University of London', 'Q50'),
    ('University of York', 'Y50'),
]

ACADEMIC_YEARS = ['2018/19', '2019/20', '2020/21', '2021/22', '2022/23']
YEARS = [2018, 2019, 2020, 2021, 2022]

SUBJECTS = [
    ('Computer Science', 'CS101'),
    ('Mathematics', 'MT102'),
    ('Physics', 'PH103'),
    ('Chemistry', 'CH104'),
    ('Biology', 'BI105'),
    ('Engineering', 'EN106'),
    ('Business Studies', 'BS107'),
    ('Economics', 'EC108'),
    ('Law', 'LW109'),
    ('Medicine', 'MD110'),
    ('Psychology', 'PS111'),
    ('Sociology', 'SO112'),
    ('History', 'HS113'),
    ('English Literature', 'EL114'),
    ('Modern Languages', 'ML115'),
]

METRIC_TYPES = [
    'student_satisfaction',
    'teaching_quality',
    'assessment_feedback',
    'academic_support',
    'learning_resources',
    'student_voice',
    'graduate_prospects',
    'research_quality',
]

DEMOGRAPHIC_TYPES = [
    'gender',
    'ethnicity',
    'disability',
    'age_group',
    'socioeconomic_group',
    'domicile',
]

DEMOGRAPHIC_CATEGORIES = {
    'gender': ['Female', 'Male', 'Other', 'Not specified'],
    'ethnicity': ['White', 'Asian', 'Black', 'Mixed', 'Other', 'Not specified'],
    'disability': ['No disability', 'Specific learning difficulty', 'Mental health condition', 
                  'Physical impairment', 'Other disability', 'Not specified'],
    'age_group': ['Under 21', '21-24', '25-29', '30-39', '40+'],
    'socioeconomic_group': ['Higher managerial', 'Lower managerial', 'Intermediate occupations', 
                           'Small employers', 'Lower supervisory', 'Semi-routine', 'Routine', 'Never worked'],
    'domicile': ['UK', 'EU', 'Non-EU International'],
}

def generate_enrollment_data(num_rows: int = 100, error_rate: float = 0.05) -> pd.DataFrame:
    """
    Generate enrollment data mimicking HESA student enrollment statistics.
    
    Args:
        num_rows: Number of rows to generate
        error_rate: Percentage of rows to contain errors
        
    Returns:
        DataFrame with enrollment data
    """
    data = []
    
    for _ in range(num_rows):
        # Select random university
        university, uni_id = random.choice(UK_UNIVERSITIES)
        
        # Select random year and academic year
        year_idx = random.randrange(len(YEARS))
        year = YEARS[year_idx]
        academic_year = ACADEMIC_YEARS[year_idx]
        
        # Select random subject
        subject, subject_code = random.choice(SUBJECTS)
        
        # Generate enrollment count (between 50 and
        count = int(np.random.normal(400, 150))
        count = max(50, count)  # Ensure at least 50 students
        
        # Introduce random errors based on error_rate
        if random.random() < error_rate:
            error_type = random.randint(1, 5)
            if error_type == 1:
                # Missing university name
                university = np.nan
            elif error_type == 2:
                # Non-numeric count
                count = f"about {count}" if count % 2 == 0 else -count
            elif error_type == 3:
                # Invalid year
                year = random.choice([1800, 2200, "not a year"])
            elif error_type == 4:
                # Invalid academic year format
                academic_year = academic_year.replace('/', '-')
            elif error_type == 5:
                # Invalid subject code
                subject_code = subject_code.lower() if subject_code[0].isupper() else subject_code + "-"
        
        # Create row
        row = {
            'institution': university,
            'institution_id': uni_id,
            'year': year,
            'academic_year': academic_year,
            'subject': subject,
            'subject_code': subject_code,
            'count': count,
            'level': random.choice(['Undergraduate', 'Postgraduate']),
            'mode': random.choice(['Full-time', 'Part-time']),
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def generate_performance_data(num_rows: int = 100, error_rate: float = 0.05) -> pd.DataFrame:
    """
    Generate performance data mimicking HESA institutional performance metrics.
    
    """
    data = []
    
    for _ in range(num_rows):
        # Select random university
        university, uni_id = random.choice(UK_UNIVERSITIES)
        
        # Select random year and academic year
        year_idx = random.randrange(len(YEARS))
        year = YEARS[year_idx]
        academic_year = ACADEMIC_YEARS[year_idx]
        
        # Select random metric type
        metric_type = random.choice(METRIC_TYPES)
        
        # Generate metric value (between 1 and 5 for satisfaction, 0-100 for percentages)
        if 'satisfaction' in metric_type or 'quality' in metric_type:
            value = round(random.uniform(2.5, 5.0), 1)
            max_value = 5.0
        else:
            value = round(random.uniform(50, 98.5), 1)
            max_value = 100.0
        
        # Calculate percentage of maximum
        percentage = (value / max_value) * 100
        
        # Generate rank (1 to number of universities)
        rank = random.randint(1, len(UK_UNIVERSITIES))
        
        # Introduce random errors based on error_rate
        if random.random() < error_rate:
            error_type = random.randint(1, 5)
            if error_type == 1:
                # Missing university name
                university = np.nan
            elif error_type == 2:
                # Invalid value
                value = f"{value} points" if value % 1 == 0 else value * 100
            elif error_type == 3:
                # Value out of range
                value = -value if random.random() < 0.5 else value * 2
            elif error_type == 4:
                # Percentage > 100
                percentage = 100 + random.uniform(0.1, 20.0)
            elif error_type == 5:
                # Invalid rank
                rank = 0 if random.random() < 0.5 else "first"
        
        # Create row
        row = {
            'institution': university,
            'institution_id': uni_id,
            'year': year,
            'academic_year': academic_year,
            'metric_type': metric_type,
            'value': value,
            'percentage': percentage,
            'rank': rank,
            'benchmark': round(random.uniform(value - 0.5, value + 0.5), 1)
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def generate_demographic_data(num_rows: int = 100, error_rate: float = 0.05) -> pd.DataFrame:
    """
    Generate demographic data mimicking HESA student demographic statistics.
    
    """
    data = []
    
    for _ in range(num_rows):
        # Select random university
        university, uni_id = random.choice(UK_UNIVERSITIES)
        
        # Select random year and academic year
        year_idx = random.randrange(len(YEARS))
        year = YEARS[year_idx]
        academic_year = ACADEMIC_YEARS[year_idx]
        
        # Select random demographic type and category
        demographic_type = random.choice(DEMOGRAPHIC_TYPES)
        category = random.choice(DEMOGRAPHIC_CATEGORIES[demographic_type])
        
        # Generate counts
        total_students = int(np.random.normal(2000, 500))
        total_students = max(500, total_students)
        
        # Calculate category count based on random percentage
        percentage = random.uniform(5, 45) if category != DEMOGRAPHIC_CATEGORIES[demographic_type][0] else random.uniform(40, 60)
        count = int((percentage / 100) * total_students)
        
        # Introduce random errors based on error_rate
        if random.random() < error_rate:
            error_type = random.randint(1, 5)
            if error_type == 1:
                # Missing university name
                university = np.nan
            elif error_type == 2:
                # Non-numeric count
                count = f"approx {count}" if count % 2 == 0 else -count
            elif error_type == 3:
                # Percentage > 100
                percentage = 100 + random.uniform(0.1, 20.0)
            elif error_type == 4:
                # Count > total
                count = total_students + random.randint(1, 100)
            elif error_type == 5:
                # Invalid demographic type
                demographic_type = demographic_type.upper()
        
        # Create row
        row = {
            'institution': university,
            'institution_id': uni_id,
            'year': year,
            'academic_year': academic_year,
            'demographic_type': demographic_type,
            'category': category,
            'count': count,
            'total': total_students,
            'percentage': round(percentage, 1)
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def generate_all_datasets(output_dir: Path, rows_per_dataset: int = 100, 
                          error_rate: float = 0.05, all_years: bool = False) -> Dict[str, Path]:
    """
    Generate all types of HESA datasets and save them to the specified directory.
    
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = {}
    
    # Generate enrollment data
    enrollment_df = generate_enrollment_data(rows_per_dataset, error_rate)
    if all_years:
        # Split by year into separate files
        for year in YEARS:
            year_df = enrollment_df[enrollment_df['year'] == year]
            if not year_df.empty:
                file_path = output_dir / f"enrollment_{year}.csv"
                year_df.to_csv(file_path, index=False)
                file_paths[f"enrollment_{year}"] = file_path
    else:
        # Save as a single file
        file_path = output_dir / "enrollment.csv"
        enrollment_df.to_csv(file_path, index=False)
        file_paths["enrollment"] = file_path
    
    # Generate performance data
    performance_df = generate_performance_data(rows_per_dataset, error_rate)
    if all_years:
        # Split by year into separate files
        for year in YEARS:
            year_df = performance_df[performance_df['year'] == year]
            if not year_df.empty:
                file_path = output_dir / f"performance_{year}.csv"
                year_df.to_csv(file_path, index=False)
                file_paths[f"performance_{year}"] = file_path
    else:
        # Save as a single file
        file_path = output_dir / "performance.csv"
        performance_df.to_csv(file_path, index=False)
        file_paths["performance"] = file_path
    
    # Generate demographic data
    demographic_df = generate_demographic_data(rows_per_dataset, error_rate)
    if all_years:
        # Split by year into separate files
        for year in YEARS:
            year_df = demographic_df[demographic_df['year'] == year]
            if not year_df.empty:
                file_path = output_dir / f"demographic_{year}.csv"
                year_df.to_csv(file_path, index=False)
                file_paths[f"demographic_{year}"] = file_path
    else:
        # Save as a single file
        file_path = output_dir / "demographic.csv"
        demographic_df.to_csv(file_path, index=False)
        file_paths["demographic"] = file_path
    
    # Generate combined dataset with all types
    combined_parts = []
    # Add some common columns
    for df, prefix in [(enrollment_df, 'enrollment'), 
                      (performance_df, 'performance'), 
                      (demographic_df, 'demographic')]:
        subset = df[['institution', 'institution_id', 'year', 'academic_year']].copy()
        # Add a type column to distinguish the source
        subset['data_type'] = prefix
        combined_parts.append(subset)
    
    combined_df = pd.concat(combined_parts, ignore_index=True)
    file_path = output_dir / "combined_metadata.csv"
    combined_df.to_csv(file_path, index=False)
    file_paths["combined_metadata"] = file_path
    
    return file_paths

def main():
    """Command line interface for generating HESA datasets"""
    parser = argparse.ArgumentParser(description='Generate HESA-like datasets for testing')
    parser.add_argument('--output-dir', type=str, default=None, 
                        help='Directory to save generated datasets (defaults to data/raw_files)')
    parser.add_argument('--rows', type=int, default=100, 
                        help='Number of rows per dataset (default: 100)')
    parser.add_argument('--error-rate', type=float, default=0.05, 
                        help='Percentage of rows with errors (default: 0.05)')
    parser.add_argument('--all-years', action='store_true', 
                        help='Create separate files for each year')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "raw_files"
    
    # Generate datasets
    print(f"Generating HESA datasets with {args.rows} rows per dataset and {args.error_rate:.1%} error rate...")
    file_paths = generate_all_datasets(output_dir, args.rows, args.error_rate, args.all_years)
    
    # Print summary
    print(f"\nGenerated datasets saved to {output_dir}:")
    for dataset_name, file_path in file_paths.items():
        print(f"  - {dataset_name}: {file_path.name}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 