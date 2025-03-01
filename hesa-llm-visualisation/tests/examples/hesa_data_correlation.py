#!/usr/bin/env python
"""
HESA Data Correlation Example

This script demonstrates correlating HESA data files from the project's
data structure:
- Reads from data/raw_files/ (original CSV files)
- Uses the data/cleaned_files/ for processed outputs
"""

import os
import sys
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.data_processing.advanced.correlation_engine import DataCorrelator

def main():
    """Run HESA data correlation example using project data structure."""
    print("HESA Data Correlation Example")
    print("=============================")
    
    # Use project's data directory structure
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_dir = project_root / "data" / "raw_files"
    cleaned_data_dir = project_root / "data" / "cleaned_files"
    
    # Ensure directories exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    cleaned_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find CSV files in raw_data directory
    csv_files = list(raw_data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {raw_data_dir}")
        print("Creating sample data for demonstration...")
        
        # Create sample data if no files exist
        sample_data_dir = raw_data_dir / "samples"
        sample_data_dir.mkdir(exist_ok=True)
        
        # Generate sample HESA-like datasets
        create_sample_hesa_data(sample_data_dir)
        
        # Update list of CSV files to include samples
        csv_files = list(sample_data_dir.glob("*.csv"))
    
    print(f"\nFound {len(csv_files)} CSV files for correlation:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Initialize correlator
    correlator = DataCorrelator()
    
    try:
        # Correlate files
        print("\nCorrelating HESA data files...")
        merged_df = correlator.correlate_files([str(f) for f in csv_files])
        print(f"Created merged dataset with shape: {merged_df.shape}")
        
        # Save merged data to cleaned_files directory
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        merged_path = cleaned_data_dir / f"merged_hesa_data_{timestamp}.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"Saved merged data to {merged_path}")
        
        # Calculate correlations
        print("\nCalculating correlations between metrics...")
        correlations = correlator.calculate_correlations(
            merged_df, methods=['pearson', 'spearman'])
        
        # Identify strong correlations
        print("\nIdentifying strongest correlations...")
        strongest = correlator.identify_strongest_correlations(
            correlations['pearson'], threshold=0.6)
        
        print("Top 5 strongest correlations between metrics:")
        for i, (col1, col2, val) in enumerate(strongest[:5]):
            print(f"  {i+1}. {col1} ↔ {col2}: {val:.4f}")
        
        # Generate comprehensive report
        print("\nGenerating correlation report...")
        report = correlator.generate_correlation_report(merged_df)
        
        # Save report to cleaned_files directory
        report_path = cleaned_data_dir / f"hesa_correlation_report_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved correlation report to {report_path}")
        
        # Visualize correlations
        print("\nCreating correlation heatmap...")
        plt.figure(figsize=(12, 10))
        
        # If too many columns, limit to most correlated ones
        if correlations['pearson'].shape[0] > 20:
            # Extract columns involved in strongest correlations
            top_cols = set()
            for col1, col2, _ in strongest[:15]:  # Get top 15 pairs
                top_cols.add(col1)
                top_cols.add(col2)
            
            # Create subset correlation matrix
            top_cols = list(top_cols)
            corr_subset = correlations['pearson'].loc[top_cols, top_cols]
            
            # Create heatmap of subset
            sns.heatmap(
                corr_subset,
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                fmt=".2f"
            )
            plt.title("Pearson Correlation Heatmap (Top Correlations)")
        else:
            # Create heatmap of all correlations
            sns.heatmap(
                correlations['pearson'],
                annot=True,
                cmap='coolwarm',
                vmin=-1,
                vmax=1,
                fmt=".2f"
            )
            plt.title("Pearson Correlation Heatmap (All Metrics)")
        
        plt.tight_layout()
        
        # Save visualization to cleaned_files directory
        viz_path = cleaned_data_dir / f"hesa_correlation_heatmap_{timestamp}.png"
        plt.savefig(viz_path)
        print(f"Saved correlation heatmap to {viz_path}")
        
    except Exception as e:
        print(f"Error during correlation: {str(e)}")
    
    print("\nExample completed!")

def create_sample_hesa_data(output_dir: Path):
    """
    Create sample HESA-like data files for demonstration purposes.
    
    Args:
        output_dir: Directory to save sample files
    """
    import numpy as np
    from datetime import datetime, timedelta
    
    # Sample institutions
    institutions = [
        "University of Leicester", 
        "University of Cambridge",
        "University of Oxford",
        "Imperial College London",
        "University College London",
        "University of Manchester",
        "University of Edinburgh",
        "King's College London",
        "London School of Economics"
    ]
    
    # Sample metrics
    metrics = {
        "undergraduate_enrollment": (5000, 1500),  # mean, std dev
        "postgraduate_enrollment": (2000, 800),
        "international_students": (1500, 700),
        "student_satisfaction": (85, 5),  # percentage
        "graduate_employment": (90, 4),   # percentage
        "research_income": (50, 30),      # millions
        "staff_count": (2000, 800),
        "student_staff_ratio": (15, 3)
    }
    
    # Generate enrollment data for multiple years
    years = list(range(2018, 2023))
    
    # Generate data for each year
    for year in years:
        data = []
        for institution in institutions:
            row = {"institution": institution, "year": year}
            
            # Add metrics with some year-to-year correlation
            base_factors = {inst: np.random.normal(1.0, 0.1) for inst in institutions}
            year_factor = 1.0 + (year - 2018) * 0.03  # 3% growth per year on average
            
            for metric, (mean, std) in metrics.items():
                # Base value for this institution
                inst_base = mean * base_factors[institution]
                
                # Add yearly trend and random variation
                value = inst_base * year_factor * np.random.normal(1.0, 0.05)
                
                # Ensure sensible values (non-negative, percentages <= 100)
                if "percentage" in metric or metric in ["student_satisfaction", "graduate_employment"]:
                    value = min(100, max(0, value))
                else:
                    value = max(0, value)
                
                row[metric] = round(value, 1)
            
            data.append(row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        filename = output_dir / f"hesa_data_{year}.csv"
        df.to_csv(filename, index=False)
        print(f"Created sample file: {filename}")
    
    # Create a separate subject performance file
    subjects = [
        "Computer Science", "Mathematics", "Physics", "Chemistry", 
        "Biology", "Engineering", "Business", "Law", "Medicine",
        "Humanities", "Arts", "Social Sciences"
    ]
    
    subject_data = []
    for institution in institutions:
        for subject in subjects:
            # Base factors for institution and subject
            inst_factor = base_factors[institution]
            subj_factor = np.random.normal(1.0, 0.15)
            
            row = {
                "institution": institution,
                "subject": subject,
                "year": 2022,
                "enrollment": round(200 * inst_factor * subj_factor),
                "satisfaction_score": min(100, max(60, round(85 + 10 * np.random.normal(0, 1)))),
                "graduate_salary": round(30000 + 10000 * np.random.normal(0, 1)),
                "female_percentage": min(100, max(0, round(50 + 15 * np.random.normal(0, 1)))),
                "international_percentage": min(100, max(0, round(30 + 15 * np.random.normal(0, 1))))
            }
            subject_data.append(row)
    
    # Create DataFrame and save to CSV
    subject_df = pd.DataFrame(subject_data)
    subject_filename = output_dir / "hesa_subject_performance_2022.csv"
    subject_df.to_csv(subject_filename, index=False)
    print(f"Created sample file: {subject_filename}")

if __name__ == "__main__":
    main() 