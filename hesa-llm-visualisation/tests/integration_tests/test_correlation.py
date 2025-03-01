#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-file Correlation Test Script

This script demonstrates how to use the DataCorrelator to analyze relationships
between different HESA datasets.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Warning: seaborn package not installed. Visualizations will be limited.")
    print("To install: pip install seaborn")
    SEABORN_AVAILABLE = False

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import the DataCorrelator
try:
    from core.data_processing.correlation_engine import DataCorrelator
except ModuleNotFoundError:
    # Try alternate import path
    try:
        from core.data_processing.advanced.correlation_engine import DataCorrelator
    except ModuleNotFoundError:
        print("Error: Could not import DataCorrelator.")
        print("Please make sure the correlation_engine.py file exists in either:")
        print("- core/data_processing/correlation_engine.py")
        print("- core/data_processing/advanced/correlation_engine.py")
        sys.exit(1)

# Custom JSON encoder to handle NumPy data types
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
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def main():
    """Main function demonstrating the cross-file correlation workflow"""
    print("=" * 60)
    print("HESA Cross-file Correlation Example")
    print("=" * 60)
    
    # Step 1: Initialize the correlator
    print("\nStep 1: Initializing DataCorrelator")
    correlator = DataCorrelator()
    
    # Step 2: Ensure we have some sample data to work with
    print("\nStep 2: Checking for sample data")
    raw_dir = project_root / "data" / "raw_files"
    
    # Check if we have the necessary sample data
    sample_files = list(raw_dir.glob("*.csv"))
    if len(sample_files) < 2:
        print(f"Not enough sample files found in {raw_dir}.")
        print("Running the validation script first to generate sample data...")
        
        # You might want to run the data generator here
        try:
            # Set PYTHONPATH to include the project root
            os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
            
            # Import the dataset generator
            from core.data_processing.test_utils.dataset_generator import generate_all_datasets
            
            # Generate datasets
            print("Generating sample datasets...")
            generate_all_datasets(raw_dir, rows_per_dataset=100, error_rate=0.05, all_years=True)
            
            # Check again for sample files
            sample_files = list(raw_dir.glob("*.csv"))
        except Exception as e:
            print(f"Error generating sample data: {str(e)}")
            return
    
    # List available files
    print(f"\nFound {len(sample_files)} sample files:")
    for i, file in enumerate(sample_files, 1):
        print(f"  {i}. {file.name}")
    
    # Step 3: Select files to correlate
    print("\nStep 3: Selecting files to correlate")
    
    # For this example, let's use enrollment and performance data if available
    enrollment_files = [f for f in sample_files if "enrollment" in f.name.lower()]
    performance_files = [f for f in sample_files if "performance" in f.name.lower()]
    demographic_files = [f for f in sample_files if "demographic" in f.name.lower()]
    
    files_to_correlate = []
    
    # Add one enrollment file if available
    if enrollment_files:
        files_to_correlate.append(str(enrollment_files[0]))
        print(f"Selected enrollment file: {enrollment_files[0].name}")
    
    # Add one performance file if available
    if performance_files:
        files_to_correlate.append(str(performance_files[0]))
        print(f"Selected performance file: {performance_files[0].name}")
    
    # Add one demographic file if available
    if demographic_files:
        files_to_correlate.append(str(demographic_files[0]))
        print(f"Selected demographic file: {demographic_files[0].name}")
    
    # If we don't have specific files, just use the first two available
    if len(files_to_correlate) < 2 and len(sample_files) >= 2:
        files_to_correlate = [str(f) for f in sample_files[:2]]
        print(f"Selected generic files: {', '.join(Path(f).name for f in files_to_correlate)}")
    
    # Check if we have enough files
    if len(files_to_correlate) < 2:
        print("Error: Need at least 2 files to demonstrate correlation")
        return
    
    # Step 4: Correlate the files
    print(f"\nStep 4: Correlating {len(files_to_correlate)} files")
    try:
        merged_data = correlator.correlate_files(files_to_correlate)
        print(f"Successfully merged data. Shape: {merged_data.shape}")
        
        # Display the columns in the merged dataset
        print("\nColumns in merged dataset:")
        for i, col in enumerate(merged_data.columns, 1):
            print(f"  {i}. {col}")
        
        # Show a sample of the merged data
        print("\nSample of merged data:")
        print(merged_data.head(3))
    except Exception as e:
        print(f"Error correlating files: {str(e)}")
        return
    
    # Step 5: Calculate correlations
    print("\nStep 5: Calculating correlations")
    try:
        # Filter to numeric columns only for correlation
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            print("Error: Not enough numeric columns for correlation analysis")
            return
            
        print(f"Found {len(numeric_cols)} numeric columns for correlation analysis")
        numeric_data = merged_data[numeric_cols]
        
        # Calculate correlations using different methods
        correlations = correlator.calculate_correlations(numeric_data)
        
        # Get Pearson correlation
        pearson_corr = correlations.get('pearson')
        if pearson_corr is not None:
            print("\nPearson correlation matrix (sample):")
            print(pearson_corr.iloc[:5, :5])  # Show just the first 5x5 section
        
        # Get Spearman correlation
        spearman_corr = correlations.get('spearman')
        if spearman_corr is not None:
            print("\nSpearman correlation matrix (sample):")
            print(spearman_corr.iloc[:5, :5])  # Show just the first 5x5 section
    except Exception as e:
        print(f"Error calculating correlations: {str(e)}")
        return
    
    # Step 6: Find strongest correlations
    print("\nStep 6: Finding strongest correlations")
    try:
        # Find pairs with correlation > 0.5
        strong_correlations = correlator.identify_strongest_correlations(
            pearson_corr, threshold=0.5
        )
        
        print(f"Found {len(strong_correlations)} strong correlations (abs > 0.5):")
        for col1, col2, corr_value in strong_correlations[:10]:  # Show top 10
            print(f"  • {col1} and {col2}: {corr_value:.3f}")
    except Exception as e:
        print(f"Error finding strongest correlations: {str(e)}")
        return
    
    # Step 7: Generate a correlation report
    print("\nStep 7: Generating correlation report")
    try:
        report = correlator.generate_correlation_report(merged_data)
        
        print("Correlation report generated successfully")
        print(f"Report contains {len(report)} sections:")
        for section in report:
            print(f"  • {section}")
        
        # Save the report
        reports_dir = project_root / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / "correlation_report.json"
        try:
            # Use the custom JSON encoder to handle NumPy data types
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            
            print(f"\nSaved correlation report to {report_path}")
        except Exception as e:
            print(f"Error saving report to JSON: {str(e)}")
            # As a fallback, try to save to a simpler format
            simple_report_path = reports_dir / "correlation_report_simple.txt"
            with open(simple_report_path, 'w') as f:
                f.write(str(report))
            print(f"Saved simplified report to {simple_report_path}")
    except Exception as e:
        print(f"Error generating correlation report: {str(e)}")
        return
    
    # Step 8: Visualize a correlation matrix
    print("\nStep 8: Visualizing correlation matrix")
    if not SEABORN_AVAILABLE:
        print("Skipping visualization - seaborn package not installed.")
        print("To enable visualizations, install seaborn: pip install seaborn")
    else:
        try:
            # Create a correlation heatmap
            plt.figure(figsize=(12, 10))
            
            # If we have many columns, just take the first 15
            if pearson_corr.shape[0] > 15:
                corr_subset = pearson_corr.iloc[:15, :15]
            else:
                corr_subset = pearson_corr
            
            # Create heatmap
            sns.heatmap(corr_subset, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
            plt.title('Correlation Matrix Heatmap', fontsize=16)
            plt.tight_layout()
            
            # Save the visualization
            figures_dir = project_root / "data" / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            fig_path = figures_dir / "correlation_heatmap.png"
            plt.savefig(fig_path)
            
            print(f"Saved correlation heatmap to {fig_path}")
            print("Note: To view the heatmap, open the saved PNG file.")
        except Exception as e:
            print(f"Error visualizing correlation matrix: {str(e)}")
            print("Note: This step requires matplotlib and seaborn packages.")
    
    print("\nCross-file correlation demonstration completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main() 