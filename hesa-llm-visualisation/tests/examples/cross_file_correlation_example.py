#!/usr/bin/env python
"""
Example demonstrating the use of the DataCorrelator for cross-file correlation.

This script:
1. Generates sample data files
2. Correlates the files
3. Analyzes correlations
4. Produces a report
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.data_processing.advanced.correlation_engine import DataCorrelator
from core.data_processing.advanced.test_utils import generate_test_datasets, cleanup_test_files


def main():
    """Run the correlation example."""
    print("Cross-File Correlation Example")
    print("===============================")
    
    # Create output directory if it doesn't exist
    output_dir = Path("data/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate test data
    print("\nGenerating sample data files...")
    file_paths, dfs = generate_test_datasets(
        num_files=3,
        rows_per_file=100,
        common_columns=2,
        unique_columns=3,
        correlation_strength=0.8
    )
    
    print(f"Created {len(file_paths)} sample files:")
    for i, path in enumerate(file_paths):
        print(f"  - File {i+1}: {path} ({dfs[i].shape[0]} rows, {dfs[i].shape[1]} columns)")
    
    # Initialize the correlator
    correlator = DataCorrelator()
    
    # Correlate the files
    print("\nCorrelating files...")
    try:
        merged_df = correlator.correlate_files(file_paths)
        print(f"Created merged dataset with shape: {merged_df.shape}")
        
        # Save merged data
        merged_path = output_dir / "merged_data.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"Saved merged data to {merged_path}")
        
        # Calculate correlations
        print("\nCalculating correlations...")
        correlations = correlator.calculate_correlations(
            merged_df, methods=['pearson', 'spearman'])
        
        # Identify strong correlations
        print("\nIdentifying strongest correlations...")
        strongest = correlator.identify_strongest_correlations(
            correlations['pearson'], threshold=0.6)
        
        print("Top 5 strongest correlations:")
        for i, (col1, col2, val) in enumerate(strongest[:5]):
            print(f"  {i+1}. {col1} -- {col2}: {val:.4f}")
        
        # Generate full report
        print("\nGenerating correlation report...")
        report = correlator.generate_correlation_report(merged_df)
        
        # Save report
        report_path = output_dir / "correlation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Saved correlation report to {report_path}")
        
        # Visualize correlations
        print("\nCreating correlation heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            correlations['pearson'], 
            annot=True, 
            cmap='coolwarm', 
            vmin=-1, 
            vmax=1,
            fmt=".2f"
        )
        plt.title("Pearson Correlation Heatmap")
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_dir / "correlation_heatmap.png"
        plt.savefig(viz_path)
        print(f"Saved correlation heatmap to {viz_path}")
        
    except Exception as e:
        print(f"Error during correlation: {str(e)}")
    
    # Clean up temporary files
    print("\nCleaning up temporary files...")
    cleanup_test_files(file_paths)
    
    print("\nExample completed!")


if __name__ == "__main__":
    main() 