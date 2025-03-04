import pandas as pd
import os
from pathlib import Path
from core.visualization.chart_generator import ChartGenerator
import json

def test_chart_generation():
    """Example of using the chart generation system"""
    try:
        # Get the project root directory
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Load data from the cleaned_files directory
        data_path = project_root / 'data' / 'cleaned_files' / 'chart-1_cleaned.csv'
        data = pd.read_csv(data_path, skiprows=11)
        
        # Initialize chart generator
        chart_gen = ChartGenerator()
        
        # Print available chart types
        print("Available chart types:", chart_gen.get_available_chart_types())
        
        # Generate example charts
        chart_types = ['line', 'bar', 'pie']
        examples_dir = project_root / 'data' / 'examples'
        examples_dir.mkdir(exist_ok=True)
        
        for chart_type in chart_types:
            print(f"\nGenerating {chart_type} chart configuration...")
            chart_config = chart_gen.create_chart(data, chart_type)
            
            # Save the configuration to examples directory
            output_file = examples_dir / f'example_{chart_type}_chart.json'
            with open(output_file, 'w') as f:
                json.dump(chart_config, f, indent=2)
            print(f"Saved {chart_type} chart configuration to {output_file}")
    except Exception as e:
        print(f"Error during chart generation: {str(e)}")
        raise

if __name__ == "__main__":
    test_chart_generation() 