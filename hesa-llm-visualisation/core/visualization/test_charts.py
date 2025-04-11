# import pandas as pd
# from chart_generator import ChartGenerator
# import json

# def test_chart_generation():
#     # 1. Load the data
#     data = pd.read_csv('../data/cleaned_files/chart-1_cleaned.csv', skiprows=11)
    
#     # 2. Initialize the chart generator
#     chart_gen = ChartGenerator()
    
#     # 3. Print available chart types
#     print("Available chart types:", chart_gen.get_available_chart_types())
    
#     # 4. Generate all types of charts
#     chart_types = ['line', 'bar', 'pie']
    
#     for chart_type in chart_types:
#         print(f"\nGenerating {chart_type} chart configuration...")
#         chart_config = chart_gen.create_chart(data, chart_type)
        
#         # Save the configuration to a JSON file for testing
#         output_file = f'test_output_{chart_type}_chart.json'
#         with open(output_file, 'w') as f:
#             json.dump(chart_config, f, indent=2)
#         print(f"Saved {chart_type} chart configuration to {output_file}")

# if __name__ == "__main__":
#     test_chart_generation() 