# import pandas as pd
# from typing import Dict, Any, Optional
# import logging
# import numpy as np
# import random

# logger = logging.getLogger(__name__)

# def create_radar_chart(data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """
#     Create a radar chart configuration for Chart.js
    
#     Args:
#         data: DataFrame containing the data to visualize
#         parameters: Optional parameters for chart customization
        
#     Returns:
#         Dictionary containing Chart.js configuration for a radar chart
#     """
#     try:
#         if data.empty:
#             return {
#                 'datasets': [],
#                 'labels': []
#             }
            
#         params = parameters or {}
        
#         # Determine category column and value columns
#         category_column = params.get('category_column')
#         value_columns = params.get('value_columns', [])
#         group_column = params.get('group_column')
        
#         # If category column is not specified, try to infer it
#         if not category_column:
#             non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
#             if len(non_numeric_cols) > 0:
#                 category_column = non_numeric_cols[0]
#             else:
#                 return {
#                     'error': 'No suitable category column found',
#                     'datasets': [],
#                     'labels': []
#                 }
        
#         # If value columns are not specified, try to infer them
#         if not value_columns:
#             numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
#             if len(numeric_cols) > 0:
#                 value_columns = numeric_cols[:5]  # Limit to first 5 to avoid overcrowding
#             else:
#                 return {
#                     'error': 'No numeric columns found for values',
#                     'datasets': [],
#                     'labels': []
#                 }
        
#         # Check if all required columns exist
#         for col in [category_column] + value_columns:
#             if col not in data.columns:
#                 return {
#                     'error': f'Column {col} not found in data',
#                     'datasets': [],
#                     'labels': []
#                 }
        
#         # Define a list of colors for datasets
#         colors = [
#             'rgba(255, 99, 132, 0.5)',    # Red
#             'rgba(54, 162, 235, 0.5)',    # Blue
#             'rgba(255, 206, 86, 0.5)',    # Yellow
#             'rgba(75, 192, 192, 0.5)',    # Green
#             'rgba(153, 102, 255, 0.5)',   # Purple
#             'rgba(255, 159, 64, 0.5)',    # Orange
#             'rgba(199, 199, 199, 0.5)',   # Gray
#             'rgba(83, 102, 255, 0.5)',    # Indigo
#             'rgba(255, 99, 255, 0.5)',    # Pink
#         ]
        
#         # Process the data
#         if group_column and group_column in data.columns:
#             # Radar chart comparing groups across categories
#             unique_groups = data[group_column].unique()
#             unique_categories = data[category_column].unique()
            
#             datasets = []
#             for i, group in enumerate(unique_groups):
#                 group_data = data[data[group_column] == group]
                
#                 # Calculate average values for each category and each value column
#                 group_values = {}
#                 for category in unique_categories:
#                     category_data = group_data[group_data[category_column] == category]
#                     if not category_data.empty:
#                         group_values[category] = {
#                             col: float(category_data[col].mean()) 
#                             for col in value_columns
#                         }
                
#                 # Create a dataset for each value column
#                 for j, value_col in enumerate(value_columns):
#                     color_idx = (i * len(value_columns) + j) % len(colors)
#                     dataset_data = [
#                         group_values.get(category, {}).get(value_col, 0)
#                         for category in unique_categories
#                     ]
                    
#                     datasets.append({
#                         'label': f'{group} - {value_col}',
#                         'data': dataset_data,
#                         'backgroundColor': colors[color_idx].replace('0.5', '0.2'),
#                         'borderColor': colors[color_idx].replace('0.5', '1'),
#                         'borderWidth': 1,
#                         'pointBackgroundColor': colors[color_idx].replace('0.5', '1'),
#                         'pointRadius': 3
#                     })
            
#             labels = [str(category) for category in unique_categories]
            
#         else:
#             # Radar chart comparing value columns across categories
#             unique_categories = data[category_column].unique()
            
#             # Calculate average values for each category and each value column
#             category_values = {}
#             for category in unique_categories:
#                 category_data = data[data[category_column] == category]
#                 if not category_data.empty:
#                     category_values[category] = {
#                         col: float(category_data[col].mean()) 
#                         for col in value_columns
#                     }
            
#             # Create a dataset for each value column
#             datasets = []
#             for i, value_col in enumerate(value_columns):
#                 color_idx = i % len(colors)
#                 dataset_data = [
#                     category_values.get(category, {}).get(value_col, 0)
#                     for category in unique_categories
#                 ]
                
#                 datasets.append({
#                     'label': value_col,
#                     'data': dataset_data,
#                     'backgroundColor': colors[color_idx].replace('0.5', '0.2'),
#                     'borderColor': colors[color_idx].replace('0.5', '1'),
#                     'borderWidth': 1,
#                     'pointBackgroundColor': colors[color_idx].replace('0.5', '1'),
#                     'pointRadius': 3
#                 })
            
#             labels = [str(category) for category in unique_categories]
        
#         # Generate the chart configuration
#         chart_config = {
#             'type': 'radar',
#             'data': {
#                 'labels': labels,
#                 'datasets': datasets
#             },
#             'options': {
#                 'responsive': True,
#                 'plugins': {
#                     'legend': {
#                         'position': 'top',
#                     },
#                     'title': {
#                         'display': True,
#                         'text': params.get('title', f'Radar Chart: {", ".join(value_columns)}')
#                     }
#                 },
#                 'scales': {
#                     'r': {
#                         'beginAtZero': params.get('begin_at_zero', True),
#                         'ticks': {
#                             'stepSize': params.get('step_size')
#                         }
#                     }
#                 }
#             }
#         }
        
#         return chart_config
        
#     except Exception as e:
#         logger.error(f"Error creating radar chart: {str(e)}")
#         return {
#             'error': str(e),
#             'datasets': [],
#             'labels': []
#         } 