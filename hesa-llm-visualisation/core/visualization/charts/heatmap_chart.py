# import pandas as pd
# from typing import Dict, Any, Optional
# import logging
# import numpy as np

# logger = logging.getLogger(__name__)

# def create_heatmap_chart(data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """
#     Create a heatmap chart configuration for Chart.js
    
#     Args:
#         data: DataFrame containing the data to visualize
#         parameters: Optional parameters for chart customization
        
#     Returns:
#         Dictionary containing Chart.js configuration for a heatmap
#     """
#     try:
#         if data.empty:
#             return {
#                 'datasets': [],
#                 'labels': []
#             }
            
#         params = parameters or {}
        
#         # Determine columns for x, y, and value
#         x_column = params.get('x_column')
#         y_column = params.get('y_column')
#         value_column = params.get('value_column')
        
#         # If columns are not specified, try to infer them
#         all_columns = data.columns.tolist()
        
#         if not x_column and len(all_columns) > 0:
#             x_column = all_columns[0]
            
#         if not y_column and len(all_columns) > 1:
#             y_column = all_columns[1]
        
#         if not value_column:
#             numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
#             if len(numeric_columns) > 0:
#                 value_column = numeric_columns[0]
#             else:
#                 return {
#                     'error': 'No numeric column found for values',
#                     'datasets': [],
#                     'labels': []
#                 }
                
#         # Ensure all required columns exist
#         for col, name in [(x_column, 'x_column'), (y_column, 'y_column'), (value_column, 'value_column')]:
#             if col not in data.columns:
#                 return {
#                     'error': f'Column {col} specified as {name} not found in data',
#                     'datasets': [],
#                     'labels': []
#                 }
                
#         # Pivot the data to create a matrix suitable for a heatmap
#         try:
#             pivot_data = data.pivot(index=y_column, columns=x_column, values=value_column)
#             pivot_data = pivot_data.fillna(0)  # Replace NaN with 0 or another appropriate value
#         except Exception as e:
#             logger.error(f"Error pivoting data for heatmap: {str(e)}")
#             return {
#                 'error': f'Error creating pivot table: {str(e)}',
#                 'datasets': [],
#                 'labels': []
#             }
            
#         # Get x and y labels from the pivot table
#         x_labels = pivot_data.columns.tolist()
#         y_labels = pivot_data.index.tolist()
        
#         # Convert data to format required by Chart.js
#         datasets = []
#         for i, y_label in enumerate(y_labels):
#             row_data = []
#             for j, x_label in enumerate(x_labels):
#                 value = pivot_data.iloc[i, j]
#                 row_data.append({
#                     'x': j,
#                     'y': i,
#                     'v': float(value)
#                 })
            
#             datasets.append({
#                 'label': str(y_label),
#                 'data': row_data
#             })
        
#         # Determine color scale
#         min_value = float(pivot_data.min().min())
#         max_value = float(pivot_data.max().max())
        
#         # Prepare chart configuration
#         chart_config = {
#             'type': 'heatmap',
#             'data': {
#                 'labels': [str(label) for label in x_labels],
#                 'datasets': datasets
#             },
#             'options': {
#                 'responsive': True,
#                 'maintainAspectRatio': False,
#                 'scales': {
#                     'x': {
#                         'title': {
#                             'display': True,
#                             'text': params.get('x_label', x_column)
#                         }
#                     },
#                     'y': {
#                         'title': {
#                             'display': True,
#                             'text': params.get('y_label', y_column)
#                         }
#                     }
#                 },
#                 'plugins': {
#                     'legend': {
#                         'display': False
#                     },
#                     'title': {
#                         'display': True,
#                         'text': params.get('title', f'Heatmap of {value_column} by {x_column} and {y_column}')
#                     },
#                     'tooltip': {
#                         'callbacks': {
#                             'label': """function(context) {
#                                 return `${context.raw.v}`;
#                             }"""
#                         }
#                     },
#                     'colorscheme': {
#                         'scheme': params.get('colorScheme', 'interpolateInferno'),
#                         'min': min_value,
#                         'max': max_value
#                     }
#                 }
#             }
#         }
        
#         return chart_config
        
#     except Exception as e:
#         logger.error(f"Error creating heatmap chart: {str(e)}")
#         return {
#             'error': str(e),
#             'datasets': [],
#             'labels': []
#         } 