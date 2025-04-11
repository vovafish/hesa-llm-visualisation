# import pandas as pd
# from typing import Dict, Any, Optional
# import logging
# import numpy as np

# logger = logging.getLogger(__name__)

# def create_funnel_chart(data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#     """
#     Create a funnel chart configuration for Chart.js
    
#     Args:
#         data: DataFrame containing the data to visualize
#         parameters: Optional parameters for chart customization
        
#     Returns:
#         Dictionary containing Chart.js configuration for a funnel chart
#     """
#     try:
#         if data.empty:
#             return {
#                 'datasets': [],
#                 'labels': []
#             }
            
#         params = parameters or {}
        
#         # Determine category column and value column
#         category_column = params.get('category_column')
#         value_column = params.get('value_column')
        
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
        
#         # If value column is not specified, try to infer it
#         if not value_column:
#             numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
#             if len(numeric_cols) > 0:
#                 value_column = numeric_cols[0]
#             else:
#                 return {
#                     'error': 'No numeric column found for values',
#                     'datasets': [],
#                     'labels': []
#                 }
        
#         # Check if all required columns exist
#         for col in [category_column, value_column]:
#             if col not in data.columns:
#                 return {
#                     'error': f'Column {col} not found in data',
#                     'datasets': [],
#                     'labels': []
#                 }
        
#         # Aggregate data by category
#         funnel_data = data.groupby(category_column)[value_column].sum().reset_index()
        
#         # Sort by value in descending order
#         funnel_data = funnel_data.sort_values(by=value_column, ascending=False)
        
#         # Get categories and values
#         categories = funnel_data[category_column].tolist()
#         values = funnel_data[value_column].tolist()
        
#         # Calculate percentages for labels
#         total = sum(values)
#         percentages = [round(val / total * 100, 1) if total > 0 else 0 for val in values]
        
#         # Generate gradient colors
#         default_color = params.get('color', 'rgba(75, 192, 192, 0.8)')  # Default teal color
#         color_base = params.get('color_base', default_color.replace('0.8', ''))
        
#         # Generate a color scale from the base color
#         colors = []
#         opacity_step = 0.7 / len(categories) if len(categories) > 1 else 0
#         for i, category in enumerate(categories):
#             opacity = 0.9 - (i * opacity_step)
#             colors.append(f"{color_base}{opacity})")
        
#         # Create dataset
#         dataset = {
#             'data': values,
#             'backgroundColor': colors,
#             'borderWidth': 1,
#             'borderColor': 'rgba(255, 255, 255, 0.5)'  # Light border
#         }
        
#         # Create labels with percentages
#         display_labels = [f"{cat} ({pct}%)" for cat, pct in zip(categories, percentages)]
        
#         # Generate the chart configuration
#         chart_config = {
#             'type': 'bar',  # Using a horizontal bar chart with custom styling for funnel
#             'data': {
#                 'labels': display_labels,
#                 'datasets': [dataset]
#             },
#             'options': {
#                 'responsive': True,
#                 'indexAxis': 'y',  # Horizontal bar
#                 'plugins': {
#                     'legend': {
#                         'display': False,  # Hide legend for funnel chart
#                     },
#                     'title': {
#                         'display': True,
#                         'text': params.get('title', f'Funnel Chart: {value_column} by {category_column}')
#                     },
#                     'tooltip': {
#                         'callbacks': {
#                             'label': """function(context) {
#                                 return `${context.parsed.x} (${percentages[context.dataIndex]}%)`;
#                             }"""
#                         }
#                     }
#                 },
#                 'scales': {
#                     'x': {
#                         'title': {
#                             'display': True,
#                             'text': value_column
#                         }
#                     },
#                     'y': {
#                         'title': {
#                             'display': True,
#                             'text': category_column
#                         }
#                     }
#                 }
#             }
#         }
        
#         return chart_config
        
#     except Exception as e:
#         logger.error(f"Error creating funnel chart: {str(e)}")
#         return {
#             'error': str(e),
#             'datasets': [],
#             'labels': []
#         } 