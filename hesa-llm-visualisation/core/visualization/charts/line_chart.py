# import pandas as pd
# from typing import Dict, Any

# def create_line_chart(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Creates a line chart configuration for Chart.js
    
#     Args:
#         data: DataFrame containing the data to visualize
#         parameters: Dictionary containing chart parameters
        
#     Returns:
#         Dictionary containing Chart.js configuration
#     """
#     return {
#         'type': 'line',
#         'data': {
#             'labels': data['Academic Year'].tolist(),
#             'datasets': [
#                 {
#                     'label': level,
#                     'data': data[data['Level of study'] == level]['Number'].tolist(),
#                     'fill': False,
#                     'tension': 0.1,
#                     'borderColor': get_color(i),
#                     'backgroundColor': get_color(i)
#                 }
#                 for i, level in enumerate(data['Level of study'].unique())
#                 if level != 'Total'  # Exclude total to avoid confusion
#             ]
#         },
#         'options': {
#             'responsive': True,
#             'plugins': {
#                 'title': {
#                     'display': True,
#                     'text': 'Student Enrollment by Level of Study'
#                 },
#                 'legend': {
#                     'position': 'top',
#                 }
#             },
#             'scales': {
#                 'y': {
#                     'beginAtZero': True,
#                     'title': {
#                         'display': True,
#                         'text': 'Number of Students'
#                     }
#                 },
#                 'x': {
#                     'title': {
#                         'display': True,
#                         'text': 'Academic Year'
#                     }
#                 }
#             }
#         }
#     }

# def get_color(index: int) -> str:
#     """Returns a color from a predefined color palette"""
#     colors = [
#         '#4e79a7',  # Blue
#         '#f28e2c',  # Orange
#         '#e15759',  # Red
#         '#76b7b2',  # Cyan
#         '#59a14f',  # Green
#         '#edc949',  # Yellow
#         '#af7aa1',  # Purple
#         '#ff9da7',  # Pink
#         '#9c755f',  # Brown
#         '#bab0ab'   # Gray
#     ]
#     return colors[index % len(colors)] 