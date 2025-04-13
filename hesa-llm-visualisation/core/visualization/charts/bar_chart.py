# import pandas as pd
# from typing import Dict, Any
# from .line_chart import get_color

# def create_bar_chart(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Creates a bar chart configuration for Chart.js
    
#     Args:
#         data: DataFrame containing the data to visualize
#         parame
# ters: Dictionary containing chart parameters
        
#     Returns:
#         Dictionary containing Chart.js configuration
#     """
#     # Get the most recent academic year for the bar chart
#     latest_year = data['Academic Year'].max()
#     latest_data = data[data['Academic Year'] == latest_year]
    
#     return {
#         'type': 'bar',
#         'data': {
#             'labels': [level for level in latest_data['Level of study'] if level != 'Total'],
#             'datasets': [{
#                 'label': f'Student Enrollment ({latest_year})',
#                 'data': [
#                     row['Number'] 
#                     for _, row in latest_data.iterrows() 
#                     if row['Level of study'] != 'Total'
#                 ],
#                 'backgroundColor': [
#                     get_color(i) 
#                     for i in range(len(latest_data[latest_data['Level of study'] != 'Total']))
#                 ],
#                 'borderColor': [
#                     get_color(i) 
#                     for i in range(len(latest_data[latest_data['Level of study'] != 'Total']))
#                 ],
#                 'borderWidth': 1
#             }]
#         },
#         'options': {
#             'responsive': True,
#             'plugins': {
#                 'title': {
#                     'display': True,
#                     'text': f'Student Enrollment by Level of Study ({latest_year})'
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
#                         'text': 'Level of Study'
#                     }
#                 }
#             }
#         }
#     } 