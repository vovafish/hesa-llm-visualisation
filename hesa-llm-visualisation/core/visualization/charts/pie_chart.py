# import pandas as pd
# from typing import Dict, Any
# from .line_chart import get_color

# def create_pie_chart(data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Creates a pie chart configuration for Chart.js
    
#     Args:
#         data: DataFrame containing the data to visualize
#         parameters: Dictionary containing chart parameters
        
#     Returns:
#         Dictionary containing Chart.js configuration
#     """
#     # Get the most recent academic year for the pie chart
#     latest_year = data['Academic Year'].max()
#     latest_data = data[data['Academic Year'] == latest_year]
    
#     # Filter out the total
#     latest_data = latest_data[latest_data['Level of study'] != 'Total']
    
#     return {
#         'type': 'pie',
#         'data': {
#             'labels': latest_data['Level of study'].tolist(),
#             'datasets': [{
#                 'data': latest_data['Number'].tolist(),
#                 'backgroundColor': [
#                     get_color(i) for i in range(len(latest_data))
#                 ],
#                 'borderColor': '#ffffff',
#                 'borderWidth': 1
#             }]
#         },
#         'options': {
#             'responsive': True,
#             'plugins': {
#                 'title': {
#                     'display': True,
#                     'text': f'Distribution of Students by Level of Study ({latest_year})'
#                 },
#                 'legend': {
#                     'position': 'top',
#                 },
#                 'tooltip': {
#                     'callbacks': {
#                         'label': '''
#                         function(context) {
#                             const total = context.dataset.data.reduce((a, b) => a + b, 0);
#                             const value = context.raw;
#                             const percentage = ((value / total) * 100).toFixed(1);
#                             return `${context.label}: ${value.toLocaleString()} (${percentage}%)`;
#                         }
#                         '''
#                     }
#                 }
#             }
#         }
#     } 