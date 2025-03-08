import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def create_scatter_chart(data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a scatter chart configuration for Chart.js
    
    Args:
        data: DataFrame containing the data to visualize
        parameters: Optional parameters for chart customization
        
    Returns:
        Dictionary containing Chart.js configuration for a scatter chart
    """
    try:
        if data.empty:
            return {
                'datasets': [],
                'labels': []
            }
            
        params = parameters or {}
        
        # Determine x and y columns
        x_column = params.get('x_column')
        y_column = params.get('y_column')
        
        # If x and y columns are not specified, try to infer them
        if not x_column:
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_columns) > 0:
                x_column = numeric_columns[0]
            else:
                return {
                    'error': 'No numeric column found for x-axis',
                    'datasets': [],
                    'labels': []
                }
                
        if not y_column:
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_columns) > 1:
                y_column = numeric_columns[1]
            elif len(numeric_columns) == 1:
                y_column = numeric_columns[0]  # Use the same column if only one is available
            else:
                return {
                    'error': 'No numeric column found for y-axis',
                    'datasets': [],
                    'labels': []
                }
                
        # Get color column if specified
        color_column = params.get('color_column')
        
        # Prepare data for scatter chart
        datasets = []
        
        if color_column and color_column in data.columns:
            # Create separate datasets for each color group
            for group, group_data in data.groupby(color_column):
                points = []
                for _, row in group_data.iterrows():
                    if pd.notna(row[x_column]) and pd.notna(row[y_column]):
                        points.append({
                            'x': float(row[x_column]),
                            'y': float(row[y_column])
                        })
                
                datasets.append({
                    'label': str(group),
                    'data': points,
                    'backgroundColor': params.get('backgroundColor', 'rgba(54, 162, 235, 0.5)'),
                    'borderColor': params.get('borderColor', 'rgba(54, 162, 235, 1)'),
                    'pointRadius': params.get('pointRadius', 5),
                    'pointHoverRadius': params.get('pointHoverRadius', 8)
                })
        else:
            # Create a single dataset with all points
            points = []
            for _, row in data.iterrows():
                if pd.notna(row[x_column]) and pd.notna(row[y_column]):
                    points.append({
                        'x': float(row[x_column]),
                        'y': float(row[y_column])
                    })
            
            datasets.append({
                'label': params.get('label', 'Data Points'),
                'data': points,
                'backgroundColor': params.get('backgroundColor', 'rgba(54, 162, 235, 0.5)'),
                'borderColor': params.get('borderColor', 'rgba(54, 162, 235, 1)'),
                'pointRadius': params.get('pointRadius', 5),
                'pointHoverRadius': params.get('pointHoverRadius', 8)
            })
            
        # Prepare chart configuration
        chart_config = {
            'type': 'scatter',
            'data': {
                'datasets': datasets
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'legend': {
                        'position': 'top',
                    },
                    'title': {
                        'display': True,
                        'text': params.get('title', f'{y_column} vs {x_column}')
                    }
                },
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': params.get('x_label', x_column)
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': params.get('y_label', y_column)
                        }
                    }
                }
            }
        }
        
        return chart_config
        
    except Exception as e:
        logger.error(f"Error creating scatter chart: {str(e)}")
        return {
            'error': str(e),
            'datasets': [],
            'labels': []
        } 