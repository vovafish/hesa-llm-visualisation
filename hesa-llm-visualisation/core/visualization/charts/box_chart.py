import pandas as pd
from typing import Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

def create_box_chart(data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a box plot chart configuration for Chart.js
    
    Args:
        data: DataFrame containing the data to visualize
        parameters: Optional parameters for chart customization
        
    Returns:
        Dictionary containing Chart.js configuration for a box plot
    """
    try:
        if data.empty:
            return {
                'datasets': [],
                'labels': []
            }
            
        params = parameters or {}
        
        # Determine value column
        value_column = params.get('value_column')
        
        # If value column is not specified, try to infer it
        if not value_column:
            numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_columns) > 0:
                value_column = numeric_columns[0]
            else:
                return {
                    'error': 'No numeric column found for values',
                    'datasets': [],
                    'labels': []
                }
                
        # Determine category column
        category_column = params.get('category_column')
        
        # Process the data for box plot
        if category_column and category_column in data.columns:
            # Group data by category
            categories = data[category_column].unique()
            box_data = []
            
            for category in categories:
                category_values = data[data[category_column] == category][value_column].dropna()
                
                if len(category_values) > 0:
                    # Calculate box plot statistics
                    q1 = float(np.percentile(category_values, 25))
                    median = float(np.percentile(category_values, 50))
                    q3 = float(np.percentile(category_values, 75))
                    min_val = float(category_values.min())
                    max_val = float(category_values.max())
                    
                    box_data.append({
                        'label': str(category),
                        'min': min_val,
                        'q1': q1,
                        'median': median,
                        'q3': q3,
                        'max': max_val
                    })
            
            # Prepare datasets for Chart.js
            labels = [item['label'] for item in box_data]
            
            datasets = [{
                'label': params.get('label', value_column),
                'data': [
                    {
                        'min': item['min'],
                        'q1': item['q1'],
                        'median': item['median'],
                        'q3': item['q3'],
                        'max': item['max'],
                        'outliers': []  # We're not identifying outliers in this simple implementation
                    }
                    for item in box_data
                ],
                'backgroundColor': params.get('backgroundColor', 'rgba(54, 162, 235, 0.5)'),
                'borderColor': params.get('borderColor', 'rgba(54, 162, 235, 1)'),
                'borderWidth': params.get('borderWidth', 1)
            }]
        else:
            # Single box plot for the entire dataset
            values = data[value_column].dropna()
            
            if len(values) > 0:
                # Calculate box plot statistics
                q1 = float(np.percentile(values, 25))
                median = float(np.percentile(values, 50))
                q3 = float(np.percentile(values, 75))
                min_val = float(values.min())
                max_val = float(values.max())
                
                labels = [params.get('label', value_column)]
                
                datasets = [{
                    'label': params.get('label', value_column),
                    'data': [{
                        'min': min_val,
                        'q1': q1,
                        'median': median,
                        'q3': q3,
                        'max': max_val,
                        'outliers': []  # We're not identifying outliers in this simple implementation
                    }],
                    'backgroundColor': params.get('backgroundColor', 'rgba(54, 162, 235, 0.5)'),
                    'borderColor': params.get('borderColor', 'rgba(54, 162, 235, 1)'),
                    'borderWidth': params.get('borderWidth', 1)
                }]
            else:
                return {
                    'error': f'No valid data in column {value_column}',
                    'datasets': [],
                    'labels': []
                }
        
        # Prepare chart configuration
        chart_config = {
            'type': 'boxplot',
            'data': {
                'labels': labels,
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
                        'text': params.get('title', f'Distribution of {value_column}')
                    }
                }
            }
        }
        
        return chart_config
        
    except Exception as e:
        logger.error(f"Error creating box chart: {str(e)}")
        return {
            'error': str(e),
            'datasets': [],
            'labels': []
        } 