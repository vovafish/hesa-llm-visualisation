import pandas as pd
from typing import Dict, Any, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

def create_bubble_chart(data: pd.DataFrame, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create a bubble chart configuration for Chart.js
    
    Args:
        data: DataFrame containing the data to visualize
        parameters: Optional parameters for chart customization
        
    Returns:
        Dictionary containing Chart.js configuration for a bubble chart
    """
    try:
        if data.empty:
            return {
                'datasets': [],
                'labels': []
            }
            
        params = parameters or {}
        
        # Determine columns for x, y, and radius values
        x_column = params.get('x_column')
        y_column = params.get('y_column')
        r_column = params.get('r_column')  # Column for bubble radius
        label_column = params.get('label_column')
        group_column = params.get('group_column')  # For coloring bubbles by group
        
        # If columns are not specified, try to infer them
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
        
        if not x_column and len(numeric_cols) > 0:
            x_column = numeric_cols[0]
            
        if not y_column and len(numeric_cols) > 1:
            y_column = numeric_cols[1]
            
        if not r_column and len(numeric_cols) > 2:
            r_column = numeric_cols[2]
        elif not r_column and len(numeric_cols) > 0:
            # Use the first numeric column if we don't have enough, but scale values
            r_column = numeric_cols[0]
            
        if not label_column and len(non_numeric_cols) > 0:
            label_column = non_numeric_cols[0]
            
        if not group_column and len(non_numeric_cols) > 1:
            group_column = non_numeric_cols[1]
        
        # Verify that required columns exist
        required_cols = [x_column, y_column, r_column]
        for col in required_cols:
            if col not in data.columns:
                return {
                    'error': f'Required column {col} not found in data',
                    'datasets': [],
                    'labels': []
                }
        
        # Prepare the dataset
        if group_column and group_column in data.columns:
            # Create multiple datasets, one per group
            groups = data[group_column].unique()
            
            # Define colors for different groups
            colors = [
                'rgba(255, 99, 132, 0.7)',   # Red
                'rgba(54, 162, 235, 0.7)',   # Blue
                'rgba(255, 206, 86, 0.7)',   # Yellow
                'rgba(75, 192, 192, 0.7)',   # Green
                'rgba(153, 102, 255, 0.7)',  # Purple
                'rgba(255, 159, 64, 0.7)',   # Orange
                'rgba(199, 199, 199, 0.7)',  # Gray
            ]
            
            datasets = []
            for i, group in enumerate(groups):
                group_data = data[data[group_column] == group]
                
                # Scale radius values for better visualization
                radius_values = group_data[r_column].values
                max_radius = np.max(radius_values) if len(radius_values) > 0 else 1
                scale_factor = params.get('scale_factor', 20)
                
                bubbles = []
                for idx, row in group_data.iterrows():
                    # Scale radius to be visually appropriate (max ~40px)
                    radius = (row[r_column] / max_radius) * scale_factor if max_radius > 0 else 5
                    
                    bubble = {
                        'x': float(row[x_column]),
                        'y': float(row[y_column]),
                        'r': float(radius)
                    }
                    
                    # Add label if available
                    if label_column and label_column in row:
                        bubble['label'] = str(row[label_column])
                        
                    bubbles.append(bubble)
                
                # Get color for this group
                color_idx = i % len(colors)
                color = colors[color_idx]
                border_color = color.replace('0.7', '1.0')
                
                datasets.append({
                    'label': str(group),
                    'data': bubbles,
                    'backgroundColor': color,
                    'borderColor': border_color,
                    'borderWidth': 1,
                    'hoverRadius': 4,
                    'hoverBorderWidth': 2
                })
        else:
            # Create a single dataset with all bubbles
            # Scale radius values for better visualization
            radius_values = data[r_column].values
            max_radius = np.max(radius_values) if len(radius_values) > 0 else 1
            scale_factor = params.get('scale_factor', 20)
            
            bubbles = []
            for idx, row in data.iterrows():
                # Scale radius to be visually appropriate (max ~40px)
                radius = (row[r_column] / max_radius) * scale_factor if max_radius > 0 else 5
                
                bubble = {
                    'x': float(row[x_column]),
                    'y': float(row[y_column]),
                    'r': float(radius)
                }
                
                # Add label if available
                if label_column and label_column in row:
                    bubble['label'] = str(row[label_column])
                    
                bubbles.append(bubble)
            
            datasets = [{
                'label': params.get('dataset_label', 'Data Points'),
                'data': bubbles,
                'backgroundColor': params.get('backgroundColor', 'rgba(75, 192, 192, 0.7)'),
                'borderColor': params.get('borderColor', 'rgba(75, 192, 192, 1.0)'),
                'borderWidth': 1,
                'hoverRadius': 4,
                'hoverBorderWidth': 2
            }]
        
        # Create chart configuration
        chart_config = {
            'type': 'bubble',
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
                        'text': params.get('title', f'Bubble Chart: {x_column} vs {y_column} (size: {r_column})')
                    },
                    'tooltip': {
                        'callbacks': {
                            'label': """function(context) {
                                const data = context.raw;
                                let label = data.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                return label + `(${data.x}, ${data.y}, ${data.r})`;
                            }"""
                        }
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
        logger.error(f"Error creating bubble chart: {str(e)}")
        return {
            'error': str(e),
            'datasets': [],
            'labels': []
        } 