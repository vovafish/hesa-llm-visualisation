import pandas as pd
from typing import Dict, Any, Optional
from .charts import (
    create_line_chart, 
    create_bar_chart, 
    create_pie_chart,
    create_scatter_chart,
    create_box_chart,
    create_heatmap_chart,
    create_radar_chart,
    create_funnel_chart,
    create_bubble_chart
)
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Main class for generating Chart.js configurations for different chart types"""
    
    def __init__(self):
        self.chart_creators = {
            'line': create_line_chart,
            'bar': create_bar_chart,
            'pie': create_pie_chart,
            'scatter': create_scatter_chart,
            'box': create_box_chart,
            'boxplot': create_box_chart,  # Alias for 'box'
            'heatmap': create_heatmap_chart,
            'radar': create_radar_chart,
            'funnel': create_funnel_chart,
            'bubble': create_bubble_chart
        }
    
    def preprocess_data(self, data: pd.DataFrame, options: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Preprocess the data for visualization
        
        Args:
            data: DataFrame containing the data to visualize
            options: Optional preprocessing options
            
        Returns:
            Processed DataFrame ready for visualization
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check if data is empty
        if df.empty:
            logger.warning("Empty DataFrame provided for visualization")
            return df
            
        # Log the columns for debugging
        logger.info(f"Columns in dataset: {list(df.columns)}")
        
        # Apply any additional preprocessing from options
        if options and 'preprocessing' in options:
            if options['preprocessing'].get('group_by'):
                group_cols = options['preprocessing']['group_by']
                if isinstance(group_cols, list) and all(col in df.columns for col in group_cols):
                    agg_cols = [col for col in df.columns if col not in group_cols]
                    if agg_cols:
                        df = df.groupby(group_cols)[agg_cols].sum().reset_index()
        
        # Ensure numeric columns are properly formatted
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass  # Keep as is if conversion fails
        
        return df
    
    def generate_chart_data(self, 
                           data: pd.DataFrame, 
                           chart_type: str = 'bar',
                           options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate chart data for the web UI based on processed data
        
        Args:
            data: DataFrame containing the data to visualize
            chart_type: Type of chart to create
            options: Optional chart options
            
        Returns:
            Dictionary containing chart data and configuration
        """
        try:
            # Preprocess the data
            processed_data = self.preprocess_data(data, options)
            
            if processed_data.empty:
                return {
                    'error': 'No data available for visualization',
                    'datasets': [],
                    'labels': []
                }
            
            # Determine chart parameters
            parameters = {}
            if options:
                parameters = options.copy()
            
            # Use the appropriate chart creator
            if chart_type in self.chart_creators:
                chart_config = self.chart_creators[chart_type](processed_data, parameters)
            else:
                logger.warning(f"Unsupported chart type: {chart_type}. Falling back to bar chart.")
                chart_config = self.chart_creators['bar'](processed_data, parameters)
            
            return chart_config
            
        except Exception as e:
            logger.error(f"Error generating chart data: {str(e)}")
            return {
                'error': str(e),
                'datasets': [],
                'labels': []
            }
    
    def create_chart(self, 
                    data: pd.DataFrame, 
                    chart_type: str, 
                    parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Creates a chart configuration based on the specified type
        
        Args:
            data: DataFrame containing the data to visualize
            chart_type: Type of chart to create ('line', 'bar', or 'pie')
            parameters: Optional parameters for chart customization
            
        Returns:
            Dictionary containing Chart.js configuration
            
        Raises:
            ValueError: If chart_type is not supported or data is invalid
        """
        if chart_type not in self.chart_creators:
            raise ValueError(f"Unsupported chart type: {chart_type}. "
                           f"Supported types are: {list(self.chart_creators.keys())}")
        
        if parameters is None:
            parameters = {}
        
        # Preprocess the data without automatic column checks
        try:
            processed_data = self.preprocess_data(data, parameters)
        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")
            
        return self.chart_creators[chart_type](processed_data, parameters)
    
    def get_available_chart_types(self) -> list:
        """Returns a list of available chart types"""
        return list(self.chart_creators.keys()) 