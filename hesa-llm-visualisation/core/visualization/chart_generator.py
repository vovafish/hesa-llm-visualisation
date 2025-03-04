import pandas as pd
from typing import Dict, Any, Optional
from .charts import create_line_chart, create_bar_chart, create_pie_chart

class ChartGenerator:
    """Main class for generating Chart.js configurations for different chart types"""
    
    def __init__(self):
        self.chart_creators = {
            'line': create_line_chart,
            'bar': create_bar_chart,
            'pie': create_pie_chart
        }
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for visualization"""
        # Clean column names
        data.columns = data.columns.str.strip()
        
        # Ensure required columns exist
        required_columns = ['Academic Year', 'Level of study', 'Number']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert Number column to numeric
        data['Number'] = pd.to_numeric(data['Number'], errors='coerce')
        
        # Drop any rows with missing values
        data = data.dropna(subset=['Academic Year', 'Level of study', 'Number'])
        
        return data
    
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
        
        # Preprocess the data
        try:
            processed_data = self.preprocess_data(data)
        except Exception as e:
            raise ValueError(f"Error preprocessing data: {str(e)}")
            
        return self.chart_creators[chart_type](processed_data, parameters)
    
    def get_available_chart_types(self) -> list:
        """Returns a list of available chart types"""
        return list(self.chart_creators.keys()) 