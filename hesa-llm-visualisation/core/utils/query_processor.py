from typing import Dict, Any, List
import pandas as pd
import logging
from ..llm.query_processor import QueryProcessor
import json

# Configure logging
logger = logging.getLogger(__name__)

def parse_llm_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the LLM response into a structured format for data operations.
    
    Args:
        response: Dictionary containing the LLM's response
        
    Returns:
        Dictionary with parsed operations
    """
    try:
        # Log the raw response for debugging
        logger.info(f"Parsing LLM response: {response}")
        
        # Validate required fields
        required_fields = ['metrics', 'time_period', 'institutions']
        missing_fields = [field for field in required_fields if field not in response]
        
        if missing_fields:
            logger.error(f"Missing required fields in LLM response: {missing_fields}")
            return {'error': f"Missing required fields: {', '.join(missing_fields)}"}
            
        # Extract and validate time period
        time_period = response.get('time_period', {})
        if not isinstance(time_period, dict) or 'start' not in time_period or 'end' not in time_period:
            logger.error("Invalid time_period format in LLM response")
            return {'error': "Invalid time period format"}
            
        # Determine data type based on metrics
        data_type = 'finance' if any('finance' in m.lower() for m in response['metrics']) else 'student'
        
        # Structure the operations
        operations = {
            'data_type': data_type,
            'metrics': response['metrics'],
            'time_period': time_period,
            'institutions': response['institutions'],
            'comparison_type': response.get('comparison_type', 'trend'),
            'visualization': response.get('visualization', {
                'type': 'bar',
                'options': {}
            })
        }
        
        logger.info(f"Parsed operations: {operations}")
        return operations
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return {'error': f"Failed to parse LLM response: {str(e)}"}

def apply_data_operations(df: pd.DataFrame, operations: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply the parsed operations to the dataset.
    
    Args:
        df: Input DataFrame
        operations: Dictionary of operations to apply
        
    Returns:
        Processed DataFrame
    """
    try:
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Filter by time period
        time_period = operations.get('time_period', {})
        if 'year' in result.columns:
            start_year = time_period.get('start')
            end_year = time_period.get('end')
            if start_year and end_year:
                result = result[
                    (result['year'] >= start_year) & 
                    (result['year'] <= end_year)
                ]
        
        # Filter by institutions
        institutions = operations.get('institutions', [])
        if institutions and 'institution' in result.columns:
            result = result[result['institution'].isin(institutions)]
        
        # Select relevant metrics
        metrics = operations.get('metrics', [])
        if metrics:
            # Keep only relevant columns plus any grouping columns
            columns_to_keep = ['year', 'institution'] + metrics
            existing_columns = [col for col in columns_to_keep if col in result.columns]
            result = result[existing_columns]
        
        # Apply comparison type operations
        comparison_type = operations.get('comparison_type', 'trend')
        if comparison_type == 'trend':
            # Group by year if trending over time
            if 'year' in result.columns:
                result = result.groupby('year').mean().reset_index()
        elif comparison_type == 'comparison':
            # Group by institution for comparisons
            if 'institution' in result.columns:
                result = result.groupby('institution').mean().reset_index()
        elif comparison_type == 'distribution':
            # Keep raw data for distribution analysis
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"Error applying data operations: {str(e)}")
        raise

def generate_pandas_query(operations: Dict[str, Any]) -> str:
    """
    Generate a pandas query string from operations
    
    Args:
        operations: Dictionary of operations
        
    Returns:
        Pandas query string
    """
    query_processor = QueryProcessor()
    return query_processor.generate_pandas_query(operations)
