from typing import Dict, Any
import pandas as pd
import logging
from ..llm.query_processor import QueryProcessor

# Configure logging
logger = logging.getLogger(__name__)

def parse_llm_response(response: str) -> Dict[str, Any]:
    """
    Parse the LLM response into actionable data operations
    
    Args:
        response: The response from the LLM system
        
    Returns:
        A dictionary of operations to apply to the dataset
    """
    try:
        # Parse the structured response from GPT-J
        if isinstance(response, dict):
            # If response is already a dictionary (from newer implementations)
            parsed = response
        else:
            # Handle string response from older implementations
            import json
            import re
            
            # Find JSON-like structure in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON structure found in response")
            
            json_str = json_match.group()
            parsed = json.loads(json_str)
        
        # Extract operations
        operations = {
            'metrics': parsed.get('metrics', []),
            'time_period': parsed.get('time_period', {}),
            'institutions': parsed.get('institutions', []),
            'comparison_type': parsed.get('comparison_type', 'comparison'),
            'chart_type': parsed.get('visualization', {}).get('type', 'bar'),
            'chart_options': parsed.get('visualization', {}).get('options', {})
        }
        
        logger.info(f"Parsed operations: {operations}")
        return operations
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return {'error': str(e)}

def apply_data_operations(df: pd.DataFrame, operations: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply parsed operations to the dataframe
    
    Args:
        df: The input DataFrame to process
        operations: Dictionary of operations from parse_llm_response
        
    Returns:
        Processed DataFrame
    """
    try:
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Filter by institutions if specified
        if operations.get('institutions'):
            institutions = operations['institutions']
            if isinstance(institutions, list) and len(institutions) > 0:
                institution_filter = result_df['institution'].isin(institutions)
                result_df = result_df[institution_filter]
        
        # Filter by time period if specified
        time_period = operations.get('time_period', {})
        if time_period.get('start'):
            start_year = int(time_period['start'])
            result_df = result_df[result_df['year'] >= start_year]
        
        if time_period.get('end'):
            end_year = int(time_period['end'])
            result_df = result_df[result_df['year'] <= end_year]
        
        # Filter by metrics if specified
        metrics = operations.get('metrics', [])
        if metrics:
            available_cols = set(result_df.columns)
            valid_metrics = [m for m in metrics if m in available_cols]
            
            if valid_metrics:
                # Include essential columns plus requested metrics
                essential_cols = ['institution', 'year', 'region', 'mission_group']
                cols_to_keep = list(set(essential_cols).intersection(available_cols))
                cols_to_keep.extend(valid_metrics)
                result_df = result_df[cols_to_keep]
        
        logger.info(f"Applied operations. Result shape: {result_df.shape}")
        return result_df
    except Exception as e:
        logger.error(f"Error applying operations: {str(e)}")
        raise Exception(f"Error applying operations: {str(e)}")

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
