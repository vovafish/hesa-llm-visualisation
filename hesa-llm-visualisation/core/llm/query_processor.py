from typing import Dict, List, Optional, Any
import re
from datetime import datetime
import logging
from .gpt_j_handler import GPTJHandler
import pandas as pd

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        """Initialize query processor with GPT-J handler."""
        self.gpt_handler = GPTJHandler()
        self.valid_metrics = [
            'enrollment', 'satisfaction', 'funding',
            'research', 'staff', 'facilities'
        ]
        self.valid_comparisons = [
            'trend', 'comparison', 'ranking',
            'distribution', 'correlation'
        ]

    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """Extract parameters from natural language query."""
        try:
            # Get structured parameters from GPT-J
            params = self.gpt_handler.process_query(query)
            
            # Validate and clean parameters
            validated_params = self.validate_parameters(params)
            
            # Add visualization suggestions
            validated_params['visualization'] = self.suggest_visualization(validated_params)
            
            return validated_params
            
        except Exception as e:
            logger.error(f"Error extracting parameters: {str(e)}")
            raise

    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted parameters."""
        try:
            validated = {}
            
            # Validate metrics
            metrics = params.get('metrics', [])
            if isinstance(metrics, str):
                metrics = [metrics]
            validated['metrics'] = [
                m.lower() for m in metrics
                if m.lower() in self.valid_metrics
            ]
            
            # Validate time period
            time_period = params.get('time_period', {})
            if isinstance(time_period, dict):
                validated['time_period'] = {
                    'start': time_period.get('start'),
                    'end': time_period.get('end')
                }
            else:
                validated['time_period'] = {
                    'start': None,
                    'end': None
                }
            
            # Validate institutions
            institutions = params.get('institutions', [])
            if isinstance(institutions, str):
                institutions = [institutions]
            validated['institutions'] = institutions
            
            # Validate comparison type
            comparison = params.get('comparison_type', '').lower()
            validated['comparison_type'] = (
                comparison if comparison in self.valid_comparisons
                else 'comparison'
            )
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating parameters: {str(e)}")
            raise

    def suggest_visualization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest appropriate visualization based on parameters."""
        try:
            suggestion = {
                'type': 'bar',  # default
                'options': {}
            }
            
            comparison_type = params.get('comparison_type')
            metrics = params.get('metrics', [])
            time_period = params.get('time_period', {})
            institutions = params.get('institutions', [])
            
            # Trend analysis over time
            if comparison_type == 'trend' and time_period.get('start') and time_period.get('end'):
                suggestion['type'] = 'line'
                suggestion['options'] = {
                    'showLine': True,
                    'tension': 0.1
                }
            
            # Comparison between institutions
            elif comparison_type == 'comparison' and len(institutions) > 1:
                if len(metrics) > 1:
                    # For multiple metrics across institutions, use radar chart
                    suggestion['type'] = 'radar'
                    suggestion['options'] = {
                        'plugins': {
                            'title': {
                                'display': True,
                                'text': 'Multi-metric Comparison'
                            }
                        },
                        'scales': {
                            'r': {
                                'beginAtZero': True
                            }
                        }
                    }
                else:
                    suggestion['type'] = 'bar'
                    suggestion['options'] = {
                        'indexAxis': 'y' if len(institutions) > 5 else 'x'
                    }
            
            # Distribution analysis
            elif comparison_type == 'distribution':
                suggestion['type'] = 'box'
                suggestion['options'] = {
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Distribution Analysis'
                        }
                    }
                }
            
            # Correlation analysis
            elif comparison_type == 'correlation' and len(metrics) >= 2:
                suggestion['type'] = 'scatter'
                suggestion['options'] = {
                    'plugins': {
                        'tooltip': {
                            'callbacks': {
                                'label': "function(context) { return `(${context.raw.x}, ${context.raw.y})`; }"
                            }
                        }
                    }
                }
            
            # Ranking visualization
            elif comparison_type == 'ranking':
                # Use funnel chart for ranking with decreasing values
                suggestion['type'] = 'funnel'
                suggestion['options'] = {
                    'indexAxis': 'y',
                    'plugins': {
                        'legend': {
                            'display': False
                        }
                    }
                }
                
            # Heatmap for multi-dimensional data
            elif comparison_type == 'heatmap' or (len(metrics) >= 1 and len(institutions) > 5):
                suggestion['type'] = 'heatmap'
                suggestion['options'] = {
                    'plugins': {
                        'tooltip': {
                            'callbacks': {
                                'label': "function(context) { return context.raw.v; }"
                            }
                        }
                    },
                    'scales': {
                        'x': {
                            'title': {
                                'display': True
                            }
                        },
                        'y': {
                            'title': {
                                'display': True
                            }
                        }
                    }
                }
                
            # Multi-dimensional analysis with 3+ variables
            elif len(metrics) >= 3:
                suggestion['type'] = 'bubble'
                suggestion['options'] = {
                    'plugins': {
                        'tooltip': {
                            'callbacks': {
                                'label': "function(context) { return `(${context.raw.x}, ${context.raw.y}, ${context.raw.r})`; }"
                            }
                        }
                    }
                }
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error suggesting visualization: {str(e)}")
            raise

    def generate_pandas_query(self, params: Dict[str, Any]) -> str:
        """Generate Pandas query string from parameters."""
        try:
            query_parts = []
            
            # Filter by institutions
            if params.get('institutions'):
                institutions = [f"institution == '{inst}'" for inst in params['institutions']]
                query_parts.append(f"({' | '.join(institutions)})")
            
            # Filter by time period
            time_period = params.get('time_period', {})
            if time_period.get('start'):
                query_parts.append(f"year >= {time_period['start']}")
            if time_period.get('end'):
                query_parts.append(f"year <= {time_period['end']}")
            
            # Combine all conditions
            query = ' & '.join(query_parts) if query_parts else ''
            
            return query
            
        except Exception as e:
            logger.error(f"Error generating Pandas query: {str(e)}")
            raise

    def process_query(self, query: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Process query and return results with visualization suggestions."""
        try:
            # Extract parameters
            params = self.extract_parameters(query)
            
            # Generate Pandas query
            pandas_query = self.generate_pandas_query(params)
            
            # Filter data
            if pandas_query:
                filtered_data = data.query(pandas_query)
            else:
                filtered_data = data
            
            # Prepare results
            results = {
                'parameters': params,
                'data': filtered_data,
                'visualization': params['visualization'],
                'query_interpretation': {
                    'original_query': query,
                    'structured_params': params,
                    'pandas_query': pandas_query
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise 