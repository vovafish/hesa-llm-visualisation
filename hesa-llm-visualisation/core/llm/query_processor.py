from typing import Dict, List, Optional
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        """Initialize the query processor with known metrics and institutions."""
        self.known_metrics = [
            'enrollment', 'postgraduate', 'undergraduate',
            'full_time', 'part_time', 'international',
            'domestic', 'research', 'teaching'
        ]
        
        self.known_institutions = [
            'University of Leicester',
            'Russell Group',
            'Million+',
            'University Alliance'
        ]

    def identify_metric(self, query: str) -> List[str]:
        """Identify metrics mentioned in the query."""
        metrics = []
        for metric in self.known_metrics:
            if metric.lower() in query.lower():
                metrics.append(metric)
        return metrics or ['total']  # Default to total if no specific metric found

    def extract_time_period(self, query: str) -> Dict[str, str]:
        """Extract time period from the query."""
        # Look for years in the format YYYY
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        
        # Look for relative time periods
        relative_periods = {
            'last year': str(datetime.now().year - 1),
            'this year': str(datetime.now().year),
            'current': str(datetime.now().year)
        }

        if years:
            return {
                'start_year': min(years),
                'end_year': max(years)
            }
        else:
            # Check for relative periods
            for period, year in relative_periods.items():
                if period in query.lower():
                    return {
                        'start_year': year,
                        'end_year': year
                    }
        
        # Default to current year if no period specified
        current_year = str(datetime.now().year)
        return {
            'start_year': current_year,
            'end_year': current_year
        }

    def extract_institutions(self, query: str) -> List[str]:
        """Extract mentioned institutions from the query."""
        institutions = []
        for institution in self.known_institutions:
            if institution.lower() in query.lower():
                institutions.append(institution)
        return institutions or ['University of Leicester']  # Default to UoL

    def identify_comparison(self, query: str) -> str:
        """Identify the type of comparison requested."""
        comparison_keywords = {
            'compare': 'comparison',
            'versus': 'comparison',
            'vs': 'comparison',
            'trend': 'trend',
            'over time': 'trend',
            'distribution': 'distribution',
            'breakdown': 'breakdown'
        }

        for keyword, comp_type in comparison_keywords.items():
            if keyword in query.lower():
                return comp_type
        
        return 'single'  # Default to single metric view

    def extract_parameters(self, query: str) -> Dict:
        """Extract all parameters from the query."""
        try:
            parameters = {
                'metrics': self.identify_metric(query),
                'time_period': self.extract_time_period(query),
                'institutions': self.extract_institutions(query),
                'comparison_type': self.identify_comparison(query)
            }
            logger.info(f"Extracted parameters: {parameters}")
            return parameters
        except Exception as e:
            logger.error(f"Error extracting parameters: {str(e)}")
            raise

    def validate_parameters(self, parameters: Dict) -> bool:
        """Validate extracted parameters."""
        try:
            # Check if we have at least one metric
            if not parameters.get('metrics'):
                return False
            
            # Check if time period is valid
            time_period = parameters.get('time_period', {})
            if not time_period.get('start_year') or not time_period.get('end_year'):
                return False
            
            # Check if we have at least one institution
            if not parameters.get('institutions'):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating parameters: {str(e)}")
            return False 