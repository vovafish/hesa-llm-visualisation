from typing import Dict, List, Optional, Union
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ParameterExtractor:
    def __init__(self):
        """Initialize the parameter extractor with validation rules."""
        self.validation_rules = {
            'metrics': {
                'required': True,
                'type': list,
                'allowed_values': [
                    'enrollment', 'postgraduate', 'undergraduate',
                    'full_time', 'part_time', 'international',
                    'domestic', 'research', 'teaching'
                ]
            },
            'time_period': {
                'required': True,
                'type': dict,
                'fields': ['start_year', 'end_year']
            },
            'institutions': {
                'required': True,
                'type': list,
                'min_length': 1
            }
        }

    def extract_from_json(self, json_str: str) -> Dict:
        """Extract parameters from JSON string response."""
        try:
            data = json.loads(json_str)
            return self.validate_and_clean(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error extracting parameters from JSON: {str(e)}")
            raise

    def validate_and_clean(self, data: Dict) -> Dict:
        """Validate and clean extracted parameters."""
        cleaned_data = {}
        
        for field, rules in self.validation_rules.items():
            # Check if required field is present
            if rules['required'] and field not in data:
                raise ValueError(f"Missing required field: {field}")
            
            value = data.get(field)
            if value is None:
                continue
            
            # Type checking
            if not isinstance(value, rules['type']):
                raise TypeError(f"Invalid type for {field}: expected {rules['type']}, got {type(value)}")
            
            # Specific validation for each field type
            if field == 'metrics':
                cleaned_data[field] = self._validate_metrics(value)
            elif field == 'time_period':
                cleaned_data[field] = self._validate_time_period(value)
            elif field == 'institutions':
                cleaned_data[field] = self._validate_institutions(value)
            
        return cleaned_data

    def _validate_metrics(self, metrics: List[str]) -> List[str]:
        """Validate metrics against allowed values."""
        valid_metrics = []
        for metric in metrics:
            if metric.lower() in self.validation_rules['metrics']['allowed_values']:
                valid_metrics.append(metric.lower())
        
        if not valid_metrics:
            raise ValueError("No valid metrics provided")
        
        return valid_metrics

    def _validate_time_period(self, time_period: Dict) -> Dict:
        """Validate time period structure and values."""
        required_fields = self.validation_rules['time_period']['fields']
        
        # Check required fields
        for field in required_fields:
            if field not in time_period:
                raise ValueError(f"Missing required time period field: {field}")
        
        # Validate year format and range
        start_year = int(time_period['start_year'])
        end_year = int(time_period['end_year'])
        current_year = datetime.now().year
        
        if not (1900 <= start_year <= current_year + 1):
            raise ValueError(f"Invalid start year: {start_year}")
        if not (1900 <= end_year <= current_year + 1):
            raise ValueError(f"Invalid end year: {end_year}")
        if start_year > end_year:
            raise ValueError(f"Start year {start_year} cannot be after end year {end_year}")
        
        return {
            'start_year': str(start_year),
            'end_year': str(end_year)
        }

    def _validate_institutions(self, institutions: List[str]) -> List[str]:
        """Validate institution list."""
        if len(institutions) < self.validation_rules['institutions']['min_length']:
            raise ValueError("At least one institution must be specified")
        
        # Could add additional validation here (e.g., check against known institutions)
        return [inst.strip() for inst in institutions]

    def suggest_corrections(self, invalid_data: Dict) -> Dict:
        """Suggest corrections for invalid parameters."""
        suggestions = {}
        
        try:
            # Suggest corrections for metrics
            if 'metrics' in invalid_data:
                suggestions['metrics'] = self._suggest_metrics(invalid_data['metrics'])
            
            # Suggest corrections for time period
            if 'time_period' in invalid_data:
                suggestions['time_period'] = self._suggest_time_period(invalid_data['time_period'])
            
            # Suggest corrections for institutions
            if 'institutions' in invalid_data:
                suggestions['institutions'] = self._suggest_institutions(invalid_data['institutions'])
            
            return suggestions
        except Exception as e:
            logger.error(f"Error generating suggestions: {str(e)}")
            return {} 