from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class QueryError(Exception):
    """Custom exception for query-related errors."""
    def __init__(self, message: str, error_type: str, details: Optional[Dict] = None):
        self.message = message
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)

class ErrorHandler:
    """Handle and format errors in the LLM query system."""
    
    ERROR_TYPES = {
        'INVALID_QUERY': 'The query could not be processed',
        'PARAMETER_ERROR': 'Invalid or missing parameters',
        'MODEL_ERROR': 'Error in model processing',
        'VALIDATION_ERROR': 'Data validation failed',
        'SYSTEM_ERROR': 'Internal system error'
    }

    def __init__(self):
        """Initialize the error handler."""
        self.error_log = []

    def handle_error(self, error: Exception) -> Dict:
        """Handle different types of errors and return formatted error response."""
        try:
            if isinstance(error, QueryError):
                return self._format_query_error(error)
            elif isinstance(error, ValueError):
                return self._format_value_error(error)
            elif isinstance(error, TypeError):
                return self._format_type_error(error)
            else:
                return self._format_generic_error(error)
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
            return self._format_generic_error(e)

    def _format_query_error(self, error: QueryError) -> Dict:
        """Format QueryError instances."""
        error_response = {
            'error': True,
            'type': error.error_type,
            'message': error.message,
            'timestamp': error.timestamp.isoformat(),
            'details': error.details
        }
        self._log_error(error_response)
        return error_response

    def _format_value_error(self, error: ValueError) -> Dict:
        """Format ValueError instances."""
        error_response = {
            'error': True,
            'type': 'PARAMETER_ERROR',
            'message': str(error),
            'timestamp': datetime.now().isoformat(),
            'details': {
                'error_class': 'ValueError',
                'suggestion': 'Please check the provided values and try again'
            }
        }
        self._log_error(error_response)
        return error_response

    def _format_type_error(self, error: TypeError) -> Dict:
        """Format TypeError instances."""
        error_response = {
            'error': True,
            'type': 'PARAMETER_ERROR',
            'message': str(error),
            'timestamp': datetime.now().isoformat(),
            'details': {
                'error_class': 'TypeError',
                'suggestion': 'Please check the data types of the provided values'
            }
        }
        self._log_error(error_response)
        return error_response

    def _format_generic_error(self, error: Exception) -> Dict:
        """Format generic exceptions."""
        error_response = {
            'error': True,
            'type': 'SYSTEM_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.now().isoformat(),
            'details': {
                'error_class': error.__class__.__name__,
                'error_message': str(error)
            }
        }
        self._log_error(error_response)
        return error_response

    def _log_error(self, error_data: Dict) -> None:
        """Log error information."""
        self.error_log.append(error_data)
        logger.error(f"Error occurred: {error_data['type']} - {error_data['message']}")

    def get_error_history(self) -> list:
        """Retrieve error history."""
        return self.error_log

    def clear_error_history(self) -> None:
        """Clear error history."""
        self.error_log = []

    def get_error_stats(self) -> Dict:
        """Get statistics about errors."""
        stats = {
            'total_errors': len(self.error_log),
            'error_types': {},
            'latest_error': None
        }
        
        for error in self.error_log:
            error_type = error['type']
            stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
        
        if self.error_log:
            stats['latest_error'] = self.error_log[-1]
        
        return stats 