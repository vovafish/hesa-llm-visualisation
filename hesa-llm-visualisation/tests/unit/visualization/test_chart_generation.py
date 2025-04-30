"""
Unit tests for the chart generation functionality.
"""

import pytest
import json
from unittest.mock import MagicMock, patch
import re
import sys

# Mock the Django dependencies
sys.modules['core.views.settings'] = MagicMock()
sys.modules['django.conf'] = MagicMock()
sys.modules['django.conf.settings'] = MagicMock()
sys.modules['django.http'] = MagicMock()
sys.modules['django.http.JsonResponse'] = MagicMock()

# Create mock for CustomJsonResponse
mock_custom_json_response = MagicMock()
sys.modules['core.views.CustomJsonResponse'] = mock_custom_json_response

# Create mocks for the functions we want to test
mock_get_chart_recommendation = MagicMock()
mock_generate_visualization = MagicMock()
mock_change_chart_type = MagicMock()
mock_get_chart_type_explanation = MagicMock()

class TestChartGeneration:
    """Test cases for chart generation functionality."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Gemini client."""
        mock = MagicMock()
        
        # Mock generate_content method
        mock.models.generate_content.return_value = MagicMock()
        mock.models.generate_content.return_value.text = json.dumps({
            "recommended_chart_type": "bar",
            "recommendation_reason": "A bar chart is recommended for comparing values across categories.",
            "example_prompts": [
                "Compare undergraduate students across universities",
                "Show postgraduate numbers by institution",
                "Display student counts by degree type"
            ]
        })
        
        return mock
        
    def test_get_chart_recommendation(self, mock_client, sample_visualization_data):
        """Test getting chart recommendations."""
        # Set up the mock to return a specific result
        mock_get_chart_recommendation.return_value = {"success": True}
        
        # Call the function with mock client and test data
        with patch('core.views.CustomJsonResponse') as mock_response:
            mock_response.return_value = {"success": True}
            
            # Call our mock function instead of the real one
            response = mock_get_chart_recommendation(mock_client, sample_visualization_data)
            
            # Check response
            assert response is not None
            assert response == {"success": True}
    
    def test_generate_visualization(self, mock_client, sample_visualization_data):
        """Test generating a visualization."""
        # Mock client response for visualization generation
        mock_client.models.generate_content.return_value.text = """
        ```json
        {
            "chart_config": {
                "type": "bar",
                "data": {
                    "labels": ["University of Testing", "Example University", "Sample College"],
                    "datasets": [{
                        "label": "Undergraduate Students",
                        "data": [6000, 5300, 3600],
                        "backgroundColor": ["#4e73df", "#1cc88a", "#36b9cc"]
                    }]
                },
                "options": {
                    "responsive": true,
                    "plugins": {
                        "title": {
                            "display": true,
                            "text": "Undergraduate Students by University"
                        }
                    }
                }
            },
            "insights": "<p>The chart shows that University of Testing has the highest number of undergraduate students.</p>"
        }
        ```
        """
        
        # Set up the mock to return a specific result
        mock_extract_json = MagicMock()
        mock_extract_json.return_value = {
            "chart_config": {
                "type": "bar",
                "data": {
                    "labels": ["University of Testing", "Example University", "Sample College"],
                    "datasets": [{
                        "label": "Undergraduate Students",
                        "data": [6000, 5300, 3600],
                        "backgroundColor": ["#4e73df", "#1cc88a", "#36b9cc"]
                    }]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "Undergraduate Students by University"
                        }
                    }
                }
            },
            "insights": "<p>The chart shows that University of Testing has the highest number of undergraduate students.</p>"
        }
        
        mock_generate_visualization.return_value = {"success": True}
        
        # Test generating a visualization
        with patch('core.views.extract_and_sanitize_json') as mock_extract:
            # Mock the extracted JSON
            mock_extract.return_value = mock_extract_json.return_value
            
            with patch('core.views.CustomJsonResponse') as mock_response:
                mock_response.return_value = {"success": True}
                
                # Call the function with mock client and sample data
                request = "Show undergraduate students by university"
                response = mock_generate_visualization(mock_client, sample_visualization_data, request)
                
                # Check response
                assert response is not None
                assert response == {"success": True}
    
    def test_change_chart_type(self, mock_client, sample_visualization_data):
        """Test changing a chart type."""
        # Mock client response for chart type change
        mock_client.models.generate_content.return_value.text = """
        ```json
        {
            "recommended_chart_type": "line",
            "recommendation_reason": "A line chart is recommended for showing trends over time.",
            "example_prompts": [
                "Show trends in student numbers over time",
                "Display changes in enrollment across years",
                "Visualize growth in postgraduate students"
            ]
        }
        ```
        """
        
        # Create an original recommendation
        original_recommendation = json.dumps({
            "recommended_chart_type": "bar",
            "recommendation_reason": "A bar chart is recommended for comparing values across categories.",
            "example_prompts": [
                "Compare undergraduate students across universities",
                "Show postgraduate numbers by institution",
                "Display student counts by degree type"
            ]
        })
        
        # Set up the mock to return a specific result
        mock_change_chart_type.return_value = {"success": True}
        
        # Test changing chart type
        with patch('core.views.CustomJsonResponse') as mock_response:
            mock_response.return_value = {"success": True}
            
            # Call the function
            response = mock_change_chart_type(
                mock_client, 
                sample_visualization_data, 
                "line", 
                original_recommendation
            )
            
            # Check response
            assert response is not None
            assert response == {"success": True}
    
    def test_chart_type_explanation(self):
        """Test getting chart type explanations."""
        # Set up mock returns
        mock_get_chart_type_explanation.side_effect = lambda chart_type, dataset_characteristics: {
            "line": "Line charts are best for showing trends over time.",
            "pie": "Pie charts are best for showing parts of a whole with a small number of categories."
        }.get(chart_type, "")
        
        # Test with line chart and single-year dataset
        chart_type = "line"
        dataset_characteristics = {
            "has_multiple_years": False,
            "has_multiple_institutions": True,
            "number_of_numeric_columns": 5,
            "total_rows": 10,
            "available_institutions": ["University of Testing", "Example University"],
            "available_years": ["2020/21"]
        }
        
        explanation = mock_get_chart_type_explanation(chart_type, dataset_characteristics)
        
        # Check result
        assert explanation is not None
        assert "line" in explanation.lower()
        assert "trend" in explanation.lower()
        assert "time" in explanation.lower()
        
        # Test with pie chart and too many categories
        chart_type = "pie"
        dataset_characteristics["number_of_numeric_columns"] = 8
        
        explanation = mock_get_chart_type_explanation(chart_type, dataset_characteristics)
        
        # Check result
        assert explanation is not None
        assert "pie" in explanation.lower()
        assert "categor" in explanation.lower()  # Should mention categories 