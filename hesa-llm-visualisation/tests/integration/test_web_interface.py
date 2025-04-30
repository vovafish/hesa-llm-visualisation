"""
Integration tests for the HESA LLM Visualization project.

This module contains tests for the MockAIClient functionality used in integration scenarios.
"""

import os
import sys
import pytest
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.mock_ai_client import MockAIClient

class TestMockAIClientIntegration:
    """Test cases for the MockAIClient in integration scenarios."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock AI client fixture."""
        return MockAIClient()
    
    def test_chart_recommendation(self, mock_client):
        """Test that chart recommendations work as expected."""
        # Create visualization data
        visualization_data = {
            'title': 'Student Enrollments 2020-21',
            'columns': ['University', 'Undergraduate', 'Postgraduate', 'Total'],
            'rows': [
                ['University A', '1000', '500', '1500'],
                ['University B', '2000', '1000', '3000'],
                ['University C', '3000', '1500', '4500']
            ]
        }
        
        # Test the chart recommendation function
        chart_recommendation = mock_client.get_chart_recommendation(visualization_data)
        
        # Assert that the mock client returns the expected structure
        assert "recommended_chart_type" in chart_recommendation, "Should include a recommended chart type"
        assert "recommendation_reason" in chart_recommendation, "Should include a reason for the recommendation"
        assert "example_prompts" in chart_recommendation, "Should include example prompts"
    
    def test_visualization_generation(self, mock_client):
        """Test visualization generation functionality."""
        # Create visualization data
        visualization_data = {
            'title': 'Student Enrollments 2020-21',
            'columns': ['University', 'Undergraduate', 'Postgraduate', 'Total'],
            'rows': [
                ['University A', '1000', '500', '1500'],
                ['University B', '2000', '1000', '3000'],
                ['University C', '3000', '1500', '4500']
            ]
        }
        
        # Generate a visualization config
        chart_config = mock_client.generate_visualization_config(
            visualization_data,
            "Show student enrollments",
            "bar"
        )
        
        # Verify the chart configuration structure
        assert chart_config is not None, "Should generate a visualization configuration"
        assert "chart_config" in chart_config, "Should include chart configuration"
        assert "insights" in chart_config, "Should include insights"
        
        # Basic test of chart configuration structure
        assert "type" in chart_config["chart_config"], "Chart config should include type"
        assert "data" in chart_config["chart_config"], "Chart config should include data"
        assert "labels" in chart_config["chart_config"]["data"], "Chart data should include labels"
    
    def test_chart_type_change(self, mock_client):
        """Test changing chart type functionality."""
        # Create visualization data
        visualization_data = {
            'title': 'Student Enrollments 2020-21',
            'columns': ['University', 'Undergraduate', 'Postgraduate', 'Total'],
            'rows': [
                ['University A', '1000', '500', '1500'],
                ['University B', '2000', '1000', '3000'],
                ['University C', '3000', '1500', '4500']
            ]
        }
        
        # Generate a visualization with a different chart type
        chart_config = mock_client.generate_visualization_config(
            visualization_data,
            "Show student enrollments",
            "pie"  # Use pie chart instead of default bar
        )
        
        # Verify the chart type was changed
        assert chart_config["chart_config"]["type"] == "pie", "Chart type should be pie"
        assert "data" in chart_config["chart_config"], "Chart config should include data"