"""
Integration tests for the HESA LLM Visualization project.

Tests the flow from user query through data processing to visualization output.
"""

import os
import sys
import pytest
import json
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tests.test_settings')

try:
    import django
    django.setup()
except ImportError:
    print("Django not installed or not configured properly")
except Exception as e:
    print(f"Error setting up Django: {e}")

from core.data_processing.csv_processor import CSVProcessor
from core.views import process_gemini_query
from tests.mock_ai_client import MockAIClient

class TestQueryToVisualizationFlow:
    """Test cases for the end-to-end flow from query to visualization."""
    
    @pytest.fixture(scope="class")
    def csv_processor(self):
        """Fixture for the CSV processor."""
        return CSVProcessor()
    
    @pytest.fixture(scope="class")
    def mock_client(self):
        """Fixture for the mock AI client."""
        return MockAIClient()
    
    @pytest.fixture(scope="class")
    def sample_cleaned_files(self, csv_processor):
        """Return a list of available cleaned files for testing."""
        available_files = csv_processor.get_available_datasets()
        if not available_files:
            pytest.skip("No cleaned CSV files available for testing")
        return available_files
    
    def test_query_parse_to_data_retrieval(self, mock_client, csv_processor, sample_cleaned_files):
        """Test the flow from query parsing to data retrieval."""
        # Skip if no data is available
        if not sample_cleaned_files:
            pytest.skip("No sample data available for testing")
            
        # Use a sample query about student enrollment
        query = "How many students were enrolled at University of Cambridge in 2020/21?"
        
        # Process the query using the mock client
        query_analysis = mock_client.analyze_query(query)
        
        # Verify the query was analyzed correctly
        assert query_analysis is not None, "Query analysis should not be None"
        assert "institutions" in query_analysis, "Query analysis should extract institutions"
        assert "years" in query_analysis, "Query analysis should extract years"
        
        # Get institutions and years from the query
        institutions = query_analysis.get("institutions", [])
        years = query_analysis.get("years", [])
        
        assert len(institutions) > 0, "At least one institution should be extracted"
        assert len(years) > 0, "At least one year should be extracted"
        
        # Now try to find matching datasets based on the query
        matching_files = []
        for file_name in sample_cleaned_files:
            for institution in institutions:
                if institution.lower() in file_name.lower():
                    for year in years:
                        if year.replace("/", "-") in file_name:
                            matching_files.append(file_name)
                            break
        
        # If we don't find exact matches, just check for any files with similar keywords
        if not matching_files:
            for file_name in sample_cleaned_files:
                # Check if it contains keywords like "student", "enrollment", etc.
                if any(keyword in file_name.lower() for keyword in ["student", "enrolment", "enrollment"]):
                    matching_files.append(file_name)
                    if len(matching_files) >= 2:  # Limit to 2 files for testing
                        break
        
        # If we still don't have matches, use the first two files
        if not matching_files and len(sample_cleaned_files) > 0:
            matching_files = sample_cleaned_files[:min(2, len(sample_cleaned_files))]
        
        assert len(matching_files) > 0, "Should find at least one matching file"
        
        # Get the metadata from the first matching file
        file_path = csv_processor.clean_dir / matching_files[0]
        metadata = csv_processor.extract_metadata(file_path)
        
        # Check that we can extract metadata from the file
        assert metadata is not None, "Should be able to extract metadata"
    
    def test_data_to_visualization_flow(self, mock_client, csv_processor, sample_cleaned_files):
        """Test the flow from data retrieval to visualization generation."""
        # Skip if no data is available
        if not sample_cleaned_files:
            pytest.skip("No sample data available for testing")
            
        # Choose a file for testing
        test_file = sample_cleaned_files[0]
        file_path = csv_processor.clean_dir / test_file
        
        # Extract metadata
        metadata = csv_processor.extract_metadata(file_path)
        
        # Load the data
        df = csv_processor.clean_csv(file_path)
        assert df is not None, "Should be able to load the CSV data"
        
        # Prepare sample visualization data (simplified)
        visualization_data = {
            'title': metadata.get('title', 'Sample Dataset'),
            'columns': df.columns.tolist(),
            'rows': df.head(5).values.tolist()
        }
        
        # Get chart recommendation
        chart_recommendation = mock_client.get_chart_recommendation(visualization_data)
        
        # Verify chart recommendation
        assert chart_recommendation is not None, "Should get a chart recommendation"
        assert "recommended_chart_type" in chart_recommendation, "Should include a recommended chart type"
        assert "recommendation_reason" in chart_recommendation, "Should include a reason for recommendation"
        
        # Generate a visualization
        chart_config = mock_client.generate_visualization_config(
            visualization_data, 
            "Show student data", 
            chart_recommendation["recommended_chart_type"]
        )
        
        # Verify visualization generation
        assert chart_config is not None, "Should generate a visualization configuration"
        assert "chart_config" in chart_config, "Should include chart configuration"
        assert "insights" in chart_config, "Should include insights"
        
        # Verify chart configuration
        assert "type" in chart_config["chart_config"], "Chart config should include type"
        assert "data" in chart_config["chart_config"], "Chart config should include data"
        assert "options" in chart_config["chart_config"], "Chart config should include options"
    
    def test_end_to_end_query_processing(self, mock_client, csv_processor, sample_cleaned_files):
        """Test the complete end-to-end flow from query to visualization."""
        # Skip if no data is available
        if not sample_cleaned_files:
            pytest.skip("No sample data available for testing")
            
        # Sample natural language query
        query = "Show me student enrollments by university in 2022/23"
        
        # Step 1: Extract query parameters
        query_analysis = mock_client.analyze_query(query)
        
        # Verify query analysis
        assert query_analysis is not None, "Query should be analyzed"
        
        # Step 2: Find matching datasets (simulated)
        matching_files = []
        for file_name in sample_cleaned_files:
            # Look for enrollment-related files
            if any(term in file_name.lower() for term in ["enrolment", "enrollment", "student"]):
                # Look for the specific year
                if "2022-23" in file_name:
                    matching_files.append(file_name)
                    break
        
        # If no exact matches, use any enrollment-related file
        if not matching_files:
            for file_name in sample_cleaned_files:
                if any(term in file_name.lower() for term in ["enrolment", "enrollment", "student"]):
                    matching_files.append(file_name)
                    break
        
        # If still no matches, use the first file
        if not matching_files and sample_cleaned_files:
            matching_files = [sample_cleaned_files[0]]
        
        assert len(matching_files) > 0, "Should find at least one matching file"
        
        # Step 3: Load and prepare data
        file_path = csv_processor.clean_dir / matching_files[0]
        df = csv_processor.clean_csv(file_path)
        assert df is not None, "Should be able to load the CSV data"
        
        # Step 4: Prepare visualization data
        visualization_data = {
            'title': matching_files[0].replace('.csv', ''),
            'columns': df.columns.tolist(),
            'rows': df.head(10).values.tolist()
        }
        
        # Step 5: Get chart recommendation
        chart_recommendation = mock_client.get_chart_recommendation(visualization_data)
        assert "recommended_chart_type" in chart_recommendation, "Should recommend a chart type"
        
        # Step 6: Generate visualization
        chart_config = mock_client.generate_visualization_config(
            visualization_data, 
            query, 
            chart_recommendation["recommended_chart_type"]
        )
        
        # Verify complete flow output
        assert chart_config is not None, "Should generate a visualization"
        assert "chart_config" in chart_config, "Should include chart configuration"
        assert "insights" in chart_config, "Should include data insights"
        
        # Check that the chart config includes all necessary components
        chart_type = chart_config["chart_config"]["type"]
        assert chart_type in ["bar", "line", "pie", "scatter", "doughnut"], f"Chart type should be valid, got {chart_type}"
        assert "data" in chart_config["chart_config"], "Chart config should include data"
        assert "labels" in chart_config["chart_config"]["data"], "Chart data should include labels"
        assert "datasets" in chart_config["chart_config"]["data"], "Chart data should include datasets" 