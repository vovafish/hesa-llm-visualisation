"""
Test configuration and shared fixtures for the HESA LLM Visualization tests.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to Python path so we can import modules properly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure Django settings to use our test settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tests.test_settings')

# Try to configure Django settings
try:
    import django
    django.setup()
except ImportError:
    print("Django not installed or not configured properly")
except Exception as e:
    print(f"Error setting up Django: {e}")

# Create fixtures for test data
@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "resources" / "test_data"

@pytest.fixture
def raw_csv_file():
    """Return the path to a sample raw CSV file for testing."""
    return Path(__file__).parent / "resources" / "test_data" / "sample_raw.csv"

@pytest.fixture
def cleaned_csv_file():
    """Return the path to a sample cleaned CSV file for testing."""
    return Path(__file__).parent / "resources" / "test_data" / "sample_cleaned.csv"

# Create test data fixtures
@pytest.fixture
def sample_raw_csv_content():
    """Return sample content for a raw CSV file."""
    return """Title: Higher Education Student Data
Data source: HESA

UKPRN,HE Provider,First degree,Other undergraduate,Postgraduate
10007789,University of Testing,5000,1000,2000
10007790,Example University,4500,800,1800
10007791,Sample College,3000,600,1200
"""

@pytest.fixture
def sample_query():
    """Return a sample natural language query for testing."""
    return "How many undergraduate students were at University of Testing in 2020?"

@pytest.fixture
def sample_visualization_data():
    """Return sample data for visualization testing."""
    return {
        'title': 'Student Numbers by University',
        'columns': ['HE Provider', 'First degree', 'Other undergraduate', 'Postgraduate'],
        'rows': [
            ['University of Testing', '5000', '1000', '2000'],
            ['Example University', '4500', '800', '1800'],
            ['Sample College', '3000', '600', '1200']
        ]
    } 