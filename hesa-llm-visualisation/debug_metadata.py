"""
Debugging script to test metadata extraction.
"""

import os
import sys
import json
from pathlib import Path

# Add the project root to Python path so we can import modules properly
project_root = Path(__file__).parent
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

# Now import the CSVProcessor
from core.data_processing.csv_processor import CSVProcessor

def debug_metadata_extraction():
    # Create a temporary file with metadata
    import tempfile
    
    temp_dir = tempfile.mkdtemp()
    test_file = Path(temp_dir) / "test_metadata.csv"
    
    # Different variations of metadata format to test
    metadata_formats = [
        '#METADATA:{"title":"Test Dataset","year":"2020/21"}\n',
        '# METADATA:{"title":"Test Dataset","year":"2020/21"}\n',
        '#METADATA: {"title":"Test Dataset","year":"2020/21"}\n',
        '#METADATA:{"title": "Test Dataset", "year": "2020/21"}\n',
        '#METADATA:{"title":"Test Dataset","year":"2020/21"}',  # No newline
    ]
    
    processor = CSVProcessor()
    
    print("Testing metadata extraction:")
    
    for i, metadata_format in enumerate(metadata_formats):
        print(f"\nTest {i+1}: {metadata_format.strip()}")
        
        # Create test file with this format
        with open(test_file, "w", encoding='utf-8') as f:
            f.write(metadata_format + "Title: Test HESA Data\nData source: HESA\n\nColumn1,Column2\nValue1,Value2")
        
        # Debug: print the first line of the file
        with open(test_file, "r", encoding="utf-8") as f:
            first_line = f.readline()
            print(f"First line in file: {repr(first_line)}")
        
        # Test extraction
        try:
            metadata = processor.extract_metadata(test_file)
            print(f"Extracted metadata: {metadata}")
            if metadata and "title" in metadata:
                print("SUCCESS: Metadata extraction worked!")
            else:
                print("FAILED: Metadata empty or missing title")
        except Exception as e:
            print(f"EXCEPTION: {str(e)}")
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    debug_metadata_extraction() 