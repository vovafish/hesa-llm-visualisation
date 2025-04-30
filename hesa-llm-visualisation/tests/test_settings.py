"""
Test settings for running the unit tests without a full Django setup.
"""

# Mock settings for tests
SECRET_KEY = 'django-insecure-test-key'
DEBUG = True

# Mock dashboard settings
DASHBOARD_SETTINGS = {
    'MAX_PREVIEW_ROWS': 3,
    'CACHE_TIMEOUT': 3600,
    'DEFAULT_CHART_TYPE': 'bar'
}

# Other required settings
INSTALLED_APPS = [
    'core',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

# Mock Gemini API settings
GEMINI_API_KEY = 'test-api-key'

# Set paths for test data
import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Use temporary directories for test data
TEMP_TEST_DIR = tempfile.mkdtemp()
DATA_DIR = Path(TEMP_TEST_DIR) / 'data'
RAW_FILES_DIR = DATA_DIR / 'raw_files'
CLEANED_FILES_DIR = DATA_DIR / 'cleaned_files'
CACHE_DIR = DATA_DIR / 'cache'

# Ensure directories exist
os.makedirs(RAW_FILES_DIR, exist_ok=True)
os.makedirs(CLEANED_FILES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True) 