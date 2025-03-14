import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import the CSVProcessor
from core.data_processing.csv_processor import CSVProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('csv_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting CSV processing")
    
    # Create CSV processor instance
    processor = CSVProcessor()
    
    # Check if raw files directory exists
    if not processor.raw_dir.exists():
        logger.error(f"Raw files directory does not exist: {processor.raw_dir}")
        return
    
    # Create cleaned files directory if it doesn't exist
    processor.clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Count files to process
    raw_files = list(processor.raw_dir.glob('*.csv'))
    logger.info(f"Found {len(raw_files)} CSV files to process")
    
    # Option 1: Process all files
    results = processor.process_all_files()
    
    # Option 2: Process specific files
    # For example, to process just one file:
    # specific_file = "Full-time HE student enrolments by HE provider and term-time accommodation 2015&16.csv"
    # result = processor.process_single_file(specific_file)
    # logger.info(f"Processed {specific_file}: {result}")
    
    # Log results
    success_count = sum(1 for result in results.values() if result)
    logger.info(f"Successfully processed {success_count} out of {len(results)} files")
    
    # Log the list of cleaned files
    cleaned_files = processor.get_available_datasets()
    logger.info(f"Cleaned files available: {len(cleaned_files)}")
    for file in cleaned_files:
        logger.info(f"  - {file}")
    
    logger.info("CSV processing completed")

if __name__ == "__main__":
    main() 