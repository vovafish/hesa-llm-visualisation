#!/usr/bin/env python
"""
CSV Cleaning Script

This script processes all raw CSV files to generate cleaned versions
with properly extracted metadata including academic years.

Usage:
    python csv_cleaning.py          # Process all files
    python csv_cleaning.py --file "filename.csv"  # Process a specific file
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Set up Django environment
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hesa_llm_visualisation.settings')
django.setup()

# Now import your custom modules
from core.data_processing.csv_processor import CSVProcessor

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('csv_cleaning.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Clean HESA CSV files and extract metadata.')
    parser.add_argument('--file', type=str, help='Process a specific file instead of all files')
    
    args = parser.parse_args()
    
    # Initialize the CSV processor
    processor = CSVProcessor()
    
    # Check if raw and clean directories exist
    logger.info(f"Raw files directory exists: {processor.raw_dir.exists()}")
    logger.info(f"Clean files directory exists: {processor.clean_dir.exists()}")
    
    # Create directories if they don't exist
    processor.raw_dir.mkdir(parents=True, exist_ok=True)
    processor.clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Process files
    if args.file:
        # Process a specific file
        file_name = args.file
        logger.info(f"Processing single file: {file_name}")
        
        # Check if file exists
        if not (processor.raw_dir / file_name).exists():
            logger.error(f"File not found: {file_name}")
            return False
        
        result = processor.process_single_file(file_name)
        if result:
            logger.info(f"Successfully processed {file_name}")
        else:
            logger.error(f"Failed to process {file_name}")
        
        return result
    else:
        # Process all files
        logger.info("Processing all files")
        results = processor.process_all_files()
        
        # Log results
        success_count = sum(results.values())
        total_count = len(results)
        logger.info(f"Processed {total_count} files: {success_count} successful, {total_count - success_count} failed")
        
        # List failed files
        failed_files = [f for f, success in results.items() if not success]
        if failed_files:
            logger.warning(f"Failed files: {', '.join(failed_files)}")
        
        return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 