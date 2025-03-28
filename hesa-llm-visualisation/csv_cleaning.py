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
import json
import datetime
from pathlib import Path

# Set up Django environment
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hesa_llm_visualisation.settings')
django.setup()

# Now import your custom modules
from core.data_processing.csv_processor import CSVProcessor

def create_index_file(processor, results):
    """
    Creates an index file at the root level containing metadata for all CSV files
    in the cleaned_files directory.
    """
    logger = logging.getLogger(__name__)
    
    # Define path for the index file
    base_dir = Path(__file__).resolve().parent
    index_file_path = base_dir / "hesa_files_index.json"
    
    # Initialize index data structure
    index_data = {
        "hesa_files": [],
        "updated_at": datetime.datetime.now().isoformat(),
        "total_files": 0
    }
    
    # Process each cleaned file to extract metadata
    for file_name in processor.get_available_datasets():
        if file_name == '.gitkeep':
            continue
            
        cleaned_file_path = processor.clean_dir / file_name
        
        try:
            # Extract metadata from the cleaned file using the processor's method
            metadata = processor.extract_metadata(cleaned_file_path)
            
            # If no metadata found, attempt to extract it directly from the file
            if not metadata:
                logger.warning(f"No metadata found using processor for {file_name}, trying direct extraction")
                try:
                    # Read the first line to extract metadata
                    with open(cleaned_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        first_line = f.readline().strip()
                    
                    if first_line.startswith('#METADATA:'):
                        # Extract the JSON metadata
                        metadata_json = first_line[9:]  # Skip the #METADATA: prefix
                        metadata = json.loads(metadata_json)
                        logger.info(f"Successfully extracted metadata directly from file: {file_name}")
                    else:
                        logger.warning(f"File {file_name} doesn't have metadata line")
                        metadata = {}
                except Exception as e:
                    logger.warning(f"Error extracting metadata from first line of {file_name}: {str(e)}")
                    metadata = {}
            
            # If still no metadata or missing key fields, try to create minimal metadata
            if not metadata or 'title' not in metadata or 'academic_year' not in metadata:
                logger.warning(f"Incomplete metadata for {file_name}, creating minimal metadata")
                
                # If title is missing, extract it from the raw file
                if 'title' not in metadata:
                    # Try to find the corresponding raw file
                    raw_file_path = processor.raw_dir / file_name
                    
                    if raw_file_path.exists():
                        try:
                            # Extract title using the same logic as in processor.process_all_files
                            with open(raw_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = [line.strip() for line in f.readlines()[:20]]
                                
                            # Extract title from the first line
                            first_line = lines[0] if lines else ""
                            
                            if 'Title:' in first_line:
                                # Get everything after 'Title:'
                                title_text = first_line.split('Title:')[1].strip()
                                
                                # Remove leading commas and quotes
                                title_text = title_text.lstrip(',').strip('"')
                                
                                # Find the part after "Table XX - " if it exists
                                import re
                                table_match = re.search(r'Table\s+\d+\s*-\s*(.*)', title_text)
                                if table_match:
                                    # Extract just the part after the dash
                                    extracted_title = table_match.group(1).strip()
                                    # Remove any trailing quotes
                                    extracted_title = extracted_title.rstrip('"')
                                    metadata['title'] = extracted_title
                                else:
                                    # If no table pattern, just use the whole text
                                    cleaned_title = title_text.strip()
                                    metadata['title'] = cleaned_title
                                
                                logger.info(f"Extracted title from raw file: {metadata['title']}")
                            else:
                                # Fallback to cleaned filename
                                metadata['title'] = file_name.replace('.csv', '').split(' 20')[0]
                                logger.warning(f"Using fallback title from filename: {metadata['title']}")
                        except Exception as e:
                            logger.warning(f"Error extracting title from raw file: {str(e)}")
                            # Fallback to cleaned filename
                            metadata['title'] = file_name.replace('.csv', '').split(' 20')[0]
                    else:
                        # Fallback to cleaned filename
                        metadata['title'] = file_name.replace('.csv', '').split(' 20')[0]
                        logger.warning(f"Raw file not found, using fallback title from filename: {metadata['title']}")
                
                # If academic year is missing, extract it from filename
                if 'academic_year' not in metadata:
                    import re
                    year_match = re.search(r'(\d{4})&(\d{2})', file_name)
                    if year_match:
                        metadata['academic_year'] = f"{year_match.group(1)}/{year_match.group(2)}"
                    else:
                        metadata['academic_year'] = "Unknown"
                
                # If columns are missing, add minimal columns from the file
                if 'columns' not in metadata or not metadata['columns']:
                    try:
                        import pandas as pd
                        # Skip the metadata line and read the header row
                        df = pd.read_csv(cleaned_file_path, nrows=1, skiprows=1)  
                        metadata['columns'] = df.columns.tolist()
                    except Exception as e:
                        logger.warning(f"Error reading columns from {file_name}: {str(e)}")
                        metadata['columns'] = []
            
            # Ensure all required fields are present
            if 'columns' not in metadata:
                metadata['columns'] = []
            
            # Add reference (file name)
            metadata['reference'] = file_name
            
            # Add metadata to index
            index_data["hesa_files"].append(metadata)
            logger.info(f"Added metadata for {file_name} to index with title: '{metadata.get('title', 'No title')}'")
            
        except Exception as e:
            logger.error(f"Error processing {file_name} for index: {str(e)}")
    
    # Update total files count
    index_data["total_files"] = len(index_data["hesa_files"])
    
    # Write index file
    try:
        with open(index_file_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        logger.info(f"Successfully created index file at {index_file_path}")
        logger.info(f"Total files indexed: {index_data['total_files']}")
        return True
    except Exception as e:
        logger.error(f"Error writing index file: {str(e)}")
        return False

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
    results = {}
    if args.file:
        # Process a specific file
        file_name = args.file
        logger.info(f"Processing single file: {file_name}")
        
        # Check if file exists
        if not (processor.raw_dir / file_name).exists():
            logger.error(f"File not found: {file_name}")
            return False
        
        result = processor.process_single_file(file_name)
        results[file_name] = result
        if result:
            logger.info(f"Successfully processed {file_name}")
        else:
            logger.error(f"Failed to process {file_name}")
        
        # Also create index file
        logger.info("Creating index file...")
        create_index_file(processor, results)
        
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
        
        # Create index file
        logger.info("Creating index file...")
        create_index_file(processor, results)
        
        return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 