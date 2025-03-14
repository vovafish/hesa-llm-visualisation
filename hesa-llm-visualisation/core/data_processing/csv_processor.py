import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import csv
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_FILES_DIR = BASE_DIR / 'data' / 'raw_files'
CLEANED_FILES_DIR = BASE_DIR / 'data' / 'cleaned_files'

class CSVProcessor:
    def __init__(self):
        self.raw_dir = RAW_FILES_DIR
        self.clean_dir = CLEANED_FILES_DIR
        
    def validate_csv(self, file_path: Path) -> bool:
        """
        Validates if the CSV file is properly formatted and contains expected data.
        """
        try:
            # First check if the file exists
            if not file_path.exists():
                logger.error(f"File {file_path} does not exist")
                return False
                
            # Read the first 20 lines to check the structure
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f.readlines()[:20]]
                
            # A HESA file should contain metadata like Title, Data source, etc.
            metadata_patterns = ['Title:', 'Data source:', 'Subtitle:']
            has_metadata = any(pattern in line for pattern in metadata_patterns for line in lines)
            
            if not has_metadata:
                logger.warning(f"File {file_path} doesn't have expected HESA metadata structure")
                # Still return True as it might be a valid CSV without HESA metadata
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {str(e)}")
            return False
    
    def find_data_start(self, file_path: Path) -> Tuple[int, List[str]]:
        """
        Find where the actual data starts in a HESA CSV file and extract headers.
        Returns the line number where data starts and the column headers.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # Log the first several lines for debugging
            logger.info(f"First 10 lines of {file_path.name}:")
            for i, line in enumerate(lines[:10]):
                logger.info(f"Line {i}: {line.strip()}")
                
            # First, look for the header row containing specific column names
            header_row = -1
            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Common header patterns in HESA files
                if ('"he provider"' in line_lower or 
                    '"ukprn"' in line_lower or 
                    '"academic year"' in line_lower or
                    'provider' in line_lower and ('ukprn' in line_lower or 'code' in line_lower)):
                    header_row = i
                    logger.info(f"Found header row at line {i}: {line.strip()}")
                    break
                
            # If header row not found with specific patterns, look for the first row that looks like a header
            if header_row == -1:
                for i, line in enumerate(lines):
                    if i > 5 and ',' in line and '"' in line:
                        parts = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line.strip())
                        if len(parts) >= 3:  # Assume a header has at least 3 columns
                            header_row = i
                            logger.info(f"Found potential header row at line {i}: {line.strip()}")
                            break
                            
            # If still not found, default to line 10 (common in HESA files)
            if header_row == -1:
                header_row = 10
                logger.warning(f"Could not identify header row, defaulting to line 10")
            
            # Get the column headers
            header_line = lines[header_row].strip()
            
            # Parse the header line, respecting quoted values
            try:
                reader = csv.reader([header_line])
                headers = next(reader)
                logger.info(f"Parsed headers: {headers}")
            except Exception as e:
                logger.error(f"Error parsing header line: {str(e)}")
                headers = header_line.split(',')
                logger.info(f"Fallback header parsing: {headers}")
            
            # Data starts on the line after the header
            data_start = header_row + 1
            
            return data_start, headers
            
        except Exception as e:
            logger.error(f"Error finding data start in {file_path}: {str(e)}")
            return 0, []
    
    def clean_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Cleans the CSV data by handling metadata, selecting relevant columns, etc.
        """
        try:
            # Find where the actual data starts
            data_start, headers = self.find_data_start(file_path)
            
            if data_start == 0 or not headers:
                logger.error(f"Could not determine data structure in {file_path}")
                return None
                
            logger.info(f"Reading data from line {data_start} with headers: {headers}")
            
            # Read the CSV file starting from the data section
            try:
                df = pd.read_csv(file_path, skiprows=data_start, names=headers, header=None)
                logger.info(f"Successfully read {len(df)} rows with {len(df.columns)} columns")
            except Exception as e:
                logger.error(f"Error reading CSV with pandas: {str(e)}")
                
                # Try alternative parsing with more flexible options
                try:
                    df = pd.read_csv(
                        file_path, 
                        skiprows=data_start, 
                        names=headers, 
                        header=None,
                        engine='python', 
                        on_bad_lines='skip'
                    )
                    logger.info(f"Alternative parsing successful: {len(df)} rows")
                except Exception as e2:
                    logger.error(f"Alternative parsing also failed: {str(e2)}")
                    return None
            
            # Remove columns with 'UKPRN' or other non-essential identifiers if specified
            columns_to_remove = []
            for col in df.columns:
                if isinstance(col, str) and ('UKPRN' in col or 'Code' in col):
                    columns_to_remove.append(col)
            
            if columns_to_remove:
                logger.info(f"Removing non-essential columns: {columns_to_remove}")
                df = df.drop(columns=columns_to_remove)
            
            # Clean column names - remove quotes, extra spaces
            df.columns = [col.strip('" ') if isinstance(col, str) else col for col in df.columns]
            
            # Remove any rows at the end that might be totals or footnotes
            # Common patterns for footer rows
            footer_patterns = ['Total', 'Grand Total', 'Source:']
            for pattern in footer_patterns:
                if df.iloc[-1:].astype(str).values[0][0].startswith(pattern):
                    logger.info(f"Removing footer row: {df.iloc[-1:].values}")
                    df = df.iloc[:-1]
            
            # Drop any completely empty rows or columns
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            
            # Remove duplicates
            df = df.drop_duplicates()
            
            logger.info(f"Cleaned CSV data: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning {file_path}: {str(e)}")
            return None
    
    def process_all_files(self) -> Dict[str, bool]:
        """
        Process all CSV files in the raw directory and save cleaned versions.
        """
        results = {}
        
        # Create cleaned directory if it doesn't exist
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        for file_path in self.raw_dir.glob('*.csv'):
            try:
                logger.info(f"Processing file: {file_path.name}")
                
                # Validate file
                if not self.validate_csv(file_path):
                    results[file_path.name] = False
                    continue
                
                # Clean data
                cleaned_df = self.clean_csv(file_path)
                if cleaned_df is None:
                    results[file_path.name] = False
                    continue
                
                # Save cleaned file
                cleaned_path = self.clean_dir / file_path.name
                cleaned_df.to_csv(cleaned_path, index=False)
                results[file_path.name] = True
                
                logger.info(f"Successfully processed {file_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {str(e)}")
                results[file_path.name] = False
                
        return results

    def get_available_datasets(self) -> List[str]:
        """
        Returns a list of available cleaned dataset names.
        """
        return [f.name for f in self.clean_dir.glob('*.csv')]
    
    def process_single_file(self, file_name: str) -> bool:
        """
        Process a single file by name from the raw directory.
        """
        file_path = self.raw_dir / file_name
        
        try:
            logger.info(f"Processing single file: {file_name}")
            
            # Validate file
            if not self.validate_csv(file_path):
                return False
            
            # Clean data
            cleaned_df = self.clean_csv(file_path)
            if cleaned_df is None:
                return False
            
            # Save cleaned file
            cleaned_path = self.clean_dir / file_name
            cleaned_df.to_csv(cleaned_path, index=False)
            
            logger.info(f"Successfully processed {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            return False 