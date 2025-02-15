import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths relative to the project root
BASE_DIR = Path(__file__).resolve().parent.parent
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
            df = pd.read_csv(file_path)
            
            # Check if file is empty
            if df.empty:
                logger.error(f"File {file_path} is empty")
                return False
                
            # Check for minimum required columns (customize based on your HESA data structure)
            required_columns = ['institution_id', 'year']  # Add your required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {str(e)}")
            return False
    
    def clean_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Cleans the CSV data by handling missing values, removing duplicates, etc.
        """
        try:
            df = pd.read_csv(file_path)
            
            # Remove duplicate rows
            df = df.drop_duplicates()
            
            # Handle missing values (customize based on your needs)
            df = df.fillna({
                'numeric_columns': 0,  # Fill numeric missing values with 0
                'string_columns': 'Unknown'  # Fill string missing values with 'Unknown'
            })
            
            # Convert date columns to datetime (if any)
            date_columns = []  # Add your date column names
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
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