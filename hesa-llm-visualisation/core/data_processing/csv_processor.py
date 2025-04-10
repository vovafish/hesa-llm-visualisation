import pandas as pd
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
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
CONFIG_DIR = BASE_DIR / 'config'

class CSVProcessor:
    def __init__(self):
        self.raw_dir = RAW_FILES_DIR
        self.clean_dir = CLEANED_FILES_DIR
        self.synonyms_mapping = self._load_synonyms_mapping()
        
    def _load_synonyms_mapping(self) -> Dict[str, List[str]]:
        """Load synonym mappings from the config file."""
        # Define a minimal set of synonyms as a fallback
        basic_synonyms = {
            "university": ["he provider", "institution", "college"],
            "student": ["learner", "pupil"],
            "course": ["program", "programme", "study"]
        }
        
        synonyms_path = CONFIG_DIR / 'synonyms.json'
        try:
            if synonyms_path.exists():
                with open(synonyms_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                logger.warning(f"Synonyms file not found: {synonyms_path}, using basic synonyms")
                return basic_synonyms
        except Exception as e:
            logger.error(f"Error loading synonyms mapping: {str(e)}, using basic synonyms")
            return basic_synonyms
    
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
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from the first line of the file if it exists."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                
            if first_line.startswith('#METADATA:'):
                # Extract the JSON metadata
                metadata_json = first_line[9:]  # Skip the #METADATA: prefix
                metadata = json.loads(metadata_json)
                return metadata
            else:
                return {}
        except Exception as e:
            logger.warning(f"Error extracting metadata from {file_path}: {str(e)}")
            return {}
    
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
            
            try:
                # Read the CSV file starting from the data section
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
                
                # Create new empty metadata
                metadata = {}
                
                # Read the first 20 lines to extract title and academic year
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f.readlines()[:20]]
                        
                    # Extract title from the first line only
                    first_line = lines[0] if lines else ""
                    
                    # Log the raw line for debugging
                    logger.info(f"Raw first line: [{first_line}]")
                    
                    # Check if the line contains the title pattern
                    if 'Title:' in first_line:
                        # Get everything after 'Title:'
                        title_text = first_line.split('Title:')[1].strip()
                        logger.info(f"After split by 'Title:': [{title_text}]")
                        
                        # Remove leading commas and quotes
                        title_text = title_text.lstrip(',').strip('"')
                        logger.info(f"After cleaning commas/quotes: [{title_text}]")
                        
                        # Find the part after "Table XX - " if it exists
                        table_match = re.search(r'Table\s+\d+\s*-\s*(.*)', title_text)
                        if table_match:
                            # Extract just the part after the dash
                            extracted_title = table_match.group(1).strip()
                            # Remove any trailing quotes
                            extracted_title = extracted_title.rstrip('"')
                            metadata['title'] = extracted_title
                            logger.info(f"Extracted title: [{extracted_title}]")
                        else:
                            # If no table pattern, just use the whole text
                            cleaned_title = title_text.strip()
                            metadata['title'] = cleaned_title
                            logger.info(f"No table pattern found, using cleaned title: [{cleaned_title}]")
                    else:
                        logger.warning(f"First line does not contain 'Title:': [{first_line}]")
                    
                    # Extract academic year
                    for line in lines:
                        # Skip lines with "to" as they indicate a range of years (e.g., "Academic years 2020/21 to 2023/24")
                        if "to" in line:
                            continue
                            
                        # Look for patterns like ",Academic year,YYYY/YY" or "Filters:,Academic year,YYYY/YY"
                        year_match = re.search(r'(?:,|\:)\s*Academic year\s*,\s*(20\d{2}/\d{2})', line)
                        if year_match:
                            metadata['academic_year'] = year_match.group(1).strip()
                            logger.info(f"Found academic year format: {metadata['academic_year']}")
                            break
                except Exception as e:
                    logger.warning(f"Error extracting title/academic year: {str(e)}")
                
                # If title not found, use "Unknown title"
                if 'title' not in metadata:
                    logger.warning(f"Could not extract title from file content")
                    metadata['title'] = "Unknown title"
                
                # If academic year not found, use "Unknown"
                if 'academic_year' not in metadata:
                    logger.warning(f"Could not extract academic year from file content")
                    
                    # Try to extract from filename before defaulting to Unknown
                    year_match = re.search(r'(\d{4})[\.\-&_](\d{2})', file_path.name)
                    if year_match:
                        metadata['academic_year'] = f"{year_match.group(1)}/{year_match.group(2)}"
                        logger.info(f"Extracted academic year from filename: {metadata['academic_year']}")
                    else:
                        metadata['academic_year'] = "Unknown"
                
                # Clean data
                cleaned_df = self.clean_csv(file_path)
                if cleaned_df is None:
                    results[file_path.name] = False
                    continue
                
                # Extract title-based keywords
                title_keywords = self.extract_keywords_from_title(metadata['title'])
                metadata['keywords_title'] = title_keywords
                
                # Extract column-based keywords
                column_keywords = self.extract_keywords_from_columns(cleaned_df.columns)
                metadata['keywords_columns'] = column_keywords
                
                # Generate new filename based on title and academic year
                original_filename = file_path.name
                new_filename = self.generate_clean_filename(metadata['title'], metadata['academic_year'], original_filename)
                
                # Store original filename in metadata
                metadata['original_filename'] = original_filename
                # Store new filename in metadata
                metadata['new_filename'] = new_filename
                
                # Create metadata line
                metadata_line = f"#METADATA:{json.dumps(metadata)}\n"
                
                # Save cleaned file with metadata using new filename
                cleaned_path = self.clean_dir / new_filename
                
                # Write metadata line first, then the data
                with open(cleaned_path, 'w', encoding='utf-8') as f:
                    f.write(metadata_line)
                
                # Then append the dataframe
                cleaned_df.to_csv(cleaned_path, index=False, mode='a')
                
                results[file_path.name] = True
                logger.info(f"Successfully processed {file_path.name} with metadata")
                logger.info(f"Title: {metadata.get('title', 'Not found')}")
                logger.info(f"Academic Year: {metadata.get('academic_year', 'Not found')}")
                logger.info(f"New filename: {new_filename}")
                logger.info(f"Title keywords: {title_keywords}")
                logger.info(f"Column keywords: {column_keywords}")
                
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
            
            # Create new empty metadata
            metadata = {}
            
            # Read the first 20 lines to extract title and academic year
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [line.strip() for line in f.readlines()[:20]]
                    
                # Extract title from the first line only
                first_line = lines[0] if lines else ""
                
                # Log the raw line for debugging
                logger.info(f"Raw first line: [{first_line}]")
                
                # Check if the line contains the title pattern
                if 'Title:' in first_line:
                    # Get everything after 'Title:'
                    title_text = first_line.split('Title:')[1].strip()
                    logger.info(f"After split by 'Title:': [{title_text}]")
                    
                    # Remove leading commas and quotes
                    title_text = title_text.lstrip(',').strip('"')
                    logger.info(f"After cleaning commas/quotes: [{title_text}]")
                    
                    # Find the part after "Table XX - " if it exists
                    table_match = re.search(r'Table\s+\d+\s*-\s*(.*)', title_text)
                    if table_match:
                        # Extract just the part after the dash
                        extracted_title = table_match.group(1).strip()
                        # Remove any trailing quotes
                        extracted_title = extracted_title.rstrip('"')
                        metadata['title'] = extracted_title
                        logger.info(f"Extracted title: [{extracted_title}]")
                    else:
                        # If no table pattern, just use the whole text
                        cleaned_title = title_text.strip()
                        metadata['title'] = cleaned_title
                        logger.info(f"No table pattern found, using cleaned title: [{cleaned_title}]")
                else:
                    logger.warning(f"First line does not contain 'Title:': [{first_line}]")
                
                # Extract academic year
                for line in lines:
                    # Skip lines with "to" as they indicate a range of years (e.g., "Academic years 2020/21 to 2023/24")
                    if "to" in line:
                        continue
                        
                    # Look for patterns like ",Academic year,YYYY/YY" or "Filters:,Academic year,YYYY/YY"
                    year_match = re.search(r'(?:,|\:)\s*Academic year\s*,\s*(20\d{2}/\d{2})', line)
                    if year_match:
                        metadata['academic_year'] = year_match.group(1).strip()
                        logger.info(f"Found academic year format: {metadata['academic_year']}")
                        break
            except Exception as e:
                logger.warning(f"Error extracting title/academic year: {str(e)}")
            
            # If title not found, use "Unknown title"
            if 'title' not in metadata:
                logger.warning(f"Could not extract title from file content")
                metadata['title'] = "Unknown title"
            
            # If academic year not found, use "Unknown"
            if 'academic_year' not in metadata:
                logger.warning(f"Could not extract academic year from file content")
                
                # Try to extract from filename before defaulting to Unknown
                year_match = re.search(r'(\d{4})[\.\-&_](\d{2})', file_name)
                if year_match:
                    metadata['academic_year'] = f"{year_match.group(1)}/{year_match.group(2)}"
                    logger.info(f"Extracted academic year from filename: {metadata['academic_year']}")
                else:
                    metadata['academic_year'] = "Unknown"
            
            # Clean data
            cleaned_df = self.clean_csv(file_path)
            if cleaned_df is None:
                return False
            
            # Extract title-based keywords
            title_keywords = self.extract_keywords_from_title(metadata['title'])
            metadata['keywords_title'] = title_keywords
            
            # Extract column-based keywords
            column_keywords = self.extract_keywords_from_columns(cleaned_df.columns)
            metadata['keywords_columns'] = column_keywords
            
            # Generate new filename based on title and academic year
            original_filename = file_name
            new_filename = self.generate_clean_filename(metadata['title'], metadata['academic_year'], original_filename)
            
            # Store original filename in metadata
            metadata['original_filename'] = original_filename
            # Store new filename in metadata
            metadata['new_filename'] = new_filename
            
            # Create metadata line
            metadata_line = f"#METADATA:{json.dumps(metadata)}\n"
            
            # Save cleaned file with metadata using new filename
            cleaned_path = self.clean_dir / new_filename
            
            # Write metadata line first, then the data
            with open(cleaned_path, 'w', encoding='utf-8') as f:
                f.write(metadata_line)
            
            # Then append the dataframe
            cleaned_df.to_csv(cleaned_path, index=False, mode='a')
            
            logger.info(f"Successfully processed {file_name} with metadata")
            logger.info(f"Title: {metadata.get('title', 'Not found')}")
            logger.info(f"Academic Year: {metadata.get('academic_year', 'Not found')}")
            logger.info(f"New filename: {new_filename}")
            logger.info(f"Title keywords: {title_keywords}")
            logger.info(f"Column keywords: {column_keywords}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {str(e)}")
            return False

    def extract_keywords_from_title(self, title: str) -> List[str]:
        """Extract meaningful keywords from the title."""
        # Convert to lowercase
        title_lower = title.lower()
        
        # Remove non-alphanumeric characters (except hyphens for compound words)
        title_clean = re.sub(r'[^\w\s\-]', ' ', title_lower)
        
        # Extract words (including hyphenated words)
        words = re.findall(r'\b\w+(?:-\w+)*\b', title_clean)
        
        # Define stop words to remove
        stop_words = ["a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "of", "in", "on", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could"]
        
        # Filter out stop words and numbers-only entries
        meaningful_words = [word for word in words if word not in stop_words and not word.isdigit() and len(word) > 2]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = [x for x in meaningful_words if not (x in seen or seen.add(x))]
        
        return unique_keywords

    def extract_keywords_from_columns(self, columns: List[str]) -> List[str]:
        """Extract meaningful keywords from column names."""
        all_keywords = []
        
        for column in columns:
            if not isinstance(column, str):
                continue
            
            # Convert to lowercase
            col_lower = column.lower()
            
            # Remove non-alphanumeric characters (except hyphens)
            col_clean = re.sub(r'[^\w\s\-]', ' ', col_lower)
            
            # Extract words (including hyphenated words)
            words = re.findall(r'\b\w+(?:-\w+)*\b', col_clean)
            
            # Define stop words to remove
            stop_words = ["a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "from", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "of", "in", "on", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might", "must", "can", "could"]
            
            # Filter out stop words and short words
            meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            # Add to keywords list
            all_keywords.extend(meaningful_words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = [x for x in all_keywords if not (x in seen or seen.add(x))]
        
        return unique_keywords 

    def generate_clean_filename(self, title, academic_year, original_filename):
        """Generate a new filename based on title and academic year."""
        # Clean title for filename (remove special characters)
        clean_title = title.strip()
        
        # Clean academic year (replace / with -)
        clean_year = academic_year.replace('/', '-') if academic_year != "Unknown" else ""
        
        # Create new filename: "Title Year OriginalFilename"
        if clean_year:
            new_filename = f"{clean_title} {clean_year} {original_filename}"
        else:
            new_filename = f"{clean_title} {original_filename}"
            
        logger.info(f"Generated new filename: {new_filename}")
        return new_filename 