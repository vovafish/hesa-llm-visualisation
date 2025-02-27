from typing import Dict, List, Optional, Union, BinaryIO
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import csv
import openpyxl
from io import BytesIO
import zipfile
from datetime import datetime

logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self):
        """Initialize file handler with supported formats and conversions."""
        self.supported_formats = {
            'csv': {
                'read': self._read_csv,
                'write': self._write_csv,
                'mime_type': 'text/csv'
            },
            'excel': {
                'read': self._read_excel,
                'write': self._write_excel,
                'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            },
            'json': {
                'read': self._read_json,
                'write': self._write_json,
                'mime_type': 'application/json'
            }
        }

    def read_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Read a file in any supported format."""
        try:
            file_format = self._detect_format(file_path)
            if file_format not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            return self.supported_formats[file_format]['read'](file_path)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    def write_file(self, df: pd.DataFrame, file_path: Path, 
                  file_format: Optional[str] = None) -> bool:
        """Write DataFrame to file in specified format."""
        try:
            if file_format is None:
                file_format = self._detect_format(file_path)
            
            if file_format not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            return self.supported_formats[file_format]['write'](df, file_path)
            
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {str(e)}")
            return False

    def convert_file(self, source_path: Path, target_path: Path) -> bool:
        """Convert file from one format to another."""
        try:
            # Read source file
            df = self.read_file(source_path)
            if df is None:
                return False
            
            # Write to target format
            return self.write_file(df, target_path)
            
        except Exception as e:
            logger.error(f"Error converting file {source_path} to {target_path}: {str(e)}")
            return False

    def create_archive(self, files: List[Path], archive_path: Path) -> bool:
        """Create a ZIP archive containing multiple files."""
        try:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in files:
                    if not file_path.exists():
                        logger.warning(f"File not found: {file_path}")
                        continue
                    
                    zip_file.write(file_path, file_path.name)
            
            logger.info(f"Created archive: {archive_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating archive {archive_path}: {str(e)}")
            return False

    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        extension = file_path.suffix.lower()[1:]  # Remove the dot
        
        if extension in ['xlsx', 'xls']:
            return 'excel'
        elif extension == 'csv':
            return 'csv'
        elif extension == 'json':
            return 'json'
        else:
            raise ValueError(f"Unsupported file extension: {extension}")

    def _read_csv(self, file_path: Path) -> pd.DataFrame:
        """Read CSV file with appropriate settings."""
        try:
            # Try to detect encoding
            encodings = ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Could not read CSV file with any supported encoding")
            
        except Exception as e:
            logger.error(f"Error reading CSV {file_path}: {str(e)}")
            raise

    def _write_csv(self, df: pd.DataFrame, file_path: Path) -> bool:
        """Write DataFrame to CSV with appropriate settings."""
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            logger.error(f"Error writing CSV {file_path}: {str(e)}")
            return False

    def _read_excel(self, file_path: Path) -> pd.DataFrame:
        """Read Excel file with appropriate settings."""
        try:
            return pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            logger.error(f"Error reading Excel {file_path}: {str(e)}")
            raise

    def _write_excel(self, df: pd.DataFrame, file_path: Path) -> bool:
        """Write DataFrame to Excel with appropriate settings."""
        try:
            writer = pd.ExcelWriter(file_path, engine='openpyxl')
            df.to_excel(writer, index=False)
            writer.save()
            return True
        except Exception as e:
            logger.error(f"Error writing Excel {file_path}: {str(e)}")
            return False

    def _read_json(self, file_path: Path) -> pd.DataFrame:
        """Read JSON file with appropriate settings."""
        try:
            return pd.read_json(file_path)
        except Exception as e:
            logger.error(f"Error reading JSON {file_path}: {str(e)}")
            raise

    def _write_json(self, df: pd.DataFrame, file_path: Path) -> bool:
        """Write DataFrame to JSON with appropriate settings."""
        try:
            df.to_json(file_path, orient='records', indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing JSON {file_path}: {str(e)}")
            return False

    def get_file_info(self, file_path: Path) -> Dict:
        """Get detailed information about a file."""
        try:
            stats = file_path.stat()
            return {
                'name': file_path.name,
                'format': self._detect_format(file_path),
                'size': stats.st_size,
                'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
                'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                'mime_type': self.supported_formats[
                    self._detect_format(file_path)]['mime_type']
            }
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {str(e)}")
            return {} 