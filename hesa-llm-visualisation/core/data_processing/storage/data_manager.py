from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import logging
import json
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, base_dir: Path):
        """Initialize data manager with directory structure."""
        self.base_dir = base_dir
        self.raw_dir = base_dir / 'raw_files'
        self.cleaned_dir = base_dir / 'cleaned_files'
        self.backup_dir = base_dir / 'backups'
        self.metadata_file = base_dir / 'metadata.json'
        
        # Create necessary directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load metadata
        self._init_metadata()

    def _init_metadata(self) -> None:
        """Initialize or load metadata file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'files': {},
                'last_update': datetime.now().isoformat(),
                'version': '1.0'
            }
            self._save_metadata()

    def _save_metadata(self) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_file(self, file_path: Path, file_type: str = 'raw') -> bool:
        """Add a new file to the data storage."""
        try:
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Determine target directory
            target_dir = self.raw_dir if file_type == 'raw' else self.cleaned_dir
            target_path = target_dir / file_path.name
            
            # Copy file to appropriate directory
            shutil.copy2(file_path, target_path)
            
            # Update metadata
            self.metadata['files'][file_path.name] = {
                'type': file_type,
                'added_date': datetime.now().isoformat(),
                'original_path': str(file_path),
                'size': file_path.stat().st_size,
                'last_modified': datetime.fromtimestamp(
                    file_path.stat().st_mtime).isoformat()
            }
            
            self._save_metadata()
            logger.info(f"Successfully added file: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding file {file_path}: {str(e)}")
            return False

    def get_file(self, file_name: str, file_type: str = 'cleaned') -> Optional[pd.DataFrame]:
        """Retrieve a file as a pandas DataFrame."""
        try:
            target_dir = self.raw_dir if file_type == 'raw' else self.cleaned_dir
            file_path = target_dir / file_name
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            
            return pd.read_csv(file_path)
            
        except Exception as e:
            logger.error(f"Error reading file {file_name}: {str(e)}")
            return None

    def list_files(self, file_type: Optional[str] = None) -> List[Dict]:
        """List all files with their metadata."""
        files = []
        for file_name, metadata in self.metadata['files'].items():
            if file_type is None or metadata['type'] == file_type:
                files.append({
                    'name': file_name,
                    **metadata
                })
        return files

    def create_backup(self, file_name: str) -> bool:
        """Create a backup of a file."""
        try:
            if file_name not in self.metadata['files']:
                logger.error(f"File not found in metadata: {file_name}")
                return False
            
            file_type = self.metadata['files'][file_name]['type']
            source_dir = self.raw_dir if file_type == 'raw' else self.cleaned_dir
            source_path = source_dir / file_name
            
            if not source_path.exists():
                logger.error(f"Source file not found: {source_path}")
                return False
            
            # Create backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{file_name}.{timestamp}.backup"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(source_path, backup_path)
            
            # Update metadata
            if 'backups' not in self.metadata['files'][file_name]:
                self.metadata['files'][file_name]['backups'] = []
            
            self.metadata['files'][file_name]['backups'].append({
                'backup_name': backup_name,
                'timestamp': timestamp,
                'size': backup_path.stat().st_size
            })
            
            self._save_metadata()
            logger.info(f"Created backup: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup for {file_name}: {str(e)}")
            return False

    def restore_backup(self, file_name: str, backup_timestamp: str) -> bool:
        """Restore a file from backup."""
        try:
            if file_name not in self.metadata['files']:
                logger.error(f"File not found in metadata: {file_name}")
                return False
            
            file_metadata = self.metadata['files'][file_name]
            backup_name = f"{file_name}.{backup_timestamp}.backup"
            backup_path = self.backup_dir / backup_name
            
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_path}")
                return False
            
            # Determine target location
            target_dir = self.raw_dir if file_metadata['type'] == 'raw' else self.cleaned_dir
            target_path = target_dir / file_name
            
            # Create backup of current file before restoring
            self.create_backup(file_name)
            
            # Restore from backup
            shutil.copy2(backup_path, target_path)
            
            logger.info(f"Restored {file_name} from backup {backup_timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup for {file_name}: {str(e)}")
            return False

    def delete_file(self, file_name: str, delete_backups: bool = False) -> bool:
        """Delete a file and optionally its backups."""
        try:
            if file_name not in self.metadata['files']:
                logger.error(f"File not found in metadata: {file_name}")
                return False
            
            file_metadata = self.metadata['files'][file_name]
            file_type = file_metadata['type']
            
            # Delete main file
            target_dir = self.raw_dir if file_type == 'raw' else self.cleaned_dir
            target_path = target_dir / file_name
            
            if target_path.exists():
                target_path.unlink()
            
            # Delete backups if requested
            if delete_backups and 'backups' in file_metadata:
                for backup in file_metadata['backups']:
                    backup_path = self.backup_dir / backup['backup_name']
                    if backup_path.exists():
                        backup_path.unlink()
            
            # Update metadata
            del self.metadata['files'][file_name]
            self._save_metadata()
            
            logger.info(f"Deleted file: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file {file_name}: {str(e)}")
            return False

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        try:
            stats = {
                'total_files': len(self.metadata['files']),
                'raw_files': len([f for f in self.metadata['files'].values() 
                                if f['type'] == 'raw']),
                'cleaned_files': len([f for f in self.metadata['files'].values() 
                                    if f['type'] == 'cleaned']),
                'total_size': sum(f['size'] for f in self.metadata['files'].values()),
                'backup_count': sum(len(f.get('backups', [])) 
                                  for f in self.metadata['files'].values()),
                'last_update': self.metadata['last_update']
            }
            return stats
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {} 