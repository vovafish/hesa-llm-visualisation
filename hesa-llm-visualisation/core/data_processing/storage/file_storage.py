from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import logging
import os
import hashlib

logger = logging.getLogger(__name__)

class FileStorage:
    def __init__(self, base_dir: Path):
        """
        Initialize file storage system with directory structure.
        
        Args:
            base_dir: Base directory for all data storage
        """
        self.base_dir = Path(base_dir)
        
        # Main data directories
        self.raw_dir = self.base_dir / 'raw_files'
        self.cleaned_dir = self.base_dir / 'cleaned_files'
        self.metadata_dir = self.base_dir / 'metadata'
        self.cache_dir = self.base_dir / 'cache'
        self.versions_dir = self.base_dir / 'versions'
        self.materialized_views_dir = self.base_dir / 'materialized_views'
        
        # Create necessary directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.materialized_views_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata index
        self.sources_file = self.metadata_dir / 'sources.json'
        self.datasets_file = self.metadata_dir / 'datasets.json'
        self.validation_file = self.metadata_dir / 'validation_results.json'
        self.materialized_views_index = self.metadata_dir / 'materialized_views.json'
        
        self._init_metadata()

    def _init_metadata(self) -> None:
        """Initialize metadata files if they don't exist."""
        if not self.sources_file.exists():
            with open(self.sources_file, 'w') as f:
                json.dump({
                    "sources": {},
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        
        if not self.datasets_file.exists():
            with open(self.datasets_file, 'w') as f:
                json.dump({
                    "datasets": {},
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        
        if not self.validation_file.exists():
            with open(self.validation_file, 'w') as f:
                json.dump({
                    "validation_results": [],
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
                
        if not self.materialized_views_index.exists():
            with open(self.materialized_views_index, 'w') as f:
                json.dump({
                    "views": {},
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)

    def _load_json(self, file_path: Path) -> Dict:
        """Load JSON data from a file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.error(f"Error loading JSON from {file_path}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict) -> bool:
        """Save JSON data to a file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON to {file_path}: {str(e)}")
            return False
    
    def register_data_source(self, name: str, description: str = "", url: str = "") -> Dict:
        """
        Register a new data source or get existing one.
        
        Args:
            name: Name of the data source (e.g., "HESA Student Data")
            description: Description of the data source
            url: URL to the source website
            
        Returns:
            Dict containing source information
        """
        sources_data = self._load_json(self.sources_file)
        sources = sources_data.get("sources", {})
        
        # Convert name to a safe key
        source_key = name.lower().replace(' ', '_')
        
        if source_key not in sources:
            # Create new source
            sources[source_key] = {
                "name": name,
                "description": description,
                "url": url,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "versions": []
            }
            logger.info(f"Created new data source: {name}")
        else:
            # Update existing source
            sources[source_key]["updated_at"] = datetime.now().isoformat()
            if description:
                sources[source_key]["description"] = description
            if url:
                sources[source_key]["url"] = url
            logger.info(f"Updated existing data source: {name}")
        
        # Save updated sources data
        sources_data["sources"] = sources
        sources_data["last_updated"] = datetime.now().isoformat()
        self._save_json(self.sources_file, sources_data)
        
        return sources[source_key]
    
    def register_data_version(self, source_name: str, version_number: str, 
                            release_date: str = None) -> Dict:
        """
        Register a new version of data from a source.
        
        Args:
            source_name: Name of the data source
            version_number: Version identifier (e.g., "2022.1")
            release_date: Release date of this version (ISO format)
            
        Returns:
            Dict containing version information
        """
        if release_date is None:
            release_date = datetime.now().date().isoformat()
        
        sources_data = self._load_json(self.sources_file)
        sources = sources_data.get("sources", {})
        
        # Convert name to a safe key
        source_key = source_name.lower().replace(' ', '_')
        
        if source_key not in sources:
            logger.error(f"Source not found: {source_name}")
            return {}
        
        # Check if version exists
        version_exists = False
        version_info = None
        for version in sources[source_key].get("versions", []):
            if version["version_number"] == version_number:
                version_exists = True
                version_info = version
                break
        
        if not version_exists:
            # Create new version
            version_info = {
                "version_number": version_number,
                "release_date": release_date,
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "notes": ""
            }
            
            # Deactivate previous versions
            for version in sources[source_key].get("versions", []):
                version["is_active"] = False
            
            # Add new version
            sources[source_key].setdefault("versions", []).append(version_info)
            logger.info(f"Created new data version: {source_name} v{version_number}")
        else:
            logger.info(f"Using existing data version: {source_name} v{version_number}")
        
        # Save updated sources data
        sources_data["sources"] = sources
        sources_data["last_updated"] = datetime.now().isoformat()
        self._save_json(self.sources_file, sources_data)
        
        return version_info
    
    def register_processed_dataset(self, name: str, data_type: str,
                                 source_name: str, version_number: str,
                                 file_path: str, df: Optional[pd.DataFrame] = None,
                                 metadata: Optional[Dict] = None) -> Dict:
        """
        Register a processed dataset in the storage system.
        
        Args:
            name: Name of the dataset
            data_type: Type of data (e.g., 'student', 'finance')
            source_name: Name of the data source
            version_number: Version identifier 
            file_path: Path to the cleaned CSV file
            df: Optional DataFrame containing the data
            metadata: Optional metadata about the dataset
            
        Returns:
            Dict containing dataset information
        """
        datasets_data = self._load_json(self.datasets_file)
        datasets = datasets_data.get("datasets", {})
        
        # Create a unique dataset ID
        dataset_id = f"{source_name.lower().replace(' ', '_')}_{name.lower().replace(' ', '_')}_{version_number}"
        
        # Load data if not provided
        if df is None and os.path.exists(file_path):
            df = pd.read_csv(file_path)
        
        # Generate metadata if not provided
        if df is not None:
            row_count = len(df)
            
            if metadata is None:
                # Generate metadata from DataFrame
                metadata = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(df[col].dtype) for col in df.columns},
                    "shape": df.shape,
                    "summary": {
                        col: {
                            "min": float(df[col].min()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                            "max": float(df[col].max()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                            "mean": float(df[col].mean()) if pd.api.types.is_numeric_dtype(df[col]) else None,
                            "null_count": int(df[col].isna().sum())
                        } for col in df.columns
                    }
                }
        else:
            row_count = 0
            if metadata is None:
                metadata = {}
        
        # Create or update dataset information
        if dataset_id in datasets:
            # Update existing dataset
            datasets[dataset_id].update({
                "updated_at": datetime.now().isoformat(),
                "row_count": row_count,
                "metadata": metadata
            })
            logger.info(f"Updated existing dataset: {name}")
        else:
            # Create new dataset
            datasets[dataset_id] = {
                "id": dataset_id,
                "name": name,
                "data_type": data_type,
                "source_name": source_name,
                "version_number": version_number,
                "file_path": file_path,
                "row_count": row_count,
                "processed_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "is_valid": True,
                "metadata": metadata
            }
            logger.info(f"Registered new dataset: {name} ({data_type})")
        
        # Save dataset to cache if available
        if df is not None:
            cache_file = self.cache_dir / f"{dataset_id}.json"
            self._save_json(cache_file, df.to_dict(orient='records'))
        
        # Save updated datasets data
        datasets_data["datasets"] = datasets
        datasets_data["last_updated"] = datetime.now().isoformat()
        self._save_json(self.datasets_file, datasets_data)
        
        return datasets[dataset_id]
    
    def register_validation_result(self, file_name: str, original_path: str,
                                 processed_path: str, status: str,
                                 error_count: int, warning_count: int,
                                 fixed_issues_count: int,
                                 validation_report: Dict) -> Dict:
        """
        Register a validation result in the storage system.
        
        Args:
            file_name: Name of the validated file
            original_path: Path to the original file
            processed_path: Path to the processed/cleaned file (if successful)
            status: Validation status ('valid', 'invalid', 'warnings')
            error_count: Number of critical errors
            warning_count: Number of warnings
            fixed_issues_count: Number of issues that were fixed
            validation_report: Full validation report
            
        Returns:
            Dict containing validation result information
        """
        validation_data = self._load_json(self.validation_file)
        validation_results = validation_data.get("validation_results", [])
        
        # Create validation result
        result = {
            "id": len(validation_results) + 1,
            "file_name": file_name,
            "original_path": str(original_path),
            "processed_path": str(processed_path) if processed_path else "",
            "status": status,
            "validation_date": datetime.now().isoformat(),
            "error_count": error_count,
            "warning_count": warning_count,
            "fixed_issues_count": fixed_issues_count,
            "validation_report": validation_report
        }
        
        # Add to validation results
        validation_results.append(result)
        
        # Save updated validation data
        validation_data["validation_results"] = validation_results
        validation_data["last_updated"] = datetime.now().isoformat()
        self._save_json(self.validation_file, validation_data)
        
        logger.info(f"Registered validation result for {file_name}: {status}")
        return result
    
    def get_latest_datasets(self, data_type: Optional[str] = None) -> List[Dict]:
        """
        Get the latest version of each dataset.
        
        Args:
            data_type: Optional filter by data type
            
        Returns:
            List of dataset information dictionaries
        """
        datasets_data = self._load_json(self.datasets_file)
        datasets = datasets_data.get("datasets", {}).values()
        
        # Get source information for active versions
        sources_data = self._load_json(self.sources_file)
        sources = sources_data.get("sources", {})
        
        active_versions = {}
        for source_key, source_info in sources.items():
            for version in source_info.get("versions", []):
                if version.get("is_active", False):
                    active_versions[(source_info["name"], version["version_number"])] = True
        
        # Filter for latest active datasets
        latest_datasets = []
        for dataset in datasets:
            if (dataset["source_name"], dataset["version_number"]) in active_versions:
                if data_type is None or dataset["data_type"] == data_type:
                    latest_datasets.append(dataset)
        
        return sorted(latest_datasets, key=lambda x: x["name"])
    
    def get_dataset_data(self, dataset_id: str) -> Optional[List[Dict]]:
        """
        Get the data for a dataset, using cache if available.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            List of dictionaries representing the data, or None if not found
        """
        # Try to get from cache first
        cache_file = self.cache_dir / f"{dataset_id}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cached data: {str(e)}")
        
        # If not in cache, load from file
        datasets_data = self._load_json(self.datasets_file)
        datasets = datasets_data.get("datasets", {})
        
        if dataset_id not in datasets:
            logger.error(f"Dataset with ID {dataset_id} not found")
            return None
        
        dataset = datasets[dataset_id]
        file_path = dataset["file_path"]
        
        try:
            df = pd.read_csv(file_path)
            data = df.to_dict(orient='records')
            
            # Cache the data for future use
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return data
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            return None
    
    def clear_dataset_cache(self, dataset_id: str) -> bool:
        """
        Clear the cache for a dataset.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            bool: True if successful
        """
        cache_file = self.cache_dir / f"{dataset_id}.json"
        if cache_file.exists():
            try:
                cache_file.unlink()
                logger.info(f"Cleared cache for dataset {dataset_id}")
                return True
            except Exception as e:
                logger.error(f"Error clearing cache: {str(e)}")
                return False
        return True
        
    def clear_all_cache(self) -> bool:
        """Clear all cached data."""
        try:
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
            logger.info("Cleared all cached data")
            return True
        except Exception as e:
            logger.error(f"Error clearing all cache: {str(e)}")
            return False

    def create_materialized_view(self, 
                               view_name: str, 
                               description: str,
                               data: Union[List[Dict], pd.DataFrame], 
                               dependencies: List[str] = None,
                               metadata: Dict = None,
                               refresh_frequency: str = "manual") -> Dict:
        """
        Create or update a materialized view for storing pre-computed query results.
        
        Args:
            view_name: Name of the materialized view
            description: Description of what this view contains
            data: The pre-computed data (DataFrame or list of dicts)
            dependencies: List of dataset IDs this view depends on
            metadata: Additional metadata about this view
            refresh_frequency: How often the view should be refreshed ("manual", "daily", "hourly")
            
        Returns:
            Dict containing materialized view information
        """
        # Convert DataFrame to dict if needed
        if isinstance(data, pd.DataFrame):
            data_dict = data.to_dict(orient='records')
        else:
            data_dict = data
            
        # Create safe filename
        safe_name = view_name.lower().replace(' ', '_').replace('-', '_')
        file_path = self.materialized_views_dir / f"{safe_name}.json"
        
        # Save data
        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
            
        # Get index of views
        views_data = self._load_json(self.materialized_views_index)
        views = views_data.get("views", {})
        
        # Update or create view metadata
        now = datetime.now().isoformat()
        
        if dependencies is None:
            dependencies = []
            
        if metadata is None:
            metadata = {}
            
        # Create or update view entry
        views[safe_name] = {
            "name": view_name,
            "description": description,
            "file_path": str(file_path),
            "created_at": views.get(safe_name, {}).get("created_at", now),
            "updated_at": now,
            "row_count": len(data_dict),
            "dependencies": dependencies,
            "refresh_frequency": refresh_frequency,
            "last_refresh": now,
            "metadata": metadata
        }
        
        # Save updated views index
        views_data["views"] = views
        views_data["last_updated"] = now
        self._save_json(self.materialized_views_index, views_data)
        
        logger.info(f"Created/updated materialized view: {view_name} with {len(data_dict)} rows")
        return views[safe_name]
    
    def get_materialized_view(self, view_name: str) -> Optional[List[Dict]]:
        """
        Get data from a materialized view.
        
        Args:
            view_name: Name of the materialized view
            
        Returns:
            List of dictionaries containing the view data, or None if not found
        """
        # Check if view exists in the index
        views_data = self._load_json(self.materialized_views_index)
        views = views_data.get("views", {})
        
        # Convert to safe name
        safe_name = view_name.lower().replace(' ', '_').replace('-', '_')
        
        if safe_name not in views:
            logger.error(f"Materialized view not found: {view_name}")
            return None
            
        view_info = views[safe_name]
        file_path = Path(view_info["file_path"])
        
        if not file_path.exists():
            logger.error(f"Materialized view file not found: {file_path}")
            return None
            
        # Load data from file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Update access metadata
            view_info["last_accessed"] = datetime.now().isoformat()
            view_info["access_count"] = view_info.get("access_count", 0) + 1
            views[safe_name] = view_info
            views_data["views"] = views
            self._save_json(self.materialized_views_index, views_data)
            
            return data
        except Exception as e:
            logger.error(f"Error loading materialized view {view_name}: {str(e)}")
            return None
    
    def get_materialized_view_as_df(self, view_name: str) -> Optional[pd.DataFrame]:
        """
        Get data from a materialized view as a pandas DataFrame.
        
        Args:
            view_name: Name of the materialized view
            
        Returns:
            DataFrame containing the view data, or None if not found
        """
        data = self.get_materialized_view(view_name)
        if data is not None:
            return pd.DataFrame(data)
        return None
    
    def list_materialized_views(self) -> List[Dict]:
        """
        List all available materialized views.
        
        Returns:
            List of view information dictionaries
        """
        views_data = self._load_json(self.materialized_views_index)
        views = views_data.get("views", {})
        return list(views.values())
    
    def is_view_stale(self, view_name: str) -> bool:
        """
        Check if a materialized view needs refreshing.
        
        Args:
            view_name: Name of the materialized view
            
        Returns:
            True if the view is stale and needs refreshing
        """
        # Convert to safe name
        safe_name = view_name.lower().replace(' ', '_').replace('-', '_')
        
        # Get view information
        views_data = self._load_json(self.materialized_views_index)
        views = views_data.get("views", {})
        
        if safe_name not in views:
            logger.error(f"Materialized view not found: {view_name}")
            return True
            
        view_info = views[safe_name]
        
        # Check based on refresh frequency
        refresh_frequency = view_info.get("refresh_frequency", "manual")
        
        if refresh_frequency == "manual":
            return False
            
        last_refresh = datetime.fromisoformat(view_info.get("last_refresh", "2000-01-01T00:00:00"))
        now = datetime.now()
        
        if refresh_frequency == "hourly":
            return (now - last_refresh).total_seconds() > 3600
        elif refresh_frequency == "daily":
            return (now - last_refresh).total_seconds() > 86400
        elif refresh_frequency == "weekly":
            return (now - last_refresh).total_seconds() > 604800
            
        return False
    
    def delete_materialized_view(self, view_name: str) -> bool:
        """
        Delete a materialized view.
        
        Args:
            view_name: Name of the materialized view
            
        Returns:
            True if successful
        """
        # Convert to safe name
        safe_name = view_name.lower().replace(' ', '_').replace('-', '_')
        
        # Get view information
        views_data = self._load_json(self.materialized_views_index)
        views = views_data.get("views", {})
        
        if safe_name not in views:
            logger.error(f"Materialized view not found: {view_name}")
            return False
            
        view_info = views[safe_name]
        file_path = Path(view_info["file_path"])
        
        # Delete file
        try:
            if file_path.exists():
                file_path.unlink()
                
            # Remove from index
            del views[safe_name]
            views_data["views"] = views
            views_data["last_updated"] = datetime.now().isoformat()
            self._save_json(self.materialized_views_index, views_data)
            
            logger.info(f"Deleted materialized view: {view_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting materialized view {view_name}: {str(e)}")
            return False 