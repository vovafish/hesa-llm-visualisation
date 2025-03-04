from typing import Optional, Dict, List, Union, Tuple, Any
import pandas as pd
from pathlib import Path
import logging
import os
import sys

# Import our storage classes
from core.data_processing.storage.file_storage import FileStorage
from core.data_processing.storage.data_manager import DataManager

logger = logging.getLogger(__name__)

class StorageService:
    """
    Service that connects the CSV processing pipeline with file storage systems.
    This class serves as the main entry point for interacting with the data storage
    in the application.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the storage service.
        
        Args:
            base_dir: Optional base directory for storage. If not provided,
                      uses the default data directory.
        """
        if base_dir is None:
            # Use default data directory
            # First check if we're in development or production
            if os.environ.get('DJANGO_SETTINGS_MODULE') == 'hesa_llm_visualisation.settings.production':
                base_dir = Path('/data')
            else:
                # For development, use data directory in project
                base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) / 'data'
        
        # Ensure directory exists
        base_dir.mkdir(parents=True, exist_ok=True)
        
        self.base_dir = base_dir
        self.file_storage = FileStorage(base_dir)
        self.data_manager = DataManager(base_dir=base_dir)
        
        logger.info(f"Initialized StorageService with base directory: {base_dir}")
    
    def register_processed_file(self, 
                               source_name: str,
                               version_number: str,
                               dataset_name: str,
                               data_type: str,
                               original_file_path: str,
                               processed_file_path: str,
                               validation_result: Dict,
                               df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Register a processed CSV file in the storage system.
        
        Args:
            source_name: Name of the data source (e.g., "HESA")
            version_number: Version of the data (e.g., "2022.1")
            dataset_name: Name of the dataset (e.g., "Student_Records")
            data_type: Type of data (e.g., 'student', 'finance')
            original_file_path: Path to the original CSV file
            processed_file_path: Path to the processed/cleaned CSV file
            validation_result: Dictionary containing validation results
            df: Optional DataFrame if already loaded
            
        Returns:
            Dict containing dataset information
        """
        # Register data source
        source = self.file_storage.register_data_source(
            name=source_name,
            description=f"Data source for {source_name} datasets",
            url=""
        )
        
        # Register data version
        version = self.file_storage.register_data_version(
            source_name=source_name,
            version_number=version_number
        )
        
        # Load DataFrame if not provided
        if df is None and processed_file_path and os.path.exists(processed_file_path):
            df = pd.read_csv(processed_file_path)
        
        # Register processed dataset
        dataset = self.file_storage.register_processed_dataset(
            name=dataset_name,
            data_type=data_type,
            source_name=source_name,
            version_number=version_number,
            file_path=processed_file_path,
            df=df
        )
        
        # Register validation result
        validation = self.file_storage.register_validation_result(
            file_name=os.path.basename(original_file_path),
            original_path=original_file_path,
            processed_path=processed_file_path,
            status=validation_result.get("status", "unknown"),
            error_count=validation_result.get("error_count", 0),
            warning_count=validation_result.get("warning_count", 0),
            fixed_issues_count=validation_result.get("fixed_issues_count", 0),
            validation_report=validation_result
        )
        
        logger.info(f"Registered processed file: {dataset_name} from {source_name} v{version_number}")
        
        # Add file information to DataManager (without the metadata that's not supported)
        if processed_file_path and os.path.exists(processed_file_path):
            self.data_manager.add_file(
                file_path=Path(processed_file_path), 
                file_type="cleaned"
            )
        
        # Return all the information
        return {
            "source": source,
            "version": version,
            "dataset": dataset,
            "validation": validation
        }
    
    def get_latest_datasets(self, data_type: Optional[str] = None) -> List[Dict]:
        """
        Get the latest version of each dataset.
        
        Args:
            data_type: Optional filter by data type
            
        Returns:
            List of dataset information dictionaries
        """
        return self.file_storage.get_latest_datasets(data_type=data_type)
    
    def get_dataset_data(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Get the data for a dataset as a pandas DataFrame.
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            DataFrame or None if not found
        """
        data = self.file_storage.get_dataset_data(dataset_id=dataset_id)
        if data:
            return pd.DataFrame(data)
        return None
    
    def get_validation_summary(self) -> Dict:
        """
        Get a summary of all validation results.
        
        Returns:
            Dictionary with validation summary
        """
        validation_data = self.file_storage._load_json(self.file_storage.validation_file)
        validation_results = validation_data.get("validation_results", [])
        
        if not validation_results:
            return {
                "total_files": 0,
                "valid_files": 0,
                "files_with_warnings": 0,
                "invalid_files": 0,
                "total_errors": 0,
                "total_warnings": 0,
                "total_fixed_issues": 0
            }
        
        # Calculate summary stats
        valid_files = sum(1 for r in validation_results if r.get("status") == "valid")
        files_with_warnings = sum(1 for r in validation_results if r.get("status") == "warnings")
        invalid_files = sum(1 for r in validation_results if r.get("status") == "invalid")
        total_errors = sum(r.get("error_count", 0) for r in validation_results)
        total_warnings = sum(r.get("warning_count", 0) for r in validation_results)
        total_fixed_issues = sum(r.get("fixed_issues_count", 0) for r in validation_results)
        
        return {
            "total_files": len(validation_results),
            "valid_files": valid_files,
            "files_with_warnings": files_with_warnings,
            "invalid_files": invalid_files,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_fixed_issues": total_fixed_issues
        }
    
    def clear_all_cache(self) -> bool:
        """Clear all cached data."""
        return self.file_storage.clear_all_cache()
    
    def backup_all_data(self, backup_dir: Optional[Path] = None) -> bool:
        """
        Create a backup of all data.
        
        Args:
            backup_dir: Optional directory for backup. If not provided,
                      creates a timestamped backup in the base directory.
        
        Returns:
            True if successful
        """
        # Use DataManager's built-in backup functionality
        try:
            return self.data_manager.create_backup()
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return False
            
    # Materialized Views methods
    
    def create_common_views(self) -> Dict[str, Dict]:
        """
        Create or update a set of commonly used materialized views.
        
        Returns:
            Dictionary mapping view names to view information
        """
        logger.info("Creating common materialized views...")
        results = {}
        
        # Get latest datasets
        datasets = self.get_latest_datasets()
        
        # Create view for datasets by type
        data_types = {}
        for dataset in datasets:
            data_type = dataset.get('data_type')
            if data_type:
                if data_type not in data_types:
                    data_types[data_type] = []
                data_types[data_type].append(dataset)
        
        # Create a summary view of all datasets
        dataset_summary = [{
            'id': dataset.get('id'),
            'name': dataset.get('name'),
            'data_type': dataset.get('data_type'),
            'source': dataset.get('source_name'),
            'version': dataset.get('version_number'),
            'row_count': dataset.get('row_count'),
            'processed_at': dataset.get('processed_at')
        } for dataset in datasets]
        
        results['dataset_summary'] = self.file_storage.create_materialized_view(
            view_name="Dataset Summary",
            description="Summary of all latest datasets",
            data=dataset_summary,
            dependencies=[d.get('id') for d in datasets],
            refresh_frequency="daily"
        )
        
        # Get validation results
        validation_data = self.file_storage._load_json(self.file_storage.validation_file)
        validation_results = validation_data.get("validation_results", [])
        
        # Create validation summary view
        if validation_results:
            validation_summary = [{
                'file_name': v.get('file_name'),
                'status': v.get('status'),
                'error_count': v.get('error_count', 0),
                'warning_count': v.get('warning_count', 0),
                'fixed_issues_count': v.get('fixed_issues_count', 0),
                'validation_date': v.get('validation_date')
            } for v in validation_results]
            
            results['validation_summary'] = self.file_storage.create_materialized_view(
                view_name="Validation Summary",
                description="Summary of validation results",
                data=validation_summary,
                refresh_frequency="daily"
            )
        
        # For each data type, create aggregate statistics (if numeric data is available)
        for data_type, type_datasets in data_types.items():
            if not type_datasets:
                continue
                
            # Try to get the first dataset to analyze its structure
            if type_datasets[0].get('id'):
                sample_data = self.get_dataset_data(type_datasets[0].get('id'))
                
                if sample_data is not None:
                    # Check for numeric columns we could aggregate
                    numeric_cols = sample_data.select_dtypes(include=['number']).columns
                    
                    if len(numeric_cols) > 0:
                        # Create aggregate stats
                        agg_stats = []
                        
                        for col in numeric_cols:
                            agg_stats.append({
                                'data_type': data_type,
                                'column': col,
                                'min': float(sample_data[col].min()),
                                'max': float(sample_data[col].max()),
                                'mean': float(sample_data[col].mean()),
                                'median': float(sample_data[col].median()),
                                'std_dev': float(sample_data[col].std())
                            })
                        
                        results[f'{data_type}_stats'] = self.file_storage.create_materialized_view(
                            view_name=f"{data_type.title()} Statistics",
                            description=f"Statistical summary of {data_type} data",
                            data=agg_stats,
                            dependencies=[d.get('id') for d in type_datasets],
                            refresh_frequency="daily"
                        )
        
        logger.info(f"Created {len(results)} common materialized views")
        return results
        
    def create_materialized_view(self, view_name: str, description: str, 
                               data: Union[pd.DataFrame, List[Dict]], 
                               dependencies: List[str] = None,
                               metadata: Dict = None,
                               refresh_frequency: str = "manual") -> Dict:
        """
        Create a materialized view for a specific query or analysis.
        
        Args:
            view_name: Name for the view
            description: Description of the view
            data: Data to store (DataFrame or List of Dicts)
            dependencies: List of dataset IDs this view depends on
            metadata: Additional metadata
            refresh_frequency: How often to refresh ("manual", "daily", "hourly")
            
        Returns:
            View information dictionary
        """
        return self.file_storage.create_materialized_view(
            view_name=view_name,
            description=description,
            data=data,
            dependencies=dependencies,
            metadata=metadata,
            refresh_frequency=refresh_frequency
        )
        
    def get_materialized_view(self, view_name: str) -> Optional[pd.DataFrame]:
        """
        Get a materialized view as a DataFrame.
        
        Args:
            view_name: Name of the view
            
        Returns:
            DataFrame containing the view data or None if not found
        """
        data = self.file_storage.get_materialized_view(view_name)
        if data is not None:
            return pd.DataFrame(data)
        return None
        
    def list_materialized_views(self) -> List[Dict]:
        """
        List all available materialized views.
        
        Returns:
            List of view information dictionaries
        """
        return self.file_storage.list_materialized_views()
        
    def refresh_view_if_stale(self, view_name: str) -> bool:
        """
        Check if a view is stale and refresh it if needed.
        
        Args:
            view_name: Name of the view
            
        Returns:
            True if view was refreshed
        """
        if not self.file_storage.is_view_stale(view_name):
            return False
            
        # View is stale, we need to refresh it
        logger.info(f"View {view_name} is stale, refreshing...")
        
        # For now, just recreate common views
        # In a real app, you'd have more sophisticated logic here
        self.create_common_views()
        
        return True 