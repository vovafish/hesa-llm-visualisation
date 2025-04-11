# import os
# import pandas as pd
# from django.core.management.base import BaseCommand
# from pathlib import Path
# import logging
# from core.data_processing.storage.storage_service import StorageService

# logger = logging.getLogger(__name__)

# class Command(BaseCommand):
#     help = 'Initialize and test the file-based storage system'

#     def add_arguments(self, parser):
#         parser.add_argument(
#             '--demo',
#             action='store_true',
#             help='Create demo data to test storage',
#         )
#         parser.add_argument(
#             '--views',
#             action='store_true',
#             help='Create materialized views from demo data',
#         )

#     def handle(self, *args, **options):
#         self.stdout.write(self.style.SUCCESS('Initializing storage system...'))
        
#         # Initialize the storage service
#         storage_service = StorageService()
        
#         self.stdout.write(f"Storage base directory: {storage_service.base_dir}")
        
#         # Create demo data if requested
#         if options['demo']:
#             self.create_demo_data(storage_service)
        
#         # Create materialized views if requested
#         if options['views']:
#             self.create_materialized_views(storage_service)
        
#         # Show storage summary
#         self.show_storage_summary(storage_service)
        
#         self.stdout.write(self.style.SUCCESS('Storage system initialized successfully!'))
    
#     def create_demo_data(self, storage_service):
#         """Create demo data to test storage system."""
#         self.stdout.write(self.style.WARNING('Creating demo data...'))
        
#         # Create demo directory if it doesn't exist
#         demo_dir = storage_service.base_dir / 'demo'
#         demo_dir.mkdir(exist_ok=True)
        
#         # Create a demo CSV file
#         self.stdout.write('Creating demo student data...')
#         student_data = {
#             'student_id': list(range(1, 101)),
#             'name': [f'Student {i}' for i in range(1, 101)],
#             'age': [18 + i % 10 for i in range(1, 101)],
#             'course': [f'Course {i % 5 + 1}' for i in range(1, 101)],
#             'grade': [(70 + i % 30) for i in range(1, 101)]
#         }
#         student_df = pd.DataFrame(student_data)
        
#         # Save to CSV
#         student_csv_path = demo_dir / 'student_data.csv'
#         student_df.to_csv(student_csv_path, index=False)
        
#         # Create another demo dataset
#         self.stdout.write('Creating demo finance data...')
#         finance_data = {
#             'institution_id': list(range(1, 21)),
#             'institution_name': [f'University {i}' for i in range(1, 21)],
#             'total_income': [10000000 + i * 500000 for i in range(1, 21)],
#             'research_income': [2000000 + i * 100000 for i in range(1, 21)],
#             'student_fees': [5000000 + i * 200000 for i in range(1, 21)]
#         }
#         finance_df = pd.DataFrame(finance_data)
        
#         # Save to CSV
#         finance_csv_path = demo_dir / 'finance_data.csv'
#         finance_df.to_csv(finance_csv_path, index=False)
        
#         # Register the datasets
#         # Student data
#         self.stdout.write('Registering student data...')
#         student_validation = {
#             "status": "valid",
#             "error_count": 0,
#             "warning_count": 2,
#             "fixed_issues_count": 1,
#             "report": {
#                 "warnings": ["Some student ages might be incorrect", 
#                             "Some grade values are unusually high"],
#                 "fixed_issues": ["Fixed missing values in student_id"]
#             }
#         }
        
#         storage_service.register_processed_file(
#             source_name="HESA",
#             version_number="2023.1",
#             dataset_name="Student Records",
#             data_type="student",
#             original_file_path=str(student_csv_path),
#             processed_file_path=str(student_csv_path),
#             validation_result=student_validation,
#             df=student_df
#         )
        
#         # Finance data
#         self.stdout.write('Registering finance data...')
#         finance_validation = {
#             "status": "valid",
#             "error_count": 0,
#             "warning_count": 0,
#             "fixed_issues_count": 0,
#             "report": {}
#         }
        
#         storage_service.register_processed_file(
#             source_name="HESA",
#             version_number="2023.1",
#             dataset_name="Financial Records",
#             data_type="finance",
#             original_file_path=str(finance_csv_path),
#             processed_file_path=str(finance_csv_path),
#             validation_result=finance_validation,
#             df=finance_df
#         )
        
#         self.stdout.write(self.style.SUCCESS('Demo data created and registered successfully!'))
    
#     def create_materialized_views(self, storage_service):
#         """Create materialized views from the stored data."""
#         self.stdout.write(self.style.WARNING('Creating materialized views...'))
        
#         # Create common views (dataset summary, validation summary, etc.)
#         results = storage_service.create_common_views()
        
#         self.stdout.write(f"Created {len(results)} common materialized views:")
#         for name, view_info in results.items():
#             self.stdout.write(f"- {view_info['name']}: {view_info['row_count']} rows")
        
#         # Create a custom materialized view
#         datasets = storage_service.get_latest_datasets()
#         if datasets:
#             # Filter for student datasets
#             student_datasets = [d for d in datasets if d.get('data_type') == 'student']
#             if student_datasets:
#                 student_dataset = student_datasets[0]
#                 student_df = storage_service.get_dataset_data(student_dataset['id'])
                
#                 if student_df is not None:
#                     # Create a custom view with grade distribution
#                     grade_distribution = student_df.groupby('course')['grade'].agg([
#                         ('min', 'min'),
#                         ('max', 'max'), 
#                         ('mean', 'mean'),
#                         ('count', 'count')
#                     ]).reset_index()
                    
#                     # Add a grade band column
#                     def get_grade_band(row):
#                         mean = row['mean']
#                         if mean >= 90:
#                             return "Excellent"
#                         elif mean >= 80:
#                             return "Very Good"
#                         elif mean >= 70:
#                             return "Good"
#                         elif mean >= 60:
#                             return "Satisfactory"
#                         else:
#                             return "Needs Improvement"
                    
#                     grade_distribution['grade_band'] = grade_distribution.apply(get_grade_band, axis=1)
                    
#                     # Save as a materialized view
#                     view_info = storage_service.create_materialized_view(
#                         view_name="Course Grade Analysis",
#                         description="Analysis of grades by course",
#                         data=grade_distribution,
#                         dependencies=[student_dataset['id']],
#                         refresh_frequency="daily"
#                     )
                    
#                     self.stdout.write(f"Created custom view: {view_info['name']} with {view_info['row_count']} rows")
        
#         self.stdout.write(self.style.SUCCESS('Materialized views created successfully!'))
    
#     def show_storage_summary(self, storage_service):
#         """Show summary of storage contents."""
#         self.stdout.write(self.style.WARNING('Storage Summary:'))
        
#         # Get all datasets
#         datasets = storage_service.get_latest_datasets()
        
#         self.stdout.write(f"Total datasets: {len(datasets)}")
        
#         if datasets:
#             self.stdout.write("\nDatasets:")
#             for i, dataset in enumerate(datasets, 1):
#                 self.stdout.write(f"{i}. {dataset['name']} ({dataset['data_type']}) - Source: {dataset['source_name']} v{dataset['version_number']}")
#                 self.stdout.write(f"   Rows: {dataset['row_count']}, File: {dataset['file_path']}")
#                 self.stdout.write(f"   Created: {dataset['processed_at']}")
#                 self.stdout.write("")
        
#         # Get validation summary
#         validation_summary = storage_service.get_validation_summary()
        
#         if validation_summary['total_files'] > 0:
#             self.stdout.write("\nValidation Summary:")
#             self.stdout.write(f"Total files: {validation_summary['total_files']}")
#             self.stdout.write(f"Valid files: {validation_summary['valid_files']}")
#             self.stdout.write(f"Files with warnings: {validation_summary['files_with_warnings']}")
#             self.stdout.write(f"Invalid files: {validation_summary['invalid_files']}")
#             self.stdout.write(f"Total errors: {validation_summary['total_errors']}")
#             self.stdout.write(f"Total warnings: {validation_summary['total_warnings']}")
#             self.stdout.write(f"Total fixed issues: {validation_summary['total_fixed_issues']}")
        
#         # List materialized views if any
#         views = storage_service.list_materialized_views()
        
#         if views:
#             self.stdout.write("\nMaterialized Views:")
#             for i, view in enumerate(views, 1):
#                 self.stdout.write(f"{i}. {view['name']} - {view['description']}")
#                 self.stdout.write(f"   Rows: {view['row_count']}, Last updated: {view['updated_at']}")
#                 self.stdout.write(f"   Refresh frequency: {view['refresh_frequency']}")
#                 if 'last_accessed' in view:
#                     self.stdout.write(f"   Last accessed: {view['last_accessed']}, Access count: {view.get('access_count', 0)}")
#                 self.stdout.write("") 