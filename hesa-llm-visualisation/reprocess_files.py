# """
# Script to reprocess all CSV files and add metadata.
# """

# import os
# import sys
# from pathlib import Path

# # Add the project directory to the Python path
# BASE_DIR = Path(__file__).resolve().parent
# sys.path.append(str(BASE_DIR))

# # Set up Django environment
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hesa_llm_visualisation.settings')
# import django
# django.setup()

# from core.data_processing.csv_processor import CSVProcessor
# import logging

# logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def reprocess_all_files():
#     """Reprocess all CSV files to add metadata."""
#     logger.info("Starting to reprocess all CSV files with metadata...")
    
#     # Initialize CSV processor
#     processor = CSVProcessor()
    
#     # Reprocess all files
#     results = processor.process_all_files()
    
#     # Log results
#     success_count = sum(1 for success in results.values() if success)
#     total_count = len(results)
    
#     logger.info(f"Reprocessing complete: {success_count}/{total_count} files successfully processed.")
    
#     # List any failures
#     if success_count < total_count:
#         failed_files = [name for name, success in results.items() if not success]
#         logger.warning(f"Failed to process the following files: {failed_files}")

# if __name__ == '__main__':
#     reprocess_all_files() 