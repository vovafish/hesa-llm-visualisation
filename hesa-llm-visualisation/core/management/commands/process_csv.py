from django.core.management.base import BaseCommand
from core.data_processing import CSVProcessor

class Command(BaseCommand):
    help = 'Process raw CSV files and create cleaned versions'

    def handle(self, *args, **options):
        processor = CSVProcessor()
        results = processor.process_all_files()
        
        # Print results
        for file_name, success in results.items():
            status = 'Successfully processed' if success else 'Failed to process'
            self.stdout.write(f"{status}: {file_name}")