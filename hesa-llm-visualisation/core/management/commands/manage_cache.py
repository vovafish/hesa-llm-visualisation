from django.core.management.base import BaseCommand
from core.utils.query_cache import QueryCache
import datetime

class Command(BaseCommand):
    help = 'Manage the query cache system'

    def add_arguments(self, parser):
        parser.add_argument(
            '--stats',
            action='store_true',
            help='Show cache statistics',
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear all cache entries',
        )
        parser.add_argument(
            '--clear-expired',
            action='store_true',
            help='Clear only expired cache entries',
        )
        parser.add_argument(
            '--query',
            type=str,
            help='Specify a query to clear from cache',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Initializing query cache management...'))
        
        # Initialize the query cache
        cache = QueryCache()
        
        if options['clear']:
            # Clear all cache entries
            count = cache.clear()
            self.stdout.write(self.style.SUCCESS(f'Cleared {count} entries from the query cache'))
        
        elif options['clear_expired']:
            # Get all cache files and check each one for expiration
            self.stdout.write('Checking for expired cache entries...')
            
            from pathlib import Path
            import json
            import os
            
            count = 0
            for cache_file in Path(cache.cache_dir).glob('*.json'):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    # Check if the cache has expired (30 days)
                    cached_time = datetime.datetime.fromisoformat(cache_data.get('cached_at', ''))
                    expiration = datetime.timedelta(days=30)
                    
                    if datetime.datetime.now() - cached_time > expiration:
                        self.stdout.write(f'  Removing expired cache: {cache_file.name}')
                        os.remove(cache_file)
                        count += 1
                
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'  Error checking cache file {cache_file}: {e}'))
                    # Remove invalid cache file
                    os.remove(cache_file)
                    count += 1
            
            self.stdout.write(self.style.SUCCESS(f'Cleared {count} expired entries from the query cache'))
        
        elif options['query']:
            # Clear a specific query from cache
            query = options['query']
            count = cache.clear(query)
            if count > 0:
                self.stdout.write(self.style.SUCCESS(f'Cleared cache for query: {query}'))
            else:
                self.stdout.write(self.style.WARNING(f'No cache found for query: {query}'))
        
        else:
            # Show cache statistics by default
            stats = cache.get_cache_stats()
            
            self.stdout.write('\nQuery Cache Statistics:')
            self.stdout.write(f"Cache directory: {cache.cache_dir}")
            self.stdout.write(f"Total entries: {stats['total_entries']}")
            self.stdout.write(f"Total size: {stats.get('total_size', '0 KB')}")
            
            if stats['newest_entry']:
                self.stdout.write(f"\nNewest entry: {stats['newest_entry']['query']}")
                self.stdout.write(f"  Cached at: {stats['newest_entry']['cached_at']}")
            
            if stats['oldest_entry']:
                self.stdout.write(f"\nOldest entry: {stats['oldest_entry']['query']}")
                self.stdout.write(f"  Cached at: {stats['oldest_entry']['cached_at']}")
            
            if stats['most_accessed']:
                self.stdout.write(f"\nMost accessed entry: {stats['most_accessed']['query']}")
                self.stdout.write(f"  Access count: {stats['most_accessed']['access_count']}")
        
        self.stdout.write(self.style.SUCCESS('Query cache management completed')) 