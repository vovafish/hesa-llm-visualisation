import os
import json
import hashlib
import datetime
import logging
from pathlib import Path
from django.conf import settings

logger = logging.getLogger(__name__)

class QueryCache:
    """
    A simple file-based cache for query results with expiration handling.
    Caches query results in JSON format in the data/cache directory.
    """
    
    def __init__(self):
        """Initialize the query cache with the cache directory path."""
        # Get base directory from settings or use default
        base_dir = getattr(settings, 'BASE_DIR', None)
        if base_dir:
            self.cache_dir = Path(base_dir) / 'data' / 'cache'
        else:
            # Fallback to a relative path if BASE_DIR is not defined
            self.cache_dir = Path(__file__).resolve().parent.parent.parent.parent / 'data' / 'cache'
        
        # Ensure the cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Query cache initialized with directory: {self.cache_dir}")
    
    def _generate_cache_key(self, query):
        """
        Generate a unique cache key for a query.
       
        """
        # Create an MD5 hash of the query string to use as the filename
        # This ensures unique filenames and handles special characters
        hash_obj = hashlib.md5(query.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _get_cache_file_path(self, cache_key):
        """
        Get the full path to a cache file.
        
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, query):
        """
        Retrieve a cached query result if it exists and hasn't expired.
        
        """
        cache_key = self._generate_cache_key(query)
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            # Check if the cache has expired (30 days)
            cached_time = datetime.datetime.fromisoformat(cached_data.get('cached_at', ''))
            expiration = datetime.timedelta(days=30)
            
            if datetime.datetime.now() - cached_time > expiration:
                logger.debug(f"Cache expired for query: {query[:50]}...")
                # Remove the expired cache file
                os.remove(cache_file)
                return None
            
            logger.debug(f"Cache hit for query: {query[:50]}...")
            
            # Update access count and last accessed time
            cached_data['access_count'] = cached_data.get('access_count', 0) + 1
            cached_data['last_accessed'] = datetime.datetime.now().isoformat()
            
            # Write the updated metadata back to the cache
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
            
            return cached_data.get('result')
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error reading cache file {cache_file}: {str(e)}")
            # Remove invalid cache file
            if cache_file.exists():
                os.remove(cache_file)
            return None
    
    def set(self, query, result):
        """
        Cache a query result.
        
     
        """
        cache_key = self._generate_cache_key(query)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            # Create cache data structure with metadata
            cache_data = {
                'query': query,
                'result': result,
                'cached_at': datetime.datetime.now().isoformat(),
                'access_count': 0,
                'last_accessed': None
            }
            
            # Write to cache file
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Cached result for query: {query[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error caching query result: {str(e)}")
            return False
    
    def clear(self, query=None):
        """
        Clear specific cache entry or all cache entries.
        
        """
        if query:
            # Clear specific cache entry
            cache_key = self._generate_cache_key(query)
            cache_file = self._get_cache_file_path(cache_key)
            
            if cache_file.exists():
                os.remove(cache_file)
                logger.info(f"Cleared cache for query: {query[:50]}...")
                return 1
            return 0
        else:
            # Clear all cache entries
            count = 0
            for cache_file in self.cache_dir.glob('*.json'):
                os.remove(cache_file)
                count += 1
            
            logger.info(f"Cleared {count} entries from the query cache")
            return count
    
    def get_cache_stats(self):
        """
        Get statistics about the current cache.
    
        """
        stats = {
            'total_entries': 0,
            'total_size_bytes': 0,
            'oldest_entry': None,
            'newest_entry': None,
            'most_accessed': None,
            'most_accessed_count': 0
        }
        
        # Get all cache files
        cache_files = list(self.cache_dir.glob('*.json'))
        stats['total_entries'] = len(cache_files)
        
        oldest_time = None
        newest_time = None
        
        for cache_file in cache_files:
            # Get file size
            file_size = os.path.getsize(cache_file)
            stats['total_size_bytes'] += file_size
            
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                cached_time = datetime.datetime.fromisoformat(cache_data.get('cached_at', ''))
                
                # Track oldest and newest entries
                if oldest_time is None or cached_time < oldest_time:
                    oldest_time = cached_time
                    stats['oldest_entry'] = {
                        'query': cache_data.get('query', '')[:50] + '...',
                        'cached_at': cached_time.isoformat()
                    }
                
                if newest_time is None or cached_time > newest_time:
                    newest_time = cached_time
                    stats['newest_entry'] = {
                        'query': cache_data.get('query', '')[:50] + '...',
                        'cached_at': cached_time.isoformat()
                    }
                
                # Track most accessed entry
                access_count = cache_data.get('access_count', 0)
                if access_count > stats['most_accessed_count']:
                    stats['most_accessed_count'] = access_count
                    stats['most_accessed'] = {
                        'query': cache_data.get('query', '')[:50] + '...',
                        'access_count': access_count
                    }
                    
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading cache file {cache_file}: {str(e)}")
        
        # Convert total size to a more readable format
        if stats['total_size_bytes'] > 1024 * 1024:
            stats['total_size'] = f"{stats['total_size_bytes'] / (1024 * 1024):.2f} MB"
        else:
            stats['total_size'] = f"{stats['total_size_bytes'] / 1024:.2f} KB"
        
        return stats 