"""Scraping cache and optimization system with TTL expiration"""

import json
import time
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from enhancement.base_scraper import ScrapingResult
from utils.logging_utils import get_logger
from utils.file_utils import FileUtils

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata"""
    url: str
    result: ScrapingResult
    cached_at: float
    ttl_seconds: int
    access_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.cached_at > self.ttl_seconds
    
    def access(self) -> None:
        """Mark cache entry as accessed"""
        self.access_count += 1
        self.last_accessed = time.time()


class ScrapingCache:
    """Persistent caching system with TTL expiration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_config = config.get('cache', {})
        
        # Cache settings
        self.cache_dir = Path(config.get('directories', {}).get('cache_dir', 'cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = self.cache_config.get('default_ttl_hours', 24) * 3600  # Convert to seconds
        self.max_cache_size = self.cache_config.get('max_cache_size_mb', 100) * 1024 * 1024  # Convert to bytes
        self.cleanup_interval = self.cache_config.get('cleanup_interval_hours', 6) * 3600  # Convert to seconds
        
        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.RLock()
        
        # Cache file
        self.cache_file = self.cache_dir / 'scraping_cache.json'
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'cleanups': 0
        }
        
        # Load existing cache
        self._load_cache()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info(f"Scraping cache initialized: {len(self.cache)} entries, TTL={self.default_ttl}s")
    
    def get(self, url: str) -> Optional[ScrapingResult]:
        """Get cached result for URL"""
        try:
            cache_key = self._generate_cache_key(url)
            
            with self.cache_lock:
                if cache_key not in self.cache:
                    self.stats['misses'] += 1
                    return None
                
                entry = self.cache[cache_key]
                
                # Check if expired
                if entry.is_expired():
                    del self.cache[cache_key]
                    self.stats['misses'] += 1
                    logger.debug(f"Cache expired for {url}")
                    return None
                
                # Mark as accessed and return result
                entry.access()
                self.stats['hits'] += 1
                
                # Mark as cache hit
                result = entry.result
                result.cache_hit = True
                
                logger.debug(f"Cache hit for {url}")
                return result
                
        except Exception as e:
            logger.error(f"Error getting cache entry for {url}: {e}")
            return None
    
    def put(self, url: str, result: ScrapingResult, ttl_seconds: Optional[int] = None) -> None:
        """Cache result for URL"""
        try:
            if not result.success:
                # Don't cache failed results
                return
            
            cache_key = self._generate_cache_key(url)
            ttl = ttl_seconds or self.default_ttl
            
            entry = CacheEntry(
                url=url,
                result=result,
                cached_at=time.time(),
                ttl_seconds=ttl
            )
            
            with self.cache_lock:
                self.cache[cache_key] = entry
                
                # Check cache size and evict if necessary
                self._evict_if_needed()
            
            logger.debug(f"Cached result for {url} (TTL: {ttl}s)")
            
        except Exception as e:
            logger.error(f"Error caching result for {url}: {e}")
    
    def invalidate(self, url: str) -> bool:
        """Invalidate cache entry for URL"""
        try:
            cache_key = self._generate_cache_key(url)
            
            with self.cache_lock:
                if cache_key in self.cache:
                    del self.cache[cache_key]
                    logger.debug(f"Invalidated cache for {url}")
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {url}: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        try:
            with self.cache_lock:
                self.cache.clear()
                
            logger.info("Cache cleared")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries"""
        try:
            expired_keys = []
            
            with self.cache_lock:
                for key, entry in self.cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.cache[key]
            
            if expired_keys:
                self.stats['cleanups'] += 1
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
            
            return len(expired_keys)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.cache_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            cache_size_bytes = self._calculate_cache_size()
            
            return {
                'entries': len(self.cache),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': round(hit_rate, 2),
                'evictions': self.stats['evictions'],
                'cleanups': self.stats['cleanups'],
                'cache_size_bytes': cache_size_bytes,
                'cache_size_mb': round(cache_size_bytes / (1024 * 1024), 2),
                'max_cache_size_mb': self.max_cache_size / (1024 * 1024),
                'default_ttl_hours': self.default_ttl / 3600
            }
    
    def save_cache(self) -> bool:
        """Save cache to disk"""
        try:
            with self.cache_lock:
                cache_data = {
                    'version': '1.0',
                    'saved_at': time.time(),
                    'entries': {},
                    'stats': self.stats
                }
                
                for key, entry in self.cache.items():
                    if not entry.is_expired():
                        # Convert ScrapingResult to dict for JSON serialization
                        result_dict = asdict(entry.result)
                        
                        cache_data['entries'][key] = {
                            'url': entry.url,
                            'result': result_dict,
                            'cached_at': entry.cached_at,
                            'ttl_seconds': entry.ttl_seconds,
                            'access_count': entry.access_count,
                            'last_accessed': entry.last_accessed
                        }
                
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"Saved {len(cache_data['entries'])} cache entries to disk")
                return True
                
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
            return False
    
    def _load_cache(self) -> None:
        """Load cache from disk"""
        try:
            if not self.cache_file.exists():
                return
            
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            entries_loaded = 0
            
            for key, entry_data in cache_data.get('entries', {}).items():
                try:
                    # Convert dict back to ScrapingResult
                    result_dict = entry_data['result']
                    result = ScrapingResult(**result_dict)
                    
                    entry = CacheEntry(
                        url=entry_data['url'],
                        result=result,
                        cached_at=entry_data['cached_at'],
                        ttl_seconds=entry_data['ttl_seconds'],
                        access_count=entry_data.get('access_count', 0),
                        last_accessed=entry_data.get('last_accessed', 0.0)
                    )
                    
                    # Only load non-expired entries
                    if not entry.is_expired():
                        self.cache[key] = entry
                        entries_loaded += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load cache entry {key}: {e}")
                    continue
            
            # Load stats
            if 'stats' in cache_data:
                self.stats.update(cache_data['stats'])
            
            logger.info(f"Loaded {entries_loaded} cache entries from disk")
            
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
    
    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()
    
    def _calculate_cache_size(self) -> int:
        """Calculate approximate cache size in bytes"""
        try:
            total_size = 0
            
            for entry in self.cache.values():
                # Rough estimation of entry size
                entry_size = (
                    len(entry.url.encode('utf-8')) +
                    len(str(entry.result.title or '').encode('utf-8')) +
                    len(str(entry.result.description or '').encode('utf-8')) +
                    sum(len(kw.encode('utf-8')) for kw in entry.result.keywords) +
                    len(json.dumps(entry.result.metadata).encode('utf-8')) +
                    200  # Overhead for other fields
                )
                total_size += entry_size
            
            return total_size
            
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
            return 0
    
    def _evict_if_needed(self) -> None:
        """Evict cache entries if size limit exceeded"""
        try:
            cache_size = self._calculate_cache_size()
            
            if cache_size <= self.max_cache_size:
                return
            
            # Sort entries by last accessed time (LRU eviction)
            entries_by_access = sorted(
                self.cache.items(),
                key=lambda x: x[1].last_accessed or x[1].cached_at
            )
            
            # Evict oldest entries until under size limit
            evicted_count = 0
            for key, entry in entries_by_access:
                del self.cache[key]
                evicted_count += 1
                
                # Check size again
                cache_size = self._calculate_cache_size()
                if cache_size <= self.max_cache_size * 0.8:  # Leave some headroom
                    break
            
            self.stats['evictions'] += evicted_count
            logger.info(f"Evicted {evicted_count} cache entries due to size limit")
            
        except Exception as e:
            logger.error(f"Error evicting cache entries: {e}")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self.cleanup_expired()
                self.save_cache()
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")


class ConcurrentScraper:
    """Concurrent scraping with configurable thread pools"""
    
    def __init__(self, config: Dict[str, Any], scrapers: List, cache: ScrapingCache):
        self.config = config
        self.scrapers = scrapers
        self.cache = cache
        
        # Concurrency settings
        self.max_workers = config.get('scraping', {}).get('max_workers', 5)
        self.retry_attempts = config.get('scraping', {}).get('retry_attempts', 3)
        self.retry_delay = config.get('scraping', {}).get('retry_delay', 1.0)
        
        logger.info(f"Concurrent scraper initialized with {self.max_workers} workers")
    
    def scrape_urls(self, urls: List[str]) -> Dict[str, ScrapingResult]:
        """Scrape multiple URLs concurrently"""
        results = {}
        
        if not urls:
            return results
        
        logger.info(f"Starting concurrent scraping of {len(urls)} URLs")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all scraping tasks
            future_to_url = {
                executor.submit(self._scrape_with_retry, url): url 
                for url in urls
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results[url] = result
                    
                    if result.success:
                        logger.debug(f"Successfully scraped {url}")
                    else:
                        logger.warning(f"Failed to scrape {url}: {result.error_message}")
                        
                except Exception as e:
                    logger.error(f"Exception scraping {url}: {e}")
                    results[url] = ScrapingResult(
                        url=url,
                        success=False,
                        error_message=str(e),
                        scraper_used="ConcurrentScraper"
                    )
        
        success_count = sum(1 for result in results.values() if result.success)
        logger.info(f"Concurrent scraping completed: {success_count}/{len(urls)} successful")
        
        return results
    
    def _scrape_with_retry(self, url: str) -> ScrapingResult:
        """Scrape URL with retry logic"""
        # Check cache first
        cached_result = self.cache.get(url)
        if cached_result:
            return cached_result
        
        # Try scraping with retries
        last_error = None
        
        for attempt in range(self.retry_attempts):
            try:
                # Find appropriate scraper
                scraper = self._find_scraper(url)
                if not scraper:
                    return ScrapingResult(
                        url=url,
                        success=False,
                        error_message="No suitable scraper found",
                        scraper_used="ConcurrentScraper"
                    )
                
                # Attempt scraping
                result = scraper.scrape(url)
                
                if result.success:
                    # Cache successful result
                    self.cache.put(url, result)
                    return result
                else:
                    last_error = result.error_message
                    
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Scraping attempt {attempt + 1} failed for {url}: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.retry_attempts - 1:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
        
        # All attempts failed
        return ScrapingResult(
            url=url,
            success=False,
            error_message=f"All {self.retry_attempts} attempts failed. Last error: {last_error}",
            scraper_used="ConcurrentScraper"
        )
    
    def _find_scraper(self, url: str):
        """Find appropriate scraper for URL"""
        for scraper in self.scrapers:
            if scraper.can_scrape(url):
                return scraper
        return None