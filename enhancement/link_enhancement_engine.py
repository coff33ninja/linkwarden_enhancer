"""LinkEnhancementEngine - Main orchestrator for bookmark enhancement"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse
from pathlib import Path

from enhancement.base_scraper import WebScraper, ScrapingResult
from enhancement.beautifulsoup_scraper import BeautifulSoupScraper
from enhancement.scraping_cache import ScrapingCache, ConcurrentScraper

try:
    from enhancement.selenium_scraper import SeleniumScraper
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from enhancement.newspaper_scraper import NewspaperScraper
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
from utils.logging_utils import get_logger
from data_models import EnhancementReport

logger = get_logger(__name__)

@dataclass
class EnhancementConfig:
    """Configuration for link enhancement"""
    enable_scraping: bool = True
    enable_caching: bool = True
    max_concurrent_requests: int = 5
    default_timeout: int = 30
    retry_attempts: int = 3
    cache_ttl_hours: int = 24
    max_cache_size_mb: int = 100
    scrapers_to_use: List[str] = field(default_factory=lambda: ['beautifulsoup'])
    fallback_enabled: bool = True
    skip_binary_files: bool = True
    respect_robots_txt: bool = True


class LinkEnhancementEngine:
    """Main orchestrator for bookmark enhancement with scraper coordination"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize link enhancement engine"""
        self.config = config
        self.enhancement_config = config.get('enhancement', {})
        
        # Create enhancement configuration
        self.econfig = EnhancementConfig(
            enable_scraping=self.enhancement_config.get('enable_scraping', True),
            enable_caching=self.enhancement_config.get('enable_caching', True),
            max_concurrent_requests=self.enhancement_config.get('max_concurrent_requests', 5),
            default_timeout=self.enhancement_config.get('default_timeout', 30),
            retry_attempts=self.enhancement_config.get('retry_attempts', 3),
            cache_ttl_hours=self.enhancement_config.get('cache_ttl_hours', 24),
            max_cache_size_mb=self.enhancement_config.get('max_cache_size_mb', 100),
            scrapers_to_use=self.enhancement_config.get('scrapers_to_use', ['beautifulsoup', 'selenium', 'newspaper']),
            fallback_enabled=self.enhancement_config.get('fallback_enabled', True),
            skip_binary_files=self.enhancement_config.get('skip_binary_files', True),
            respect_robots_txt=self.enhancement_config.get('respect_robots_txt', True)
        )
        
        # Initialize cache
        self.cache = None
        if self.econfig.enable_caching:
            self.cache = ScrapingCache(config)
        
        # Initialize scrapers
        self.scrapers = self._initialize_scrapers()
        
        # Initialize concurrent scraper
        self.concurrent_scraper = None
        if self.cache:
            self.concurrent_scraper = ConcurrentScraper(config, self.scrapers, self.cache)
        
        # Statistics
        self.stats = {
            'total_enhanced': 0,
            'successful_enhancements': 0,
            'failed_enhancements': 0,
            'cache_hits': 0,
            'scraping_time_total': 0.0,
            'scrapers_used': {},
            'error_types': {}
        }
        
        logger.info(f"LinkEnhancementEngine initialized with {len(self.scrapers)} scrapers")
    
    def _initialize_scrapers(self) -> List[WebScraper]:
        """Initialize available scrapers based on configuration"""
        scrapers = []
        
        try:
            # BeautifulSoup scraper (always available)
            if 'beautifulsoup' in self.econfig.scrapers_to_use:
                try:
                    scraper = BeautifulSoupScraper(self.config)
                    scrapers.append(scraper)
                    logger.info("BeautifulSoup scraper initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize BeautifulSoup scraper: {e}")
            
            # Selenium scraper for JavaScript-heavy sites
            if 'selenium' in self.econfig.scrapers_to_use and SELENIUM_AVAILABLE:
                try:
                    scraper = SeleniumScraper(self.config)
                    scrapers.append(scraper)
                    logger.info("Selenium scraper initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Selenium scraper: {e}")
            
            # Newspaper3k scraper for article content
            if 'newspaper' in self.econfig.scrapers_to_use and NEWSPAPER_AVAILABLE:
                try:
                    scraper = NewspaperScraper(self.config)
                    scrapers.append(scraper)
                    logger.info("Newspaper scraper initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Newspaper scraper: {e}")
            
            # TODO: Add other scrapers when implemented
            # - Custom scrapers for specific domains
            
            if not scrapers:
                logger.warning("No scrapers could be initialized")
            
            return scrapers
            
        except Exception as e:
            logger.error(f"Failed to initialize scrapers: {e}")
            return []
    
    def enhance_bookmark(self, url: str, existing_data: Optional[Dict[str, Any]] = None) -> ScrapingResult:
        """Enhance a single bookmark with metadata"""
        
        if not self.econfig.enable_scraping:
            return ScrapingResult(
                url=url,
                success=False,
                error_message="Scraping is disabled",
                scraper_used="LinkEnhancementEngine"
            )
        
        start_time = time.time()
        
        try:
            logger.debug(f"Enhancing bookmark: {url}")
            
            # Check cache first
            if self.cache:
                cached_result = self.cache.get(url)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    self.stats['total_enhanced'] += 1
                    logger.debug(f"Cache hit for {url}")
                    return cached_result
            
            # Find appropriate scraper
            scraper = self._select_scraper(url)
            if not scraper:
                error_msg = "No suitable scraper found for URL"
                self._update_error_stats(error_msg)
                return ScrapingResult(
                    url=url,
                    success=False,
                    error_message=error_msg,
                    scraper_used="LinkEnhancementEngine"
                )
            
            # Perform scraping
            result = scraper.scrape(url)
            
            # Update statistics
            scraping_time = time.time() - start_time
            self.stats['total_enhanced'] += 1
            self.stats['scraping_time_total'] += scraping_time
            
            if result.success:
                self.stats['successful_enhancements'] += 1
                self._update_scraper_stats(result.scraper_used)
                
                # Cache successful result
                if self.cache:
                    self.cache.put(url, result)
                
                logger.debug(f"Successfully enhanced {url} in {scraping_time:.2f}s")
            else:
                self.stats['failed_enhancements'] += 1
                self._update_error_stats(result.error_message or "Unknown error")
                logger.warning(f"Failed to enhance {url}: {result.error_message}")
            
            return result
            
        except Exception as e:
            error_msg = f"Enhancement failed: {e}"
            logger.error(f"Error enhancing {url}: {error_msg}")
            
            self.stats['total_enhanced'] += 1
            self.stats['failed_enhancements'] += 1
            self._update_error_stats(error_msg)
            
            return ScrapingResult(
                url=url,
                success=False,
                error_message=error_msg,
                scraping_time=time.time() - start_time,
                scraper_used="LinkEnhancementEngine"
            )
    
    def enhance_bookmarks_batch(self, urls: List[str]) -> Dict[str, ScrapingResult]:
        """Enhance multiple bookmarks concurrently"""
        
        if not self.econfig.enable_scraping:
            return {url: ScrapingResult(
                url=url,
                success=False,
                error_message="Scraping is disabled",
                scraper_used="LinkEnhancementEngine"
            ) for url in urls}
        
        if not urls:
            return {}
        
        logger.info(f"Starting batch enhancement of {len(urls)} bookmarks")
        start_time = time.time()
        
        try:
            # Filter URLs that can be scraped
            scrapable_urls = []
            results = {}
            
            for url in urls:
                if self._can_enhance_url(url):
                    scrapable_urls.append(url)
                else:
                    results[url] = ScrapingResult(
                        url=url,
                        success=False,
                        error_message="URL cannot be scraped (binary file or unsupported)",
                        scraper_used="LinkEnhancementEngine"
                    )
            
            # Use concurrent scraper if available
            if self.concurrent_scraper and len(scrapable_urls) > 1:
                concurrent_results = self.concurrent_scraper.scrape_urls(scrapable_urls)
                results.update(concurrent_results)
            else:
                # Fallback to sequential processing
                for url in scrapable_urls:
                    results[url] = self.enhance_bookmark(url)
            
            # Update batch statistics
            batch_time = time.time() - start_time
            successful = sum(1 for result in results.values() if result.success)
            
            logger.info(f"Batch enhancement completed: {successful}/{len(urls)} successful in {batch_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch enhancement failed: {e}")
            
            # Return error results for all URLs
            return {url: ScrapingResult(
                url=url,
                success=False,
                error_message=f"Batch enhancement failed: {e}",
                scraper_used="LinkEnhancementEngine"
            ) for url in urls}
    
    def enhance_bookmark_collection(self, bookmarks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], EnhancementReport]:
        """Enhance a collection of bookmarks and return enhanced data with report"""
        
        logger.info(f"Enhancing collection of {len(bookmarks)} bookmarks")
        start_time = time.time()
        
        # Extract URLs from bookmarks
        urls = []
        url_to_bookmark = {}
        
        for bookmark in bookmarks:
            url = bookmark.get('url')
            if url:
                urls.append(url)
                url_to_bookmark[url] = bookmark
        
        # Enhance all URLs
        enhancement_results = self.enhance_bookmarks_batch(urls)
        
        # Apply enhancements to bookmarks
        enhanced_bookmarks = []
        metadata_fields_added = 0
        scraping_failures = 0
        scrapers_used = {}
        total_scraping_time = 0.0
        
        for bookmark in bookmarks:
            url = bookmark.get('url')
            enhanced_bookmark = bookmark.copy()
            
            if url and url in enhancement_results:
                result = enhancement_results[url]
                
                if result.success:
                    # Apply enhancements
                    enhanced_bookmark.update(self._apply_enhancement_to_bookmark(bookmark, result))
                    
                    # Count metadata fields added
                    original_fields = len([k for k, v in bookmark.items() if v])
                    enhanced_fields = len([k for k, v in enhanced_bookmark.items() if v])
                    metadata_fields_added += max(0, enhanced_fields - original_fields)
                    
                    # Track scraper usage
                    scraper_name = result.scraper_used
                    scrapers_used[scraper_name] = scrapers_used.get(scraper_name, 0) + 1
                    
                    total_scraping_time += result.scraping_time
                else:
                    scraping_failures += 1
            
            enhanced_bookmarks.append(enhanced_bookmark)
        
        # Calculate cache hit rate
        cache_hits = sum(1 for result in enhancement_results.values() if result.cache_hit)
        cache_hit_rate = (cache_hits / len(enhancement_results)) * 100 if enhancement_results else 0
        
        # Calculate average scraping time
        successful_scrapings = sum(1 for result in enhancement_results.values() if result.success)
        average_scraping_time = (total_scraping_time / successful_scrapings) if successful_scrapings > 0 else 0
        
        # Create enhancement report
        report = EnhancementReport(
            bookmarks_enhanced=len(bookmarks),
            metadata_fields_added=metadata_fields_added,
            scraping_failures=scraping_failures,
            scrapers_used=scrapers_used,
            average_scraping_time=average_scraping_time,
            cache_hit_rate=cache_hit_rate
        )
        
        total_time = time.time() - start_time
        logger.info(f"Collection enhancement completed in {total_time:.2f}s: {len(enhanced_bookmarks)} bookmarks processed")
        
        return enhanced_bookmarks, report
    
    def _select_scraper(self, url: str) -> Optional[WebScraper]:
        """Select appropriate scraper for URL based on patterns and capabilities"""
        
        if not self.scrapers:
            return None
        
        try:
            # Find scrapers that can handle this URL
            capable_scrapers = [scraper for scraper in self.scrapers if scraper.can_scrape(url)]
            
            if not capable_scrapers:
                # If no scraper specifically claims it can handle the URL,
                # use the first available scraper as fallback
                if self.econfig.fallback_enabled and self.scrapers:
                    return self.scrapers[0]
                return None
            
            # Categorize scrapers by type
            selenium_scrapers = [s for s in capable_scrapers if 'Selenium' in s.__class__.__name__]
            newspaper_scrapers = [s for s in capable_scrapers if 'Newspaper' in s.__class__.__name__]
            beautifulsoup_scrapers = [s for s in capable_scrapers if 'BeautifulSoup' in s.__class__.__name__]
            
            # Check URL characteristics
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Check for article/news content (highest priority for Newspaper scraper)
            article_indicators = [
                'news.', 'blog.', 'article.', 'post.',
                '/article/', '/post/', '/blog/', '/news/', '/story/',
                'medium.com', 'substack.com', 'dev.to', 'techcrunch.com',
                'arstechnica.com', 'theverge.com', 'bbc.com', 'cnn.com'
            ]
            
            is_article = any(indicator in domain or indicator in path for indicator in article_indicators)
            
            # Check for JavaScript-heavy content
            js_indicators = [
                'app.', 'dashboard.', 'admin.', 'console.', 'portal.',
                '/app/', '/dashboard/', '/admin/', '/console/',
                'github.com', 'gitlab.com'  # Some pages are SPAs
            ]
            
            needs_js = any(indicator in domain or indicator in path for indicator in js_indicators)
            
            # Select scraper based on content type priority
            if is_article and newspaper_scrapers:
                logger.debug(f"Using Newspaper scraper for article content: {url}")
                return newspaper_scrapers[0]
            elif needs_js and selenium_scrapers:
                logger.debug(f"Using Selenium scraper for JavaScript-heavy site: {url}")
                return selenium_scrapers[0]
            elif beautifulsoup_scrapers:
                logger.debug(f"Using BeautifulSoup scraper for standard site: {url}")
                return beautifulsoup_scrapers[0]
            else:
                # Return first capable scraper
                return capable_scrapers[0]
            
        except Exception as e:
            logger.error(f"Error selecting scraper for {url}: {e}")
            return None
    
    def _can_enhance_url(self, url: str) -> bool:
        """Check if URL can be enhanced"""
        
        if not url:
            return False
        
        try:
            parsed = urlparse(url)
            
            # Skip non-HTTP(S) URLs
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Skip binary files if configured
            if self.econfig.skip_binary_files:
                binary_extensions = {
                    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                    '.zip', '.rar', '.tar', '.gz', '.7z',
                    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
                    '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
                    '.exe', '.msi', '.dmg', '.deb', '.rpm'
                }
                
                path = parsed.path.lower()
                if any(path.endswith(ext) for ext in binary_extensions):
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking if can enhance {url}: {e}")
            return False
    
    def _apply_enhancement_to_bookmark(self, original_bookmark: Dict[str, Any], result: ScrapingResult) -> Dict[str, Any]:
        """Apply scraping result to bookmark data"""
        
        enhanced = original_bookmark.copy()
        
        try:
            # Update title if not present or if scraped title is better
            if result.title and (not enhanced.get('name') or len(result.title) > len(enhanced.get('name', ''))):
                enhanced['name'] = result.title
            
            # Update description if not present
            if result.description and not enhanced.get('description'):
                enhanced['description'] = result.description
            
            # Add or merge keywords/tags
            if result.keywords:
                existing_tags = enhanced.get('tags', [])
                
                # Convert existing tags to list of strings
                tag_names = set()
                for tag in existing_tags:
                    if isinstance(tag, dict):
                        tag_name = tag.get('name', '')
                    else:
                        tag_name = str(tag)
                    if tag_name:
                        tag_names.add(tag_name.lower())
                
                # Add new keywords as tags
                new_tags = []
                for keyword in result.keywords:
                    if keyword.lower() not in tag_names:
                        new_tags.append({'name': keyword})
                        tag_names.add(keyword.lower())
                
                if new_tags:
                    enhanced['tags'] = existing_tags + new_tags
            
            # Add enhancement metadata
            if not enhanced.get('content'):
                enhanced['content'] = {}
            
            enhanced['content'].update({
                'enhanced_at': time.time(),
                'scraper_used': result.scraper_used,
                'scraping_time': result.scraping_time,
                'favicon_url': result.favicon_url,
                'image_url': result.image_url,
                'author': result.author,
                'published_date': result.published_date,
                'content_type': result.content_type,
                'language': result.language,
                'enhancement_metadata': result.metadata
            })
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error applying enhancement to bookmark: {e}")
            return enhanced
    
    def _update_scraper_stats(self, scraper_name: str) -> None:
        """Update scraper usage statistics"""
        self.stats['scrapers_used'][scraper_name] = self.stats['scrapers_used'].get(scraper_name, 0) + 1
    
    def _update_error_stats(self, error_message: str) -> None:
        """Update error statistics"""
        # Categorize error types
        error_type = "unknown"
        
        if "timeout" in error_message.lower():
            error_type = "timeout"
        elif "connection" in error_message.lower():
            error_type = "connection"
        elif "not found" in error_message.lower() or "404" in error_message:
            error_type = "not_found"
        elif "forbidden" in error_message.lower() or "403" in error_message:
            error_type = "forbidden"
        elif "rate limit" in error_message.lower():
            error_type = "rate_limit"
        elif "scraper" in error_message.lower():
            error_type = "scraper_error"
        
        self.stats['error_types'][error_type] = self.stats['error_types'].get(error_type, 0) + 1
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhancement statistics"""
        
        cache_stats = {}
        if self.cache:
            cache_stats = self.cache.get_stats()
        
        return {
            'enhancement_stats': self.stats.copy(),
            'cache_stats': cache_stats,
            'scrapers_available': len(self.scrapers),
            'scraper_info': [scraper.get_scraper_info() for scraper in self.scrapers],
            'configuration': {
                'enable_scraping': self.econfig.enable_scraping,
                'enable_caching': self.econfig.enable_caching,
                'max_concurrent_requests': self.econfig.max_concurrent_requests,
                'scrapers_to_use': self.econfig.scrapers_to_use,
                'fallback_enabled': self.econfig.fallback_enabled
            }
        }
    
    def clear_cache(self) -> bool:
        """Clear enhancement cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Enhancement cache cleared")
            return True
        return False
    
    def save_cache(self) -> bool:
        """Save cache to disk"""
        if self.cache:
            return self.cache.save_cache()
        return False
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.cache:
                self.cache.save_cache()
            
            # Close scraper sessions and cleanup resources
            for scraper in self.scrapers:
                if hasattr(scraper, 'cleanup'):
                    scraper.cleanup()
                elif hasattr(scraper, 'session'):
                    scraper.session.close()
            
            logger.info("LinkEnhancementEngine cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")