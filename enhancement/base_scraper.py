"""Base web scraper framework with rate limiting and common functionality"""

import time
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.logging_utils import get_logger
from utils.url_utils import UrlUtils

logger = get_logger(__name__)


@dataclass
class ScrapingResult:
    """Result from web scraping operation"""
    url: str
    success: bool
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    favicon_url: Optional[str] = None
    image_url: Optional[str] = None
    author: Optional[str] = None
    published_date: Optional[str] = None
    content_type: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    scraping_time: float = 0.0
    scraper_used: str = ""
    error_message: Optional[str] = None
    cache_hit: bool = False


class RateLimiter:
    """Rate limiting system to respect website policies"""
    
    def __init__(self, requests_per_second: float = 1.0, burst_size: int = 5):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.domain_limits = {}
    
    def wait_if_needed(self, domain: str) -> None:
        """Wait if rate limit would be exceeded"""
        current_time = time.time()
        
        # Refill tokens based on time passed
        time_passed = current_time - self.last_update
        self.tokens = min(self.burst_size, self.tokens + time_passed * self.requests_per_second)
        self.last_update = current_time
        
        # Check domain-specific limits
        if domain in self.domain_limits:
            domain_last_request = self.domain_limits[domain]
            time_since_last = current_time - domain_last_request
            min_interval = 1.0 / self.requests_per_second
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
                time.sleep(sleep_time)
        
        # Use a token
        if self.tokens < 1:
            sleep_time = (1 - self.tokens) / self.requests_per_second
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s (token bucket)")
            time.sleep(sleep_time)
            self.tokens = 0
        else:
            self.tokens -= 1
        
        # Record this request for domain limiting
        self.domain_limits[domain] = time.time()


class WebScraper(ABC):
    """Abstract base class for web scrapers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scraping_config = config.get('scraping', {})
        
        # Rate limiting
        self.rate_limiter = RateLimiter(
            requests_per_second=self.scraping_config.get('requests_per_second', 1.0),
            burst_size=self.scraping_config.get('burst_size', 5)
        )
        
        # HTTP session with retries
        self.session = self._create_session()
        
        # User agent
        self.user_agent = self.scraping_config.get(
            'user_agent', 
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        
        # Timeouts
        self.timeout = self.scraping_config.get('timeout', 30)
        self.max_content_size = self.scraping_config.get('max_content_size', 10 * 1024 * 1024)  # 10MB
        
        logger.info(f"Initialized {self.__class__.__name__} scraper")
    
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @abstractmethod
    def can_scrape(self, url: str) -> bool:
        """Check if this scraper can handle the given URL"""
        pass
    
    @abstractmethod
    def scrape(self, url: str) -> ScrapingResult:
        """Scrape the given URL and return results"""
        pass
    
    def _make_request(self, url: str, **kwargs) -> Optional[requests.Response]:
        """Make HTTP request with rate limiting and error handling"""
        try:
            domain = UrlUtils.extract_domain(url)
            self.rate_limiter.wait_if_needed(domain)
            
            headers = kwargs.get('headers', {})
            headers['User-Agent'] = self.user_agent
            kwargs['headers'] = headers
            kwargs['timeout'] = kwargs.get('timeout', self.timeout)
            
            logger.debug(f"Making request to {url}")
            response = self.session.get(url, **kwargs)
            response.raise_for_status()
            
            # Check content size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_content_size:
                logger.warning(f"Content too large: {content_length} bytes for {url}")
                return None
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None
    
    def _extract_favicon(self, url: str, soup=None) -> Optional[str]:
        """Extract favicon URL from page"""
        try:
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            
            # Try to find favicon in HTML
            if soup:
                # Look for various favicon link tags
                favicon_selectors = [
                    'link[rel="icon"]',
                    'link[rel="shortcut icon"]',
                    'link[rel="apple-touch-icon"]',
                    'link[rel="apple-touch-icon-precomposed"]'
                ]
                
                for selector in favicon_selectors:
                    favicon_link = soup.select_one(selector)
                    if favicon_link and favicon_link.get('href'):
                        favicon_url = urljoin(url, favicon_link['href'])
                        return favicon_url
            
            # Fallback to standard favicon.ico
            favicon_url = urljoin(base_url, '/favicon.ico')
            
            # Check if favicon exists
            response = self._make_request(favicon_url, stream=True)
            if response and response.status_code == 200:
                return favicon_url
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract favicon for {url}: {e}")
            return None
    
    def _extract_meta_tags(self, soup) -> Dict[str, Any]:
        """Extract metadata from HTML meta tags"""
        metadata = {}
        
        if not soup:
            return metadata
        
        try:
            # Standard meta tags
            meta_tags = {
                'description': ['name="description"', 'property="description"'],
                'keywords': ['name="keywords"'],
                'author': ['name="author"'],
                'language': ['name="language"', 'http-equiv="content-language"']
            }
            
            for key, selectors in meta_tags.items():
                for selector in selectors:
                    meta = soup.select_one(f'meta[{selector}]')
                    if meta and meta.get('content'):
                        metadata[key] = meta['content']
                        break
            
            # Open Graph tags
            og_tags = soup.select('meta[property^="og:"]')
            for tag in og_tags:
                property_name = tag.get('property', '').replace('og:', '')
                content = tag.get('content')
                if property_name and content:
                    metadata[f'og_{property_name}'] = content
            
            # Twitter Card tags
            twitter_tags = soup.select('meta[name^="twitter:"]')
            for tag in twitter_tags:
                name = tag.get('name', '').replace('twitter:', '')
                content = tag.get('content')
                if name and content:
                    metadata[f'twitter_{name}'] = content
            
            return metadata
            
        except Exception as e:
            logger.debug(f"Failed to extract meta tags: {e}")
            return metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove common unwanted characters
        text = text.replace('\u00a0', ' ')  # Non-breaking space
        text = text.replace('\u200b', '')   # Zero-width space
        
        return text.strip()
    
    def _generate_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def get_scraper_info(self) -> Dict[str, Any]:
        """Get information about this scraper"""
        return {
            'name': self.__class__.__name__,
            'config': self.scraping_config,
            'user_agent': self.user_agent,
            'timeout': self.timeout,
            'max_content_size': self.max_content_size
        }