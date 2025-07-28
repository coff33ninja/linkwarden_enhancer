"""URL processing utilities"""

import re
from urllib.parse import urlparse, urljoin
from typing import Optional, List

from .logging_utils import get_logger

logger = get_logger(__name__)


class UrlUtils:
    """Utilities for URL processing and analysis"""
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception as e:
            logger.warning(f"Failed to extract domain from {url}: {e}")
            return ""
    
    @staticmethod
    def extract_path_segments(url: str) -> List[str]:
        """Extract meaningful path segments from URL"""
        try:
            parsed = urlparse(url)
            path = parsed.path
            segments = [seg for seg in path.split('/') if seg and len(seg) > 2]
            return segments
        except Exception as e:
            logger.warning(f"Failed to extract path segments from {url}: {e}")
            return []
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for comparison"""
        try:
            # Remove trailing slash
            url = url.rstrip('/')
            
            # Convert to lowercase (except path)
            parsed = urlparse(url)
            normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}"
            
            if parsed.query:
                normalized += f"?{parsed.query}"
            
            return normalized
        except Exception as e:
            logger.warning(f"Failed to normalize URL {url}: {e}")
            return url
    
    @staticmethod
    def get_base_url(url: str) -> str:
        """Get base URL (scheme + netloc)"""
        try:
            parsed = urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except Exception as e:
            logger.warning(f"Failed to get base URL from {url}: {e}")
            return ""
    
    @staticmethod
    def resolve_relative_url(base_url: str, relative_url: str) -> str:
        """Resolve relative URL against base URL"""
        try:
            return urljoin(base_url, relative_url)
        except Exception as e:
            logger.warning(f"Failed to resolve relative URL {relative_url} against {base_url}: {e}")
            return relative_url
    
    @staticmethod
    def extract_url_keywords(url: str) -> List[str]:
        """Extract keywords from URL for analysis"""
        keywords = []
        
        try:
            # Extract from domain
            domain = UrlUtils.extract_domain(url)
            if domain:
                # Split domain parts
                domain_parts = domain.replace('.', ' ').replace('-', ' ').split()
                keywords.extend([part for part in domain_parts if len(part) > 2])
            
            # Extract from path
            path_segments = UrlUtils.extract_path_segments(url)
            for segment in path_segments:
                # Split on common separators
                segment_words = re.split(r'[-_.]', segment)
                keywords.extend([word for word in segment_words if len(word) > 2])
            
            # Remove duplicates and common words
            common_words = {'com', 'org', 'net', 'www', 'http', 'https', 'html', 'php', 'asp'}
            keywords = list(set([kw.lower() for kw in keywords if kw.lower() not in common_words]))
            
        except Exception as e:
            logger.warning(f"Failed to extract keywords from URL {url}: {e}")
        
        return keywords
    
    @staticmethod
    def classify_url_type(url: str) -> str:
        """Classify URL type based on patterns"""
        try:
            domain = UrlUtils.extract_domain(url)
            path_segments = UrlUtils.extract_path_segments(url)
            
            # GitHub patterns
            if 'github.com' in domain:
                if len(path_segments) >= 2:
                    return 'github_repository'
                return 'github_profile'
            
            # Documentation patterns
            if any(keyword in url.lower() for keyword in ['docs', 'documentation', 'wiki', 'readme']):
                return 'documentation'
            
            # API patterns
            if any(keyword in url.lower() for keyword in ['api', 'swagger', 'openapi']):
                return 'api'
            
            # News/Blog patterns
            if any(keyword in url.lower() for keyword in ['blog', 'news', 'article', 'post']):
                return 'article'
            
            # Tool/Software patterns
            if any(keyword in url.lower() for keyword in ['tool', 'app', 'software', 'download']):
                return 'tool'
            
            return 'general'
            
        except Exception as e:
            logger.warning(f"Failed to classify URL type for {url}: {e}")
            return 'unknown'