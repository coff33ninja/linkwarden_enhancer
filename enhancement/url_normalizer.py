"""URL normalization engine for duplicate detection"""

import re
import asyncio
import aiohttp
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from typing import Optional, Dict, Set, List, Tuple
from dataclasses import dataclass

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class NormalizedURL:
    """Normalized URL with metadata"""
    original_url: str
    normalized_url: str
    canonical_url: Optional[str] = None
    redirect_chain: List[str] = None
    domain_aliases: Set[str] = None
    parameters_removed: Dict[str, str] = None
    
    def __post_init__(self):
        if self.redirect_chain is None:
            self.redirect_chain = []
        if self.domain_aliases is None:
            self.domain_aliases = set()
        if self.parameters_removed is None:
            self.parameters_removed = {}


class URLNormalizer:
    """Advanced URL normalization for consistent comparison"""
    
    def __init__(self, timeout: int = 10, max_redirects: int = 5):
        self.timeout = timeout
        self.max_redirects = max_redirects
        
        # Common tracking parameters to remove
        self.tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'ref', 'referrer', 'source',
            '_ga', '_gid', 'mc_cid', 'mc_eid', 'campaign_id', 'ad_id',
            'click_id', 'affiliate_id', 'partner_id', 'tracking_id'
        }
        
        # Domain aliases (www vs non-www, different subdomains)
        self.domain_aliases = {
            'github.com': {'www.github.com', 'github.com'},
            'stackoverflow.com': {'www.stackoverflow.com', 'stackoverflow.com'},
            'reddit.com': {'www.reddit.com', 'old.reddit.com', 'new.reddit.com', 'reddit.com'},
            'youtube.com': {'www.youtube.com', 'm.youtube.com', 'youtube.com'},
            'twitter.com': {'www.twitter.com', 'mobile.twitter.com', 'twitter.com', 'x.com'},
            'medium.com': {'www.medium.com', 'medium.com'},
        }
        
        # URL patterns that should be considered equivalent
        self.equivalent_patterns = [
            # GitHub patterns
            (r'github\.com/([^/]+)/([^/]+)/?$', r'github.com/\1/\2'),
            (r'github\.com/([^/]+)/([^/]+)/tree/main/?$', r'github.com/\1/\2'),
            (r'github\.com/([^/]+)/([^/]+)/tree/master/?$', r'github.com/\1/\2'),
            # YouTube patterns - handle youtu.be conversion
            (r'youtu\.be/([^?]+)', r'youtube.com/watch?v=\1'),
            # Reddit patterns
            (r'reddit\.com/r/([^/]+)/?$', r'reddit.com/r/\1'),
            (r'reddit\.com/r/([^/]+)/comments/([^/]+)/[^/]*/?$', r'reddit.com/r/\1/comments/\2'),
        ]
    
    def normalize_url(self, url: str) -> NormalizedURL:
        """Normalize URL for consistent comparison"""
        try:
            original_url = url.strip()
            
            # Parse URL
            parsed = urlparse(original_url)
            
            # Normalize scheme
            scheme = parsed.scheme.lower() if parsed.scheme else 'https'
            
            # Normalize domain
            domain = self._normalize_domain(parsed.netloc)
            
            # Normalize path
            path = self._normalize_path(parsed.path)
            
            # Filter and normalize query parameters
            query_params, removed_params = self._normalize_query_parameters(parsed.query)
            
            # Remove fragment
            fragment = ''
            
            # Reconstruct normalized URL
            normalized_parsed = (scheme, domain, path, parsed.params, query_params, fragment)
            normalized_url = urlunparse(normalized_parsed)
            
            # Handle trailing slash removal for domain-only URLs
            if path in ('', '/') and not query_params and not parsed.params:
                # Remove trailing slash for domain-only URLs
                normalized_url = normalized_url.rstrip('/')
            
            # Apply pattern-based normalization
            normalized_url = self._apply_pattern_normalization(normalized_url)
            
            # Get domain aliases
            aliases = self._get_domain_aliases(domain)
            
            return NormalizedURL(
                original_url=original_url,
                normalized_url=normalized_url,
                domain_aliases=aliases,
                parameters_removed=removed_params
            )
            
        except Exception as e:
            logger.warning(f"Failed to normalize URL {url}: {e}")
            return NormalizedURL(
                original_url=url,
                normalized_url=url,
                domain_aliases=set(),
                parameters_removed={}
            )
    
    def _normalize_domain(self, netloc: str) -> str:
        """Normalize domain name"""
        if not netloc:
            return ''
        
        domain = netloc.lower()
        
        # Remove port if it's default
        if ':80' in domain and not domain.startswith('https'):
            domain = domain.replace(':80', '')
        elif ':443' in domain and domain.startswith('https'):
            domain = domain.replace(':443', '')
        
        # Handle www prefix consistently
        if domain.startswith('www.'):
            base_domain = domain[4:]
            # Check if this domain should keep www or not
            if base_domain in ['github.com', 'stackoverflow.com']:
                return base_domain  # Remove www for these domains
            else:
                return domain  # Keep www for others
        
        return domain
    
    def _normalize_path(self, path: str) -> str:
        """Normalize URL path"""
        if not path:
            return ''
        
        # For root path, keep it as is
        if path == '/':
            return '/'
        
        # Remove trailing slash for non-root paths
        normalized_path = path.rstrip('/')
        
        # Decode percent-encoded characters for common cases
        normalized_path = normalized_path.replace('%20', ' ')
        
        # Remove duplicate slashes
        normalized_path = re.sub(r'/+', '/', normalized_path)
        
        return normalized_path
    
    def _normalize_query_parameters(self, query: str) -> Tuple[str, Dict[str, str]]:
        """Normalize query parameters, removing tracking params"""
        if not query:
            return '', {}
        
        params = parse_qs(query, keep_blank_values=True)
        removed_params = {}
        filtered_params = {}
        
        for key, values in params.items():
            if key.lower() in self.tracking_params:
                removed_params[key] = values[0] if values else ''
            else:
                filtered_params[key] = values
        
        # Sort parameters for consistent ordering
        if filtered_params:
            # Convert back to single values for sorting
            sorted_items = []
            for key, values in sorted(filtered_params.items()):
                if isinstance(values, list) and len(values) == 1:
                    sorted_items.append((key, values[0]))
                else:
                    sorted_items.append((key, values))
            
            normalized_query = urlencode(sorted_items, doseq=True)
            return normalized_query, removed_params
        
        return '', removed_params
    
    def _apply_pattern_normalization(self, url: str) -> str:
        """Apply pattern-based URL normalization"""
        for pattern, replacement in self.equivalent_patterns:
            url = re.sub(pattern, replacement, url, flags=re.IGNORECASE)
        
        return url
    
    def _get_domain_aliases(self, domain: str) -> Set[str]:
        """Get known aliases for a domain"""
        for base_domain, aliases in self.domain_aliases.items():
            if domain in aliases or domain == base_domain:
                return aliases.copy()
        
        # Generate common aliases for unknown domains
        aliases = {domain}
        if domain.startswith('www.'):
            aliases.add(domain[4:])
        else:
            aliases.add(f'www.{domain}')
        
        return aliases
    
    async def resolve_redirects(self, url: str) -> NormalizedURL:
        """Resolve redirects and get canonical URL"""
        normalized = self.normalize_url(url)
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(limit=10)
            ) as session:
                redirect_chain = []
                current_url = normalized.normalized_url
                
                for _ in range(self.max_redirects):
                    try:
                        async with session.head(
                            current_url,
                            allow_redirects=False,
                            headers={'User-Agent': 'LinkwardenEnhancer/1.0'}
                        ) as response:
                            if response.status in (301, 302, 303, 307, 308):
                                redirect_url = response.headers.get('Location')
                                if redirect_url:
                                    if not redirect_url.startswith('http'):
                                        # Handle relative redirects
                                        from urllib.parse import urljoin
                                        redirect_url = urljoin(current_url, redirect_url)
                                    
                                    redirect_chain.append(current_url)
                                    current_url = redirect_url
                                    continue
                            
                            # No more redirects
                            break
                            
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout resolving redirects for {current_url}")
                        break
                    except Exception as e:
                        logger.warning(f"Error resolving redirects for {current_url}: {e}")
                        break
                
                # Try to get canonical URL from final page
                canonical_url = await self._get_canonical_url(session, current_url)
                
                normalized.redirect_chain = redirect_chain
                normalized.canonical_url = canonical_url or current_url
                
        except Exception as e:
            logger.warning(f"Failed to resolve redirects for {url}: {e}")
        
        return normalized
    
    async def _get_canonical_url(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Extract canonical URL from page head"""
        try:
            async with session.get(
                url,
                headers={'User-Agent': 'LinkwardenEnhancer/1.0'}
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Look for canonical link
                    canonical_match = re.search(
                        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
                        content,
                        re.IGNORECASE
                    )
                    
                    if canonical_match:
                        canonical_url = canonical_match.group(1)
                        if not canonical_url.startswith('http'):
                            from urllib.parse import urljoin
                            canonical_url = urljoin(url, canonical_url)
                        return canonical_url
                        
        except Exception as e:
            logger.debug(f"Could not extract canonical URL from {url}: {e}")
        
        return None
    
    def are_urls_equivalent(self, url1: str, url2: str) -> bool:
        """Check if two URLs are equivalent after normalization"""
        try:
            norm1 = self.normalize_url(url1)
            norm2 = self.normalize_url(url2)
            
            # Direct comparison
            if norm1.normalized_url == norm2.normalized_url:
                return True
            
            # Check domain aliases
            domain1 = urlparse(norm1.normalized_url).netloc
            domain2 = urlparse(norm2.normalized_url).netloc
            
            if domain1 in norm2.domain_aliases and domain2 in norm1.domain_aliases:
                # Same path with different domain aliases
                path1 = urlparse(norm1.normalized_url).path
                path2 = urlparse(norm2.normalized_url).path
                query1 = urlparse(norm1.normalized_url).query
                query2 = urlparse(norm2.normalized_url).query
                
                return path1 == path2 and query1 == query2
            
            return False
            
        except Exception as e:
            logger.warning(f"Error comparing URLs {url1} and {url2}: {e}")
            return False
    
    async def batch_normalize_urls(self, urls: List[str], resolve_redirects: bool = False) -> List[NormalizedURL]:
        """Normalize multiple URLs efficiently"""
        if resolve_redirects:
            tasks = [self.resolve_redirects(url) for url in urls]
            return await asyncio.gather(*tasks, return_exceptions=True)
        else:
            return [self.normalize_url(url) for url in urls]
    
    def get_url_similarity_score(self, url1: str, url2: str) -> float:
        """Calculate similarity score between two URLs (0.0 to 1.0)"""
        try:
            norm1 = self.normalize_url(url1)
            norm2 = self.normalize_url(url2)
            
            # Exact match
            if norm1.normalized_url == norm2.normalized_url:
                return 1.0
            
            parsed1 = urlparse(norm1.normalized_url)
            parsed2 = urlparse(norm2.normalized_url)
            
            # Domain similarity
            domain_score = 0.0
            if parsed1.netloc == parsed2.netloc:
                domain_score = 1.0
            elif parsed1.netloc in norm2.domain_aliases or parsed2.netloc in norm1.domain_aliases:
                domain_score = 0.9
            else:
                # Calculate domain similarity based on common parts
                domain1_parts = parsed1.netloc.split('.')
                domain2_parts = parsed2.netloc.split('.')
                
                # Only consider domains similar if they share the main domain name
                # (not just TLD like .com)
                if len(domain1_parts) >= 2 and len(domain2_parts) >= 2:
                    # Compare the main domain part (second-to-last part)
                    main1 = domain1_parts[-2] if len(domain1_parts) >= 2 else ''
                    main2 = domain2_parts[-2] if len(domain2_parts) >= 2 else ''
                    
                    if main1 == main2 and main1:  # Same main domain
                        common_parts = set(domain1_parts) & set(domain2_parts)
                        domain_score = len(common_parts) / max(len(domain1_parts), len(domain2_parts))
                    else:
                        domain_score = 0.0
                else:
                    domain_score = 0.0
            
            # Path similarity
            path_score = 0.0
            if parsed1.path == parsed2.path:
                # Only give full path score if domains are similar too
                if domain_score > 0.5:
                    path_score = 1.0
                else:
                    # Same path but different domains shouldn't get high path score
                    path_score = 0.3 if parsed1.path not in ('', '/') else 0.1
            else:
                # Calculate path similarity based on common segments
                path1_segments = [s for s in parsed1.path.split('/') if s]
                path2_segments = [s for s in parsed2.path.split('/') if s]
                
                if path1_segments and path2_segments:
                    common_segments = 0
                    min_length = min(len(path1_segments), len(path2_segments))
                    
                    for i in range(min_length):
                        if path1_segments[i] == path2_segments[i]:
                            common_segments += 1
                        else:
                            break
                    
                    path_score = common_segments / max(len(path1_segments), len(path2_segments))
            
            # Query similarity
            query_score = 0.0
            if parsed1.query == parsed2.query:
                query_score = 1.0
            elif parsed1.query and parsed2.query:
                params1 = set(parse_qs(parsed1.query).keys())
                params2 = set(parse_qs(parsed2.query).keys())
                common_params = params1 & params2
                if common_params:
                    query_score = len(common_params) / len(params1 | params2)
            
            # Weighted final score
            final_score = (domain_score * 0.5 + path_score * 0.4 + query_score * 0.1)
            return min(1.0, final_score)
            
        except Exception as e:
            logger.warning(f"Error calculating URL similarity for {url1} and {url2}: {e}")
            return 0.0