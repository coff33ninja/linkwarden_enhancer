"""BeautifulSoup-based scraper for basic HTML parsing"""

import time
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

from enhancement.base_scraper import WebScraper, ScrapingResult
from utils.logging_utils import get_logger
from utils.text_utils import TextUtils

logger = get_logger(__name__)


class BeautifulSoupScraper(WebScraper):
    """BeautifulSoup-based scraper for basic HTML parsing"""
    
    def __init__(self, config: Dict[str, Any]):
        if not BEAUTIFULSOUP_AVAILABLE:
            raise ImportError("BeautifulSoup4 is required. Install with: pip install beautifulsoup4 lxml")
        
        super().__init__(config)
        
        # Parser preference
        self.parser = self.scraping_config.get('parser', 'lxml')
        
        # Content extraction settings
        self.extract_keywords = self.scraping_config.get('extract_keywords', True)
        self.max_keywords = self.scraping_config.get('max_keywords', 10)
        self.min_content_length = self.scraping_config.get('min_content_length', 100)
        
        logger.info(f"BeautifulSoup scraper initialized with {self.parser} parser")
    
    def can_scrape(self, url: str) -> bool:
        """Check if this scraper can handle the given URL"""
        try:
            parsed = urlparse(url)
            
            # Skip non-HTTP(S) URLs
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Skip known binary file extensions
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
            
            # This scraper can handle most HTML content
            return True
            
        except Exception as e:
            logger.debug(f"Error checking if can scrape {url}: {e}")
            return False
    
    def scrape(self, url: str) -> ScrapingResult:
        """Scrape the given URL and return results"""
        start_time = time.time()
        
        try:
            logger.debug(f"Scraping {url} with BeautifulSoup")
            
            # Make HTTP request
            response = self._make_request(url)
            if not response:
                return ScrapingResult(
                    url=url,
                    success=False,
                    error_message="Failed to fetch URL",
                    scraping_time=time.time() - start_time,
                    scraper_used="BeautifulSoup"
                )
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                return ScrapingResult(
                    url=url,
                    success=False,
                    error_message=f"Unsupported content type: {content_type}",
                    scraping_time=time.time() - start_time,
                    scraper_used="BeautifulSoup"
                )
            
            # Parse HTML
            soup = BeautifulSoup(response.content, self.parser)
            
            # Extract basic information
            title = self._extract_title(soup)
            description = self._extract_description(soup)
            keywords = self._extract_keywords(soup, response.text) if self.extract_keywords else []
            favicon_url = self._extract_favicon(url, soup)
            image_url = self._extract_image(soup, url)
            author = self._extract_author(soup)
            published_date = self._extract_published_date(soup)
            language = self._extract_language(soup)
            
            # Extract metadata
            metadata = self._extract_meta_tags(soup)
            metadata.update({
                'content_type': content_type,
                'response_status': response.status_code,
                'response_headers': dict(response.headers),
                'page_size': len(response.content),
                'parser_used': self.parser
            })
            
            # Determine content type category
            content_category = self._classify_content_type(soup, url)
            
            scraping_time = time.time() - start_time
            
            result = ScrapingResult(
                url=url,
                success=True,
                title=title,
                description=description,
                keywords=keywords,
                favicon_url=favicon_url,
                image_url=image_url,
                author=author,
                published_date=published_date,
                content_type=content_category,
                language=language,
                metadata=metadata,
                scraping_time=scraping_time,
                scraper_used="BeautifulSoup"
            )
            
            logger.debug(f"Successfully scraped {url} in {scraping_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Scraping failed: {e}"
            logger.error(f"Error scraping {url}: {error_msg}")
            
            return ScrapingResult(
                url=url,
                success=False,
                error_message=error_msg,
                scraping_time=time.time() - start_time,
                scraper_used="BeautifulSoup"
            )
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title"""
        try:
            # Try Open Graph title first
            og_title = soup.select_one('meta[property="og:title"]')
            if og_title and og_title.get('content'):
                return self._clean_text(og_title['content'])
            
            # Try Twitter title
            twitter_title = soup.select_one('meta[name="twitter:title"]')
            if twitter_title and twitter_title.get('content'):
                return self._clean_text(twitter_title['content'])
            
            # Try standard title tag
            title_tag = soup.select_one('title')
            if title_tag and title_tag.get_text():
                return self._clean_text(title_tag.get_text())
            
            # Try h1 as fallback
            h1_tag = soup.select_one('h1')
            if h1_tag and h1_tag.get_text():
                return self._clean_text(h1_tag.get_text())
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract title: {e}")
            return None
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page description"""
        try:
            # Try Open Graph description first
            og_desc = soup.select_one('meta[property="og:description"]')
            if og_desc and og_desc.get('content'):
                return self._clean_text(og_desc['content'])
            
            # Try Twitter description
            twitter_desc = soup.select_one('meta[name="twitter:description"]')
            if twitter_desc and twitter_desc.get('content'):
                return self._clean_text(twitter_desc['content'])
            
            # Try standard meta description
            meta_desc = soup.select_one('meta[name="description"]')
            if meta_desc and meta_desc.get('content'):
                return self._clean_text(meta_desc['content'])
            
            # Try to extract from first paragraph
            first_p = soup.select_one('p')
            if first_p and first_p.get_text():
                text = self._clean_text(first_p.get_text())
                if len(text) > self.min_content_length:
                    return text[:500] + "..." if len(text) > 500 else text
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract description: {e}")
            return None
    
    def _extract_keywords(self, soup: BeautifulSoup, html_content: str) -> List[str]:
        """Extract keywords from page content"""
        try:
            # Try meta keywords first
            meta_keywords = soup.select_one('meta[name="keywords"]')
            if meta_keywords and meta_keywords.get('content'):
                keywords = [kw.strip() for kw in meta_keywords['content'].split(',')]
                return keywords[:self.max_keywords]
            
            # Extract from page content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text_content = soup.get_text()
            
            # Use TextUtils to extract keywords
            keywords = TextUtils.extract_keywords(text_content, max_keywords=self.max_keywords)
            return keywords
            
        except Exception as e:
            logger.debug(f"Failed to extract keywords: {e}")
            return []
    
    def _extract_image(self, soup: BeautifulSoup, base_url: str) -> Optional[str]:
        """Extract main image URL"""
        try:
            # Try Open Graph image first
            og_image = soup.select_one('meta[property="og:image"]')
            if og_image and og_image.get('content'):
                return urljoin(base_url, og_image['content'])
            
            # Try Twitter image
            twitter_image = soup.select_one('meta[name="twitter:image"]')
            if twitter_image and twitter_image.get('content'):
                return urljoin(base_url, twitter_image['content'])
            
            # Try to find first significant image
            images = soup.select('img[src]')
            for img in images:
                src = img.get('src')
                if src and not any(skip in src.lower() for skip in ['logo', 'icon', 'avatar', 'button']):
                    # Check if image seems significant (has alt text or is large)
                    alt = img.get('alt', '')
                    width = img.get('width')
                    height = img.get('height')
                    
                    if alt or (width and height and int(width) > 200 and int(height) > 200):
                        return urljoin(base_url, src)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract image: {e}")
            return None
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author information"""
        try:
            # Try various author meta tags
            author_selectors = [
                'meta[name="author"]',
                'meta[property="article:author"]',
                'meta[name="twitter:creator"]',
                '[rel="author"]',
                '.author',
                '.byline'
            ]
            
            for selector in author_selectors:
                element = soup.select_one(selector)
                if element:
                    if element.name == 'meta':
                        content = element.get('content')
                        if content:
                            return self._clean_text(content)
                    else:
                        text = element.get_text()
                        if text:
                            return self._clean_text(text)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract author: {e}")
            return None
    
    def _extract_published_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date"""
        try:
            # Try various date meta tags
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="date"]',
                'meta[name="pubdate"]',
                'meta[property="og:updated_time"]',
                'time[datetime]',
                '.date',
                '.published'
            ]
            
            for selector in date_selectors:
                element = soup.select_one(selector)
                if element:
                    if element.name == 'meta':
                        content = element.get('content')
                        if content:
                            return content
                    elif element.name == 'time':
                        datetime_attr = element.get('datetime')
                        if datetime_attr:
                            return datetime_attr
                    else:
                        text = element.get_text()
                        if text:
                            return self._clean_text(text)
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract published date: {e}")
            return None
    
    def _extract_language(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page language"""
        try:
            # Try html lang attribute
            html_tag = soup.select_one('html[lang]')
            if html_tag:
                return html_tag.get('lang')
            
            # Try meta language tag
            meta_lang = soup.select_one('meta[http-equiv="content-language"]')
            if meta_lang and meta_lang.get('content'):
                return meta_lang['content']
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract language: {e}")
            return None
    
    def _classify_content_type(self, soup: BeautifulSoup, url: str) -> str:
        """Classify the type of content"""
        try:
            url_lower = url.lower()
            
            # Check URL patterns
            if any(pattern in url_lower for pattern in ['blog', 'article', 'post']):
                return 'article'
            elif any(pattern in url_lower for pattern in ['docs', 'documentation', 'api']):
                return 'documentation'
            elif any(pattern in url_lower for pattern in ['github.com', 'gitlab.com']):
                return 'code_repository'
            elif any(pattern in url_lower for pattern in ['youtube.com', 'vimeo.com']):
                return 'video'
            
            # Check page structure
            if soup.select('article') or soup.select('.article'):
                return 'article'
            elif soup.select('pre code') or soup.select('.highlight'):
                return 'code'
            elif soup.select('video') or soup.select('.video'):
                return 'video'
            
            # Check meta tags
            og_type = soup.select_one('meta[property="og:type"]')
            if og_type and og_type.get('content'):
                return og_type['content']
            
            return 'webpage'
            
        except Exception as e:
            logger.debug(f"Failed to classify content type: {e}")
            return 'webpage'    
    def _extract_favicon(self, base_url: str, soup: BeautifulSoup) -> Optional[str]:
        """Extract favicon URL"""
        try:
            # Try various favicon link tags
            favicon_selectors = [
                'link[rel="icon"]',
                'link[rel="shortcut icon"]',
                'link[rel="apple-touch-icon"]',
                'link[rel="apple-touch-icon-precomposed"]'
            ]
            
            for selector in favicon_selectors:
                favicon_link = soup.select_one(selector)
                if favicon_link and favicon_link.get('href'):
                    return urljoin(base_url, favicon_link['href'])
            
            # Try default favicon.ico
            parsed = urlparse(base_url)
            default_favicon = f"{parsed.scheme}://{parsed.netloc}/favicon.ico"
            return default_favicon
            
        except Exception as e:
            logger.debug(f"Failed to extract favicon: {e}")
            return None
    
    def _extract_meta_tags(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract all meta tags"""
        metadata = {}
        
        try:
            # Extract all meta tags
            meta_tags = soup.select('meta')
            
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
                content = meta.get('content')
                
                if name and content:
                    metadata[f'meta_{name}'] = content
            
            # Extract structured data (JSON-LD)
            json_ld_scripts = soup.select('script[type="application/ld+json"]')
            if json_ld_scripts:
                structured_data = []
                for script in json_ld_scripts:
                    try:
                        import json
                        data = json.loads(script.get_text())
                        structured_data.append(data)
                    except:
                        continue
                
                if structured_data:
                    metadata['structured_data'] = structured_data
            
            # Extract microdata
            microdata_elements = soup.select('[itemscope]')
            if microdata_elements:
                microdata = []
                for element in microdata_elements[:5]:  # Limit to first 5
                    item_type = element.get('itemtype')
                    if item_type:
                        microdata.append({
                            'type': item_type,
                            'properties': self._extract_microdata_properties(element)
                        })
                
                if microdata:
                    metadata['microdata'] = microdata
            
        except Exception as e:
            logger.debug(f"Failed to extract meta tags: {e}")
        
        return metadata
    
    def _extract_microdata_properties(self, element) -> Dict[str, str]:
        """Extract microdata properties from element"""
        properties = {}
        
        try:
            prop_elements = element.select('[itemprop]')
            for prop_element in prop_elements:
                prop_name = prop_element.get('itemprop')
                prop_value = (
                    prop_element.get('content') or 
                    prop_element.get('datetime') or 
                    prop_element.get_text().strip()
                )
                
                if prop_name and prop_value:
                    properties[prop_name] = prop_value
        
        except Exception as e:
            logger.debug(f"Failed to extract microdata properties: {e}")
        
        return properties
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = ' '.join(text.split())
        
        # Remove common unwanted characters
        cleaned = cleaned.replace('\u00a0', ' ')  # Non-breaking space
        cleaned = cleaned.replace('\u200b', '')   # Zero-width space
        
        return cleaned.strip()
    
    def get_scraper_info(self) -> Dict[str, Any]:
        """Get scraper information"""
        return {
            'name': 'BeautifulSoupScraper',
            'description': 'BeautifulSoup-based scraper for basic HTML parsing',
            'capabilities': [
                'HTML parsing',
                'Meta tag extraction',
                'Open Graph data',
                'Twitter Card data',
                'Structured data (JSON-LD)',
                'Microdata extraction',
                'Favicon detection',
                'Content classification'
            ],
            'parser': self.parser,
            'beautifulsoup_available': BEAUTIFULSOUP_AVAILABLE,
            'extract_keywords': self.extract_keywords,
            'max_keywords': self.max_keywords
        }