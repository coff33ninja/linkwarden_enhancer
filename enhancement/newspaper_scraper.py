"""Newspaper3k-based scraper for article content extraction"""

import time
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, urljoin
from datetime import datetime

try:
    from newspaper import Article, Config
    from newspaper.utils import BeautifulSoup
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

from .base_scraper import WebScraper, ScrapingResult
from ..utils.logging_utils import get_logger
from ..utils.text_utils import TextUtils

logger = get_logger(__name__)


class NewspaperScraper(WebScraper):
    """Newspaper3k-based scraper for article content extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Newspaper scraper"""
        super().__init__(config)
        
        if not NEWSPAPER_AVAILABLE:
            raise ImportError("newspaper3k is required for NewspaperScraper. Install with: pip install newspaper3k")
        
        self.newspaper_config = config.get('newspaper', {})
        
        # Newspaper3k configuration
        self.language = self.newspaper_config.get('language', 'en')
        self.memoize_articles = self.newspaper_config.get('memoize_articles', True)
        self.fetch_images = self.newspaper_config.get('fetch_images', True)
        self.follow_meta_refresh = self.newspaper_config.get('follow_meta_refresh', True)
        self.request_timeout = self.newspaper_config.get('request_timeout', 30)
        self.number_threads = self.newspaper_config.get('number_threads', 1)
        
        # Content processing settings
        self.max_summary_length = self.newspaper_config.get('max_summary_length', 500)
        self.extract_keywords_count = self.newspaper_config.get('extract_keywords_count', 10)
        self.min_article_length = self.newspaper_config.get('min_article_length', 200)
        
        # Create newspaper configuration
        self.config_obj = Config()
        self.config_obj.language = self.language
        self.config_obj.memoize_articles = self.memoize_articles
        self.config_obj.fetch_images = self.fetch_images
        self.config_obj.follow_meta_refresh = self.follow_meta_refresh
        self.config_obj.request_timeout = self.request_timeout
        self.config_obj.number_threads = self.number_threads
        
        # User agent
        self.config_obj.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        logger.info(f"Newspaper scraper initialized for language: {self.language}")
    
    def can_scrape(self, url: str) -> bool:
        """Check if this scraper can handle the URL"""
        
        if not NEWSPAPER_AVAILABLE:
            return False
        
        try:
            parsed = urlparse(url)
            
            # Only handle HTTP(S)
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Check for news/article domains and patterns
            article_domains = {
                # News sites
                'bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org',
                'guardian.com', 'nytimes.com', 'washingtonpost.com',
                'bloomberg.com', 'wsj.com', 'ft.com',
                
                # Tech news
                'techcrunch.com', 'arstechnica.com', 'theverge.com', 'wired.com',
                'engadget.com', 'gizmodo.com', 'mashable.com', 'recode.net',
                
                # Blogs and articles
                'medium.com', 'substack.com', 'dev.to', 'hashnode.com',
                'towards.ai', 'machinelearningmastery.com', 'distill.pub',
                
                # Documentation that might be article-like
                'blog.', 'news.', 'article.', 'post.'
            }
            
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # Check domain patterns
            if any(article_domain in domain for article_domain in article_domains):
                return True
            
            # Check URL path patterns that suggest articles
            article_patterns = [
                '/article/', '/post/', '/blog/', '/news/', '/story/',
                '/articles/', '/posts/', '/blogs/', '/stories/',
                '/2020/', '/2021/', '/2022/', '/2023/', '/2024/', '/2025/',  # Date patterns
                '/how-to/', '/tutorial/', '/guide/'
            ]
            
            if any(pattern in path for pattern in article_patterns):
                return True
            
            # Check for common article URL structures
            # e.g., /year/month/day/title or /category/title
            path_segments = [seg for seg in path.split('/') if seg]
            if len(path_segments) >= 2:
                # Check for date-based URLs
                try:
                    if len(path_segments) >= 3:
                        year = int(path_segments[0])
                        month = int(path_segments[1])
                        if 2000 <= year <= 2030 and 1 <= month <= 12:
                            return True
                except ValueError:
                    pass
                
                # Check for category/title structure
                if len(path_segments) >= 2 and len(path_segments[-1]) > 10:
                    # Last segment is likely a title if it's long enough
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if can scrape {url}: {e}")
            return False
    
    def scrape(self, url: str) -> ScrapingResult:
        """Scrape URL using Newspaper3k"""
        
        start_time = time.time()
        
        try:
            logger.debug(f"Starting Newspaper scraping for {url}")
            
            # Create article object
            article = Article(url, config=self.config_obj)
            
            # Download and parse article
            article.download()
            article.parse()
            
            # Extract natural language processing features
            try:
                article.nlp()
            except Exception as e:
                logger.debug(f"NLP processing failed for {url}: {e}")
                # Continue without NLP features
            
            # Extract metadata
            title = self._extract_title(article)
            description = self._extract_description(article)
            keywords = self._extract_keywords(article)
            content = self._extract_content(article)
            author = self._extract_author(article)
            published_date = self._extract_published_date(article)
            image_url = self._extract_image_url(article, url)
            
            # Generate summary if content is long enough
            summary = None
            if content and len(content) > self.min_article_length:
                summary = self._generate_summary(content)
            
            # Extract additional metadata
            metadata = self._extract_additional_metadata(article, content)
            if summary:
                metadata['article_summary'] = summary
            
            scraping_time = time.time() - start_time
            
            result = ScrapingResult(
                url=url,
                title=title,
                description=description or summary,  # Use summary as fallback description
                keywords=keywords,
                favicon_url=None,  # Newspaper3k doesn't extract favicons
                image_url=image_url,
                author=author,
                published_date=published_date,
                content_type='article',
                language=self.language,
                success=True,
                scraping_time=scraping_time,
                scraper_used="NewspaperScraper",
                metadata=metadata
            )
            
            logger.debug(f"Newspaper scraping completed for {url} in {scraping_time:.2f}s")
            return result
            
        except Exception as e:
            scraping_time = time.time() - start_time
            error_msg = f"Newspaper scraping failed: {e}"
            logger.error(f"Error scraping {url}: {error_msg}")
            
            return ScrapingResult(
                url=url,
                success=False,
                error_message=error_msg,
                scraping_time=scraping_time,
                scraper_used="NewspaperScraper"
            )
    
    def _extract_title(self, article: Article) -> Optional[str]:
        """Extract article title"""
        try:
            title = article.title
            if title and title.strip():
                return title.strip()
            return None
        except Exception as e:
            logger.debug(f"Error extracting title: {e}")
            return None
    
    def _extract_description(self, article: Article) -> Optional[str]:
        """Extract article description/meta description"""
        try:
            # Try meta description first
            if hasattr(article, 'meta_description') and article.meta_description:
                desc = article.meta_description.strip()
                if desc:
                    return desc
            
            # Fallback to first paragraph of article text
            if article.text:
                paragraphs = article.text.split('\n\n')
                for paragraph in paragraphs[:3]:  # Check first 3 paragraphs
                    paragraph = paragraph.strip()
                    if paragraph and len(paragraph) > 50:
                        return paragraph[:500]  # Limit length
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting description: {e}")
            return None
    
    def _extract_keywords(self, article: Article) -> List[str]:
        """Extract keywords from article"""
        try:
            keywords = []
            
            # Use Newspaper3k's keyword extraction
            if hasattr(article, 'keywords') and article.keywords:
                keywords.extend(article.keywords[:self.extract_keywords_count])
            
            # Extract additional keywords from title and text
            if article.title:
                title_keywords = TextUtils.extract_keywords(article.title, max_keywords=3)
                keywords.extend(title_keywords)
            
            if article.text and len(keywords) < self.extract_keywords_count:
                # Extract from first few paragraphs
                text_sample = article.text[:1000]  # First 1000 characters
                text_keywords = TextUtils.extract_keywords(text_sample, max_keywords=5)
                keywords.extend(text_keywords)
            
            # Remove duplicates and return
            return list(dict.fromkeys(keywords))[:self.extract_keywords_count]
            
        except Exception as e:
            logger.debug(f"Error extracting keywords: {e}")
            return []
    
    def _extract_content(self, article: Article) -> Optional[str]:
        """Extract full article content"""
        try:
            if article.text and article.text.strip():
                return article.text.strip()
            return None
        except Exception as e:
            logger.debug(f"Error extracting content: {e}")
            return None
    
    def _extract_author(self, article: Article) -> Optional[str]:
        """Extract article author"""
        try:
            # Try multiple author extraction methods
            authors = []
            
            # Newspaper3k authors
            if hasattr(article, 'authors') and article.authors:
                authors.extend(article.authors)
            
            # Single author
            if hasattr(article, 'author') and article.author:
                authors.append(article.author)
            
            # Clean and return first author
            if authors:
                author = authors[0].strip()
                if author:
                    return author
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting author: {e}")
            return None
    
    def _extract_published_date(self, article: Article) -> Optional[str]:
        """Extract article published date"""
        try:
            if hasattr(article, 'publish_date') and article.publish_date:
                # Convert datetime to ISO string
                if isinstance(article.publish_date, datetime):
                    return article.publish_date.isoformat()
                else:
                    return str(article.publish_date)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting published date: {e}")
            return None
    
    def _extract_image_url(self, article: Article, base_url: str) -> Optional[str]:
        """Extract main article image"""
        try:
            # Try Newspaper3k's top image
            if hasattr(article, 'top_image') and article.top_image:
                image_url = article.top_image
                if image_url:
                    return urljoin(base_url, image_url)
            
            # Try images list
            if hasattr(article, 'images') and article.images:
                for image_url in list(article.images)[:3]:  # Check first 3 images
                    if image_url and not any(skip in image_url.lower() for skip in ['logo', 'icon', 'avatar']):
                        return urljoin(base_url, image_url)
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting image: {e}")
            return None
    
    def _generate_summary(self, content: str) -> Optional[str]:
        """Generate article summary"""
        try:
            if not content or len(content) < self.min_article_length:
                return None
            
            # Simple extractive summarization
            sentences = content.split('. ')
            
            if len(sentences) <= 3:
                return content[:self.max_summary_length]
            
            # Take first few sentences as summary
            summary_sentences = sentences[:3]
            summary = '. '.join(summary_sentences)
            
            # Ensure it ends with a period
            if not summary.endswith('.'):
                summary += '.'
            
            # Limit length
            if len(summary) > self.max_summary_length:
                summary = summary[:self.max_summary_length - 3] + '...'
            
            return summary
            
        except Exception as e:
            logger.debug(f"Error generating summary: {e}")
            return None
    
    def _extract_additional_metadata(self, article: Article, content: Optional[str]) -> Dict[str, Any]:
        """Extract additional metadata"""
        metadata = {}
        
        try:
            # Article statistics
            if content:
                word_count = len(content.split())
                metadata['word_count'] = word_count
                metadata['estimated_read_time'] = max(1, word_count // 200)  # ~200 words per minute
            
            # Article structure
            if hasattr(article, 'html') and article.html:
                try:
                    soup = BeautifulSoup(article.html, 'html.parser')
                    
                    # Count headings
                    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if headings:
                        metadata['heading_count'] = len(headings)
                    
                    # Count paragraphs
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        metadata['paragraph_count'] = len(paragraphs)
                    
                    # Count links
                    links = soup.find_all('a', href=True)
                    if links:
                        metadata['link_count'] = len(links)
                        
                        # Extract external links
                        external_links = []
                        for link in links[:10]:  # Limit to first 10
                            href = link.get('href', '')
                            if href.startswith('http') and article.url not in href:
                                external_links.append(href)
                        
                        if external_links:
                            metadata['external_links'] = external_links
                
                except Exception as e:
                    logger.debug(f"Error analyzing article structure: {e}")
            
            # Publication metadata
            if hasattr(article, 'meta_data') and article.meta_data:
                meta_data = article.meta_data
                
                # Extract useful meta tags
                useful_meta = {}
                for key, value in meta_data.items():
                    if key in ['og:site_name', 'twitter:site', 'article:section', 'article:tag']:
                        useful_meta[key] = value
                
                if useful_meta:
                    metadata['meta_tags'] = useful_meta
            
            # Content quality indicators
            if content:
                # Check for quotes
                quote_count = content.count('"') + content.count('"') + content.count('"')
                if quote_count > 0:
                    metadata['quote_count'] = quote_count // 2  # Approximate quote pairs
                
                # Check for lists
                list_indicators = content.count('•') + content.count('1.') + content.count('2.')
                if list_indicators > 0:
                    metadata['has_lists'] = True
                
                # Check for code blocks (for tech articles)
                code_indicators = content.count('```') + content.count('`')
                if code_indicators > 0:
                    metadata['has_code'] = True
            
        except Exception as e:
            logger.debug(f"Error extracting additional metadata: {e}")
        
        return metadata
    
    def get_scraper_info(self) -> Dict[str, Any]:
        """Get scraper information"""
        return {
            'name': 'NewspaperScraper',
            'description': 'Newspaper3k-based scraper for article content extraction',
            'capabilities': [
                'Article text extraction',
                'Author detection',
                'Publication date extraction',
                'Keyword extraction',
                'Image extraction',
                'Content summarization',
                'NLP processing'
            ],
            'language': self.language,
            'newspaper_available': NEWSPAPER_AVAILABLE,
            'min_article_length': self.min_article_length,
            'max_summary_length': self.max_summary_length
        }