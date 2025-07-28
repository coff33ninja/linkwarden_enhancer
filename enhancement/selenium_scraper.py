"""Selenium-based scraper for JavaScript-heavy sites and dynamic content"""

import time
import base64
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin, urlparse
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.firefox.service import Service as FirefoxService
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

from enhancement.base_scraper import WebScraper, ScrapingResult
from utils.logging_utils import get_logger
from utils.text_utils import TextUtils

logger = get_logger(__name__)


class SeleniumScraper(WebScraper):
    """Selenium-based scraper for JavaScript-heavy sites and dynamic content"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Selenium scraper"""
        super().__init__(config)
        
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for SeleniumScraper. Install with: pip install selenium")
        
        self.selenium_config = config.get('selenium', {})
        
        # Browser configuration
        self.browser = self.selenium_config.get('browser', 'chrome')  # chrome, firefox
        self.headless = self.selenium_config.get('headless', True)
        self.window_size = self.selenium_config.get('window_size', (1920, 1080))
        self.page_load_timeout = self.selenium_config.get('page_load_timeout', 30)
        self.implicit_wait = self.selenium_config.get('implicit_wait', 10)
        self.wait_for_js = self.selenium_config.get('wait_for_js', 3)
        
        # Screenshot configuration
        self.enable_screenshots = self.selenium_config.get('enable_screenshots', True)
        self.screenshot_dir = Path(config.get('directories', {}).get('screenshots_dir', 'screenshots'))
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Browser options
        self.browser_options = self._get_browser_options()
        
        # Driver management
        self.driver = None
        self.driver_service = None
        
        logger.info(f"Selenium scraper initialized with {self.browser} browser (headless: {self.headless})")
    
    def _get_browser_options(self) -> Any:
        """Get browser-specific options"""
        
        if self.browser.lower() == 'chrome':
            options = ChromeOptions()
            
            if self.headless:
                options.add_argument('--headless')
            
            # Performance and security options
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-plugins')
            options.add_argument('--disable-images')  # Faster loading
            options.add_argument('--disable-javascript-harmony-shipping')
            options.add_argument('--disable-background-timer-throttling')
            options.add_argument('--disable-renderer-backgrounding')
            options.add_argument('--disable-backgrounding-occluded-windows')
            
            # User agent
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Window size
            options.add_argument(f'--window-size={self.window_size[0]},{self.window_size[1]}')
            
            return options
            
        elif self.browser.lower() == 'firefox':
            options = FirefoxOptions()
            
            if self.headless:
                options.add_argument('--headless')
            
            # Performance options
            options.set_preference('dom.webnotifications.enabled', False)
            options.set_preference('media.volume_scale', '0.0')
            options.set_preference('permissions.default.image', 2)  # Block images
            
            return options
        
        else:
            raise ValueError(f"Unsupported browser: {self.browser}")
    
    def _create_driver(self) -> webdriver.Remote:
        """Create and configure WebDriver instance"""
        
        try:
            if self.browser.lower() == 'chrome':
                # Try to use system Chrome first, then ChromeDriver
                try:
                    driver = webdriver.Chrome(options=self.browser_options)
                except Exception as e:
                    logger.debug(f"Failed to use system Chrome: {e}")
                    # Fallback to explicit ChromeDriver path if needed
                    driver = webdriver.Chrome(options=self.browser_options)
                    
            elif self.browser.lower() == 'firefox':
                driver = webdriver.Firefox(options=self.browser_options)
            else:
                raise ValueError(f"Unsupported browser: {self.browser}")
            
            # Configure timeouts
            driver.set_page_load_timeout(self.page_load_timeout)
            driver.implicitly_wait(self.implicit_wait)
            
            logger.debug(f"Created {self.browser} WebDriver instance")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to create WebDriver: {e}")
            raise
    
    def can_scrape(self, url: str) -> bool:
        """Check if this scraper can handle the URL"""
        
        if not SELENIUM_AVAILABLE:
            return False
        
        # Selenium can handle most HTTP(S) URLs, but it's slower
        # Use it for known JavaScript-heavy sites or as fallback
        
        try:
            parsed = urlparse(url)
            
            # Only handle HTTP(S)
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Check for known JavaScript-heavy domains
            js_heavy_domains = {
                'github.com',  # Some GitHub pages are SPA
                'gitlab.com',
                'bitbucket.org',
                'app.',  # Many apps use this subdomain
                'dashboard.',
                'admin.',
                'console.',
                'portal.',
                'spa.',
                'react.',
                'vue.',
                'angular.'
            }
            
            domain = parsed.netloc.lower()
            
            # Check if domain suggests JavaScript-heavy content
            if any(js_domain in domain for js_domain in js_heavy_domains):
                return True
            
            # Check URL path for SPA indicators
            spa_indicators = ['/app/', '/dashboard/', '/admin/', '/console/']
            if any(indicator in parsed.path.lower() for indicator in spa_indicators):
                return True
            
            # For now, let other scrapers handle most URLs
            # Selenium will be used as fallback in LinkEnhancementEngine
            return False
            
        except Exception as e:
            logger.debug(f"Error checking if can scrape {url}: {e}")
            return False
    
    def scrape(self, url: str) -> ScrapingResult:
        """Scrape URL using Selenium WebDriver"""
        
        start_time = time.time()
        
        try:
            logger.debug(f"Starting Selenium scraping for {url}")
            
            # Create driver if not exists
            if not self.driver:
                self.driver = self._create_driver()
            
            # Navigate to URL
            self.driver.get(url)
            
            # Wait for JavaScript to load
            if self.wait_for_js > 0:
                time.sleep(self.wait_for_js)
            
            # Wait for basic page elements
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except TimeoutException:
                logger.warning(f"Timeout waiting for body element on {url}")
            
            # Extract metadata
            title = self._extract_title()
            description = self._extract_description()
            keywords = self._extract_keywords()
            favicon_url = self._extract_favicon_url(url)
            image_url = self._extract_image_url(url)
            author = self._extract_author()
            published_date = self._extract_published_date()
            content_type = self._extract_content_type()
            language = self._extract_language()
            
            # Take screenshot if enabled
            screenshot_path = None
            if self.enable_screenshots:
                screenshot_path = self._take_screenshot(url)
            
            # Extract additional metadata
            metadata = self._extract_additional_metadata()
            if screenshot_path:
                metadata['screenshot_path'] = str(screenshot_path)
            
            scraping_time = time.time() - start_time
            
            result = ScrapingResult(
                url=url,
                title=title,
                description=description,
                keywords=keywords,
                favicon_url=favicon_url,
                image_url=image_url,
                author=author,
                published_date=published_date,
                content_type=content_type,
                language=language,
                success=True,
                scraping_time=scraping_time,
                scraper_used="SeleniumScraper",
                metadata=metadata
            )
            
            logger.debug(f"Selenium scraping completed for {url} in {scraping_time:.2f}s")
            return result
            
        except Exception as e:
            scraping_time = time.time() - start_time
            error_msg = f"Selenium scraping failed: {e}"
            logger.error(f"Error scraping {url}: {error_msg}")
            
            return ScrapingResult(
                url=url,
                success=False,
                error_message=error_msg,
                scraping_time=scraping_time,
                scraper_used="SeleniumScraper"
            )
    
    def _extract_title(self) -> Optional[str]:
        """Extract page title"""
        try:
            # Try multiple methods to get title
            title = None
            
            # Method 1: Standard title tag
            try:
                title = self.driver.title
                if title and title.strip():
                    return title.strip()
            except:
                pass
            
            # Method 2: Open Graph title
            try:
                og_title = self.driver.find_element(By.XPATH, "//meta[@property='og:title']")
                title = og_title.get_attribute('content')
                if title and title.strip():
                    return title.strip()
            except:
                pass
            
            # Method 3: Twitter Card title
            try:
                twitter_title = self.driver.find_element(By.XPATH, "//meta[@name='twitter:title']")
                title = twitter_title.get_attribute('content')
                if title and title.strip():
                    return title.strip()
            except:
                pass
            
            # Method 4: First h1 tag
            try:
                h1 = self.driver.find_element(By.TAG_NAME, "h1")
                title = h1.text
                if title and title.strip():
                    return title.strip()
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting title: {e}")
            return None
    
    def _extract_description(self) -> Optional[str]:
        """Extract page description"""
        try:
            # Method 1: Meta description
            try:
                meta_desc = self.driver.find_element(By.XPATH, "//meta[@name='description']")
                desc = meta_desc.get_attribute('content')
                if desc and desc.strip():
                    return desc.strip()
            except:
                pass
            
            # Method 2: Open Graph description
            try:
                og_desc = self.driver.find_element(By.XPATH, "//meta[@property='og:description']")
                desc = og_desc.get_attribute('content')
                if desc and desc.strip():
                    return desc.strip()
            except:
                pass
            
            # Method 3: Twitter Card description
            try:
                twitter_desc = self.driver.find_element(By.XPATH, "//meta[@name='twitter:description']")
                desc = twitter_desc.get_attribute('content')
                if desc and desc.strip():
                    return desc.strip()
            except:
                pass
            
            # Method 4: First paragraph
            try:
                paragraphs = self.driver.find_elements(By.TAG_NAME, "p")
                for p in paragraphs[:3]:  # Check first 3 paragraphs
                    text = p.text.strip()
                    if text and len(text) > 50:  # Meaningful paragraph
                        return text[:500]  # Limit length
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting description: {e}")
            return None
    
    def _extract_keywords(self) -> List[str]:
        """Extract keywords from page content"""
        try:
            keywords = []
            
            # Method 1: Meta keywords
            try:
                meta_keywords = self.driver.find_element(By.XPATH, "//meta[@name='keywords']")
                keyword_content = meta_keywords.get_attribute('content')
                if keyword_content:
                    keywords.extend([k.strip() for k in keyword_content.split(',') if k.strip()])
            except:
                pass
            
            # Method 2: Extract from page content
            try:
                # Get page text content
                body = self.driver.find_element(By.TAG_NAME, "body")
                page_text = body.text
                
                if page_text:
                    # Use TextUtils to extract keywords
                    extracted_keywords = TextUtils.extract_keywords(page_text, max_keywords=10)
                    keywords.extend(extracted_keywords)
            except:
                pass
            
            # Method 3: Extract from headings
            try:
                headings = self.driver.find_elements(By.XPATH, "//h1 | //h2 | //h3")
                heading_text = " ".join([h.text for h in headings[:5] if h.text])
                if heading_text:
                    heading_keywords = TextUtils.extract_keywords(heading_text, max_keywords=5)
                    keywords.extend(heading_keywords)
            except:
                pass
            
            # Remove duplicates and return
            return list(dict.fromkeys(keywords))[:15]  # Limit to 15 keywords
            
        except Exception as e:
            logger.debug(f"Error extracting keywords: {e}")
            return []
    
    def _extract_favicon_url(self, base_url: str) -> Optional[str]:
        """Extract favicon URL"""
        try:
            # Method 1: Link rel="icon"
            try:
                favicon_link = self.driver.find_element(By.XPATH, "//link[@rel='icon' or @rel='shortcut icon']")
                href = favicon_link.get_attribute('href')
                if href:
                    return urljoin(base_url, href)
            except:
                pass
            
            # Method 2: Apple touch icon
            try:
                apple_icon = self.driver.find_element(By.XPATH, "//link[@rel='apple-touch-icon']")
                href = apple_icon.get_attribute('href')
                if href:
                    return urljoin(base_url, href)
            except:
                pass
            
            # Method 3: Default favicon.ico
            parsed = urlparse(base_url)
            default_favicon = f"{parsed.scheme}://{parsed.netloc}/favicon.ico"
            return default_favicon
            
        except Exception as e:
            logger.debug(f"Error extracting favicon: {e}")
            return None
    
    def _extract_image_url(self, base_url: str) -> Optional[str]:
        """Extract main image URL"""
        try:
            # Method 1: Open Graph image
            try:
                og_image = self.driver.find_element(By.XPATH, "//meta[@property='og:image']")
                image_url = og_image.get_attribute('content')
                if image_url:
                    return urljoin(base_url, image_url)
            except:
                pass
            
            # Method 2: Twitter Card image
            try:
                twitter_image = self.driver.find_element(By.XPATH, "//meta[@name='twitter:image']")
                image_url = twitter_image.get_attribute('content')
                if image_url:
                    return urljoin(base_url, image_url)
            except:
                pass
            
            # Method 3: First significant image
            try:
                images = self.driver.find_elements(By.TAG_NAME, "img")
                for img in images[:5]:  # Check first 5 images
                    src = img.get_attribute('src')
                    if src and not any(skip in src.lower() for skip in ['logo', 'icon', 'avatar', 'button']):
                        return urljoin(base_url, src)
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting image: {e}")
            return None
    
    def _extract_author(self) -> Optional[str]:
        """Extract author information"""
        try:
            # Method 1: Meta author
            try:
                meta_author = self.driver.find_element(By.XPATH, "//meta[@name='author']")
                author = meta_author.get_attribute('content')
                if author and author.strip():
                    return author.strip()
            except:
                pass
            
            # Method 2: Article author
            try:
                article_author = self.driver.find_element(By.XPATH, "//meta[@name='article:author']")
                author = article_author.get_attribute('content')
                if author and author.strip():
                    return author.strip()
            except:
                pass
            
            # Method 3: Schema.org author
            try:
                schema_author = self.driver.find_element(By.XPATH, "//*[@itemprop='author']")
                author = schema_author.text or schema_author.get_attribute('content')
                if author and author.strip():
                    return author.strip()
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting author: {e}")
            return None
    
    def _extract_published_date(self) -> Optional[str]:
        """Extract published date"""
        try:
            # Method 1: Article published time
            try:
                published_time = self.driver.find_element(By.XPATH, "//meta[@property='article:published_time']")
                date = published_time.get_attribute('content')
                if date and date.strip():
                    return date.strip()
            except:
                pass
            
            # Method 2: Schema.org datePublished
            try:
                schema_date = self.driver.find_element(By.XPATH, "//*[@itemprop='datePublished']")
                date = schema_date.get_attribute('datetime') or schema_date.text
                if date and date.strip():
                    return date.strip()
            except:
                pass
            
            # Method 3: Time element
            try:
                time_element = self.driver.find_element(By.TAG_NAME, "time")
                date = time_element.get_attribute('datetime') or time_element.text
                if date and date.strip():
                    return date.strip()
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting published date: {e}")
            return None
    
    def _extract_content_type(self) -> Optional[str]:
        """Extract content type"""
        try:
            # Method 1: Open Graph type
            try:
                og_type = self.driver.find_element(By.XPATH, "//meta[@property='og:type']")
                content_type = og_type.get_attribute('content')
                if content_type and content_type.strip():
                    return content_type.strip()
            except:
                pass
            
            # Method 2: Schema.org type
            try:
                schema_type = self.driver.find_element(By.XPATH, "//*[@itemtype]")
                item_type = schema_type.get_attribute('itemtype')
                if item_type:
                    # Extract type from schema URL
                    if '/' in item_type:
                        return item_type.split('/')[-1]
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting content type: {e}")
            return None
    
    def _extract_language(self) -> Optional[str]:
        """Extract page language"""
        try:
            # Method 1: HTML lang attribute
            try:
                html_element = self.driver.find_element(By.TAG_NAME, "html")
                lang = html_element.get_attribute('lang')
                if lang and lang.strip():
                    return lang.strip()
            except:
                pass
            
            # Method 2: Meta content-language
            try:
                meta_lang = self.driver.find_element(By.XPATH, "//meta[@http-equiv='content-language']")
                lang = meta_lang.get_attribute('content')
                if lang and lang.strip():
                    return lang.strip()
            except:
                pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting language: {e}")
            return None
    
    def _take_screenshot(self, url: str) -> Optional[Path]:
        """Take screenshot of the page"""
        try:
            # Generate filename from URL
            parsed = urlparse(url)
            filename = f"{parsed.netloc}_{int(time.time())}.png"
            screenshot_path = self.screenshot_dir / filename
            
            # Take screenshot
            if self.driver.save_screenshot(str(screenshot_path)):
                logger.debug(f"Screenshot saved: {screenshot_path}")
                return screenshot_path
            else:
                logger.warning(f"Failed to save screenshot for {url}")
                return None
                
        except Exception as e:
            logger.debug(f"Error taking screenshot: {e}")
            return None
    
    def _extract_additional_metadata(self) -> Dict[str, Any]:
        """Extract additional metadata"""
        metadata = {}
        
        try:
            # Page load time
            navigation_start = self.driver.execute_script("return window.performance.timing.navigationStart")
            load_complete = self.driver.execute_script("return window.performance.timing.loadEventEnd")
            if navigation_start and load_complete:
                metadata['page_load_time'] = (load_complete - navigation_start) / 1000.0
            
            # JavaScript frameworks detection
            frameworks = []
            
            # Check for common frameworks
            framework_checks = {
                'React': "return typeof React !== 'undefined'",
                'Vue': "return typeof Vue !== 'undefined'",
                'Angular': "return typeof angular !== 'undefined' || typeof ng !== 'undefined'",
                'jQuery': "return typeof jQuery !== 'undefined' || typeof $ !== 'undefined'"
            }
            
            for framework, check in framework_checks.items():
                try:
                    if self.driver.execute_script(check):
                        frameworks.append(framework)
                except:
                    pass
            
            if frameworks:
                metadata['javascript_frameworks'] = frameworks
            
            # Page dimensions
            try:
                page_height = self.driver.execute_script("return document.body.scrollHeight")
                page_width = self.driver.execute_script("return document.body.scrollWidth")
                metadata['page_dimensions'] = {'width': page_width, 'height': page_height}
            except:
                pass
            
            # Number of elements
            try:
                element_count = len(self.driver.find_elements(By.XPATH, "//*"))
                metadata['element_count'] = element_count
            except:
                pass
            
        except Exception as e:
            logger.debug(f"Error extracting additional metadata: {e}")
        
        return metadata
    
    def get_scraper_info(self) -> Dict[str, Any]:
        """Get scraper information"""
        return {
            'name': 'SeleniumScraper',
            'description': 'Selenium-based scraper for JavaScript-heavy sites and dynamic content',
            'capabilities': [
                'JavaScript execution',
                'Dynamic content loading',
                'Screenshot capture',
                'SPA support',
                'Advanced metadata extraction'
            ],
            'browser': self.browser,
            'headless': self.headless,
            'screenshots_enabled': self.enable_screenshots,
            'selenium_available': SELENIUM_AVAILABLE,
            'driver_active': self.driver is not None
        }
    
    def cleanup(self) -> None:
        """Cleanup WebDriver resources"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
                logger.debug("Selenium WebDriver closed")
        except Exception as e:
            logger.error(f"Error closing WebDriver: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()