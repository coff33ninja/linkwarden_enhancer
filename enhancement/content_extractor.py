"""Content snippet extraction system for bookmark descriptions"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, unquote
from bs4 import BeautifulSoup

from utils.logging_utils import get_logger
from utils.text_utils import TextUtils

logger = get_logger(__name__)


@dataclass
class ContentSnippet:
    """Extracted content snippet"""
    text: str
    source: str  # 'first_paragraph', 'content_area', 'url_analysis', 'title_expansion'
    confidence: float
    quality_score: float
    word_count: int
    extraction_method: str


@dataclass
class UrlAnalysisResult:
    """Result from URL structure analysis"""
    path_segments: List[str]
    meaningful_segments: List[str]
    file_name: str
    query_params: Dict[str, str]
    inferred_topic: str
    confidence: float


class ContentExtractor:
    """Extract meaningful content snippets as fallback for descriptions"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize content extractor"""
        self.config = config
        self.enhancement_config = config.get('enhancement', {})
        
        # Extraction settings
        self.min_snippet_length = self.enhancement_config.get('min_snippet_length', 50)
        self.max_snippet_length = self.enhancement_config.get('max_snippet_length', 200)
        self.ideal_snippet_length = 120
        
        # Content area selectors (in priority order)
        self.content_selectors = [
            'main',
            'article',
            '.content',
            '.main-content',
            '#content',
            '#main',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.page-content',
            'section',
            '.container'
        ]
        
        # Paragraph selectors for first paragraph extraction
        self.paragraph_selectors = [
            'main p',
            'article p',
            '.content p',
            '.main-content p',
            '#content p',
            '.post-content p',
            '.entry-content p',
            'p'
        ]
        
        # Elements to exclude from content extraction
        self.exclude_selectors = [
            'nav', 'header', 'footer', 'aside', 'menu',
            '.navigation', '.nav', '.menu', '.sidebar',
            '.header', '.footer', '.ads', '.advertisement',
            '.social', '.share', '.comments', '.comment',
            '.breadcrumb', '.pagination', '.tags', '.categories',
            'script', 'style', 'noscript'
        ]
        
        # Boilerplate text patterns
        self.boilerplate_patterns = [
            r'copyright \d{4}',
            r'all rights reserved',
            r'privacy policy',
            r'terms of service',
            r'cookie policy',
            r'subscribe to',
            r'follow us',
            r'share this',
            r'read more',
            r'click here',
            r'sign up',
            r'log in',
            r'login',
            r'register',
            r'home\s*>\s*',
            r'breadcrumb',
            r'skip to',
            r'back to top',
            r'loading\.\.\.?',
            r'please wait',
            r'redirecting'
        ]
        
        # URL analysis patterns
        self.url_patterns = {
            'github': {
                'patterns': [r'github\.com/([^/]+)/([^/]+)'],
                'template': 'GitHub repository: {1} by {0}'
            },
            'stackoverflow': {
                'patterns': [r'stackoverflow\.com/questions/\d+/([^/]+)'],
                'template': 'Stack Overflow: {0}'
            },
            'medium': {
                'patterns': [r'medium\.com/@?([^/]+)/([^/]+)'],
                'template': 'Medium article: {1} by {0}'
            },
            'youtube': {
                'patterns': [r'youtube\.com/watch\?v=([^&]+)', r'youtu\.be/([^?]+)'],
                'template': 'YouTube video: {0}'
            },
            'reddit': {
                'patterns': [r'reddit\.com/r/([^/]+)/comments/[^/]+/([^/]+)'],
                'template': 'Reddit discussion in r/{0}: {1}'
            }
        }
        
        logger.info("Content extractor initialized")
    
    def extract_content_snippet(self, html_content: str, url: str = "", title: str = "") -> ContentSnippet:
        """Extract the best content snippet from HTML"""
        
        if not html_content:
            return self._fallback_to_url_analysis(url, title)
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Try different extraction methods in priority order
            extraction_methods = [
                ('first_paragraph', self._extract_first_paragraph),
                ('content_area', self._extract_from_content_area),
                ('any_paragraph', self._extract_any_meaningful_paragraph),
                ('text_blocks', self._extract_text_blocks)
            ]
            
            best_snippet = None
            
            for method_name, method in extraction_methods:
                snippet = method(soup, url, title)
                
                if snippet and snippet.text:
                    snippet.extraction_method = method_name
                    
                    # If this snippet is good enough, use it
                    if snippet.quality_score > 0.7 and snippet.confidence > 0.6:
                        logger.debug(f"Using {method_name} extraction: {len(snippet.text)} chars")
                        return snippet
                    
                    # Keep track of best snippet so far
                    if not best_snippet or snippet.quality_score > best_snippet.quality_score:
                        best_snippet = snippet
            
            # Return best snippet found, or fallback to URL analysis
            if best_snippet:
                logger.debug(f"Using best snippet from {best_snippet.extraction_method}: {len(best_snippet.text)} chars")
                return best_snippet
            else:
                logger.debug("No content snippet found, falling back to URL analysis")
                return self._fallback_to_url_analysis(url, title)
                
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return self._fallback_to_url_analysis(url, title)
    
    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from soup"""
        
        for selector in self.exclude_selectors:
            for element in soup.select(selector):
                element.decompose()
    
    def _extract_first_paragraph(self, soup: BeautifulSoup, url: str, title: str) -> Optional[ContentSnippet]:
        """Extract first meaningful paragraph"""
        
        try:
            for selector in self.paragraph_selectors:
                paragraphs = soup.select(selector)
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    
                    # Skip short paragraphs
                    if len(text) < self.min_snippet_length:
                        continue
                    
                    # Skip boilerplate text
                    if self._is_boilerplate_text(text):
                        continue
                    
                    # Clean and validate
                    cleaned_text = self._clean_snippet_text(text)
                    if not cleaned_text or len(cleaned_text) < self.min_snippet_length:
                        continue
                    
                    # Truncate if too long
                    if len(cleaned_text) > self.max_snippet_length:
                        cleaned_text = self._smart_truncate(cleaned_text, self.max_snippet_length)
                    
                    quality_score = self._calculate_snippet_quality(cleaned_text, 'first_paragraph', title)
                    confidence = self._calculate_confidence(cleaned_text, 'first_paragraph', quality_score)
                    
                    return ContentSnippet(
                        text=cleaned_text,
                        source='first_paragraph',
                        confidence=confidence,
                        quality_score=quality_score,
                        word_count=len(cleaned_text.split()),
                        extraction_method='first_paragraph'
                    )
            
            return None
            
        except Exception as e:
            logger.debug(f"First paragraph extraction failed: {e}")
            return None
    
    def _extract_from_content_area(self, soup: BeautifulSoup, url: str, title: str) -> Optional[ContentSnippet]:
        """Extract from main content areas"""
        
        try:
            for selector in self.content_selectors:
                content_areas = soup.select(selector)
                
                for area in content_areas:
                    # Get text from this content area
                    text = area.get_text(strip=True)
                    
                    if len(text) < self.min_snippet_length:
                        continue
                    
                    # Try to find the first meaningful sentence or paragraph
                    sentences = text.split('.')
                    meaningful_text = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) < 20:  # Skip very short sentences
                            continue
                        
                        if self._is_boilerplate_text(sentence):
                            continue
                        
                        # Build meaningful text
                        if not meaningful_text:
                            meaningful_text = sentence
                        else:
                            if len(meaningful_text + '. ' + sentence) <= self.max_snippet_length:
                                meaningful_text += '. ' + sentence
                            else:
                                break
                        
                        # Stop if we have enough content
                        if len(meaningful_text) >= self.ideal_snippet_length:
                            break
                    
                    if meaningful_text:
                        cleaned_text = self._clean_snippet_text(meaningful_text)
                        if cleaned_text and len(cleaned_text) >= self.min_snippet_length:
                            
                            # Ensure proper ending
                            if not cleaned_text.endswith('.'):
                                cleaned_text += '.'
                            
                            quality_score = self._calculate_snippet_quality(cleaned_text, 'content_area', title)
                            confidence = self._calculate_confidence(cleaned_text, 'content_area', quality_score)
                            
                            return ContentSnippet(
                                text=cleaned_text,
                                source='content_area',
                                confidence=confidence,
                                quality_score=quality_score,
                                word_count=len(cleaned_text.split()),
                                extraction_method='content_area'
                            )
            
            return None
            
        except Exception as e:
            logger.debug(f"Content area extraction failed: {e}")
            return None
    
    def _extract_any_meaningful_paragraph(self, soup: BeautifulSoup, url: str, title: str) -> Optional[ContentSnippet]:
        """Extract any meaningful paragraph as fallback"""
        
        try:
            # Get all paragraphs
            all_paragraphs = soup.find_all('p')
            
            # Score paragraphs by various criteria
            scored_paragraphs = []
            
            for p in all_paragraphs:
                text = p.get_text(strip=True)
                
                if len(text) < self.min_snippet_length:
                    continue
                
                if self._is_boilerplate_text(text):
                    continue
                
                # Score this paragraph
                score = 0.0
                
                # Length scoring (prefer moderate length)
                if self.ideal_snippet_length * 0.8 <= len(text) <= self.ideal_snippet_length * 1.5:
                    score += 0.3
                elif len(text) >= self.min_snippet_length:
                    score += 0.2
                
                # Position scoring (earlier paragraphs are often better)
                position = len(scored_paragraphs)
                position_score = max(0, 1 - position * 0.1)
                score += position_score * 0.2
                
                # Content quality scoring
                if re.search(r'[.!?]', text):  # Has proper punctuation
                    score += 0.1
                
                if len(text.split()) > 10:  # Has reasonable word count
                    score += 0.1
                
                # Title relevance
                if title:
                    title_words = set(title.lower().split())
                    text_words = set(text.lower().split())
                    if title_words and text_words:
                        overlap = len(title_words & text_words) / len(title_words)
                        score += overlap * 0.3
                
                scored_paragraphs.append((text, score))
            
            if not scored_paragraphs:
                return None
            
            # Sort by score and take the best
            scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
            best_text, best_score = scored_paragraphs[0]
            
            # Clean and prepare the text
            cleaned_text = self._clean_snippet_text(best_text)
            if not cleaned_text:
                return None
            
            # Truncate if needed
            if len(cleaned_text) > self.max_snippet_length:
                cleaned_text = self._smart_truncate(cleaned_text, self.max_snippet_length)
            
            quality_score = self._calculate_snippet_quality(cleaned_text, 'any_paragraph', title)
            confidence = self._calculate_confidence(cleaned_text, 'any_paragraph', quality_score)
            
            return ContentSnippet(
                text=cleaned_text,
                source='any_paragraph',
                confidence=confidence,
                quality_score=quality_score,
                word_count=len(cleaned_text.split()),
                extraction_method='any_paragraph'
            )
            
        except Exception as e:
            logger.debug(f"Any paragraph extraction failed: {e}")
            return None
    
    def _extract_text_blocks(self, soup: BeautifulSoup, url: str, title: str) -> Optional[ContentSnippet]:
        """Extract from any text blocks as last resort"""
        
        try:
            # Get all text from the page
            all_text = soup.get_text(separator=' ', strip=True)
            
            if len(all_text) < self.min_snippet_length:
                return None
            
            # Split into sentences and find meaningful ones
            sentences = re.split(r'[.!?]+', all_text)
            meaningful_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                if len(sentence) < 20:
                    continue
                
                if self._is_boilerplate_text(sentence):
                    continue
                
                meaningful_sentences.append(sentence)
                
                # Stop when we have enough
                if len(meaningful_sentences) >= 3:
                    break
            
            if not meaningful_sentences:
                return None
            
            # Combine sentences up to length limit
            combined_text = ""
            for sentence in meaningful_sentences:
                if not combined_text:
                    combined_text = sentence
                else:
                    if len(combined_text + '. ' + sentence) <= self.max_snippet_length:
                        combined_text += '. ' + sentence
                    else:
                        break
            
            if not combined_text:
                return None
            
            # Clean and finalize
            cleaned_text = self._clean_snippet_text(combined_text)
            if not cleaned_text or len(cleaned_text) < self.min_snippet_length:
                return None
            
            if not cleaned_text.endswith('.'):
                cleaned_text += '.'
            
            quality_score = self._calculate_snippet_quality(cleaned_text, 'text_blocks', title)
            confidence = self._calculate_confidence(cleaned_text, 'text_blocks', quality_score)
            
            return ContentSnippet(
                text=cleaned_text,
                source='text_blocks',
                confidence=confidence,
                quality_score=quality_score,
                word_count=len(cleaned_text.split()),
                extraction_method='text_blocks'
            )
            
        except Exception as e:
            logger.debug(f"Text blocks extraction failed: {e}")
            return None
    
    def _fallback_to_url_analysis(self, url: str, title: str) -> ContentSnippet:
        """Generate description from URL structure analysis"""
        
        try:
            if not url:
                return self._create_title_based_snippet(title)
            
            # Analyze URL structure
            url_analysis = self._analyze_url_structure(url)
            
            # Try domain-specific patterns first
            domain_description = self._generate_domain_specific_description(url, url_analysis)
            if domain_description:
                quality_score = self._calculate_snippet_quality(domain_description, 'url_analysis', title)
                confidence = self._calculate_confidence(domain_description, 'url_analysis', quality_score)
                
                return ContentSnippet(
                    text=domain_description,
                    source='url_analysis',
                    confidence=confidence,
                    quality_score=quality_score,
                    word_count=len(domain_description.split()),
                    extraction_method='url_analysis'
                )
            
            # Generate generic description from URL components
            generic_description = self._generate_generic_url_description(url, url_analysis, title)
            
            quality_score = self._calculate_snippet_quality(generic_description, 'url_analysis', title)
            confidence = self._calculate_confidence(generic_description, 'url_analysis', quality_score)
            
            return ContentSnippet(
                text=generic_description,
                source='url_analysis',
                confidence=confidence,
                quality_score=quality_score,
                word_count=len(generic_description.split()),
                extraction_method='url_analysis'
            )
            
        except Exception as e:
            logger.debug(f"URL analysis fallback failed: {e}")
            return self._create_title_based_snippet(title)
    
    def _analyze_url_structure(self, url: str) -> UrlAnalysisResult:
        """Analyze URL structure for meaningful components"""
        
        try:
            parsed = urlparse(url)
            
            # Extract path segments
            path_segments = [segment for segment in parsed.path.split('/') if segment]
            
            # Decode URL-encoded segments
            decoded_segments = [unquote(segment) for segment in path_segments]
            
            # Find meaningful segments (not just IDs or common paths)
            meaningful_segments = []
            common_paths = {'api', 'v1', 'v2', 'docs', 'blog', 'post', 'article', 'page', 'index'}
            
            for segment in decoded_segments:
                # Skip numeric IDs
                if segment.isdigit():
                    continue
                
                # Skip very short segments
                if len(segment) < 3:
                    continue
                
                # Skip common path components
                if segment.lower() in common_paths:
                    continue
                
                # Convert dashes/underscores to spaces and clean
                cleaned_segment = re.sub(r'[-_]+', ' ', segment)
                cleaned_segment = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_segment)
                
                if cleaned_segment.strip():
                    meaningful_segments.append(cleaned_segment.strip())
            
            # Extract filename
            file_name = ""
            if path_segments:
                last_segment = path_segments[-1]
                if '.' in last_segment:
                    file_name = last_segment.split('.')[0]
                    file_name = re.sub(r'[-_]+', ' ', file_name)
            
            # Parse query parameters
            query_params = {}
            if parsed.query:
                for param in parsed.query.split('&'):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        query_params[unquote(key)] = unquote(value)
            
            # Infer topic from segments
            inferred_topic = ""
            if meaningful_segments:
                inferred_topic = ' '.join(meaningful_segments[:3])  # Use first 3 segments
            elif file_name:
                inferred_topic = file_name
            
            # Calculate confidence based on how much meaningful info we extracted
            confidence = 0.0
            if meaningful_segments:
                confidence += 0.4
            if file_name:
                confidence += 0.2
            if query_params:
                confidence += 0.1
            if inferred_topic:
                confidence += 0.3
            
            return UrlAnalysisResult(
                path_segments=path_segments,
                meaningful_segments=meaningful_segments,
                file_name=file_name,
                query_params=query_params,
                inferred_topic=inferred_topic,
                confidence=min(1.0, confidence)
            )
            
        except Exception as e:
            logger.debug(f"URL structure analysis failed: {e}")
            return UrlAnalysisResult(
                path_segments=[],
                meaningful_segments=[],
                file_name="",
                query_params={},
                inferred_topic="",
                confidence=0.0
            )
    
    def _generate_domain_specific_description(self, url: str, url_analysis: UrlAnalysisResult) -> Optional[str]:
        """Generate domain-specific description using URL patterns"""
        
        try:
            for domain, config in self.url_patterns.items():
                for pattern in config['patterns']:
                    match = re.search(pattern, url)
                    if match:
                        # Extract matched groups
                        groups = match.groups()
                        
                        # Clean up the groups (replace dashes/underscores with spaces)
                        cleaned_groups = []
                        for group in groups:
                            cleaned = re.sub(r'[-_]+', ' ', group)
                            cleaned = unquote(cleaned)
                            cleaned_groups.append(cleaned.title())
                        
                        # Format using template
                        try:
                            description = config['template'].format(*cleaned_groups)
                            return description
                        except (IndexError, KeyError):
                            continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Domain-specific description generation failed: {e}")
            return None
    
    def _generate_generic_url_description(self, url: str, url_analysis: UrlAnalysisResult, title: str) -> str:
        """Generate generic description from URL components"""
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Start with domain
            description_parts = []
            
            # Add domain context
            if domain:
                domain_name = domain.split('.')[0].title()
                description_parts.append(f"Content from {domain_name}")
            
            # Add topic if available
            if url_analysis.inferred_topic:
                topic = url_analysis.inferred_topic.title()
                description_parts.append(f"about {topic}")
            
            # Add title context if available and different from topic
            if title and url_analysis.inferred_topic:
                title_words = set(title.lower().split())
                topic_words = set(url_analysis.inferred_topic.lower().split())
                if len(title_words - topic_words) > 2:  # Title has additional info
                    description_parts.append(f"titled '{title}'")
            elif title:
                description_parts.append(f"titled '{title}'")
            
            # Combine parts
            if len(description_parts) == 1:
                description = description_parts[0] + "."
            elif len(description_parts) == 2:
                description = f"{description_parts[0]} {description_parts[1]}."
            else:
                description = f"{description_parts[0]} {description_parts[1]} {description_parts[2]}."
            
            # Ensure reasonable length
            if len(description) > self.max_snippet_length:
                description = description[:self.max_snippet_length-3] + "..."
            
            return description
            
        except Exception as e:
            logger.debug(f"Generic URL description generation failed: {e}")
            return f"Web content from {urlparse(url).netloc if url else 'unknown source'}."
    
    def _create_title_based_snippet(self, title: str) -> ContentSnippet:
        """Create snippet based on title as ultimate fallback"""
        
        if title and len(title) > 10:
            description = f"Content titled '{title}'."
        else:
            description = "Web content."
        
        return ContentSnippet(
            text=description,
            source='title_expansion',
            confidence=0.2,
            quality_score=0.3,
            word_count=len(description.split()),
            extraction_method='title_expansion'
        )
    
    def _clean_snippet_text(self, text: str) -> str:
        """Clean and normalize snippet text"""
        
        if not text:
            return ""
        
        # Basic text cleaning
        cleaned = TextUtils.clean_text(text)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove common prefixes
        prefixes_to_remove = [
            'description:', 'summary:', 'about:', 'overview:',
            'read more:', 'learn more:', 'click here:'
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Ensure proper capitalization
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned.strip()
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """Intelligently truncate text at sentence boundaries"""
        
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundary
        sentences = text.split('.')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + '.') <= max_length - 3:
                truncated += sentence + '.'
            else:
                break
        
        if truncated and len(truncated) >= self.min_snippet_length:
            return truncated.strip()
        else:
            # Hard truncate with ellipsis
            return text[:max_length-3] + "..."
    
    def _is_boilerplate_text(self, text: str) -> bool:
        """Check if text appears to be boilerplate content"""
        
        text_lower = text.lower()
        
        # Check against boilerplate patterns
        for pattern in self.boilerplate_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for navigation-like structure
        if text.count('|') > 3 or text.count('»') > 2:
            return True
        
        # Check for very repetitive content
        words = text_lower.split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If any word appears more than 30% of the time, it's likely boilerplate
            max_count = max(word_counts.values())
            if max_count / len(words) > 0.3:
                return True
        
        return False
    
    def _calculate_snippet_quality(self, text: str, source: str, title: str) -> float:
        """Calculate quality score for extracted snippet"""
        
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length scoring (0.3 weight)
        length = len(text)
        if self.ideal_snippet_length * 0.8 <= length <= self.ideal_snippet_length * 1.2:
            length_score = 1.0
        elif self.min_snippet_length <= length <= self.max_snippet_length:
            length_score = 0.8
        else:
            length_score = 0.5
        
        score += length_score * 0.3
        
        # Source quality (0.2 weight)
        source_scores = {
            'first_paragraph': 1.0,
            'content_area': 0.9,
            'any_paragraph': 0.7,
            'text_blocks': 0.5,
            'url_analysis': 0.4,
            'title_expansion': 0.2
        }
        
        source_score = source_scores.get(source, 0.5)
        score += source_score * 0.2
        
        # Content quality (0.3 weight)
        content_score = 0.0
        
        # Check for complete sentences
        if re.search(r'[.!?]$', text):
            content_score += 0.3
        
        # Check for reasonable word count
        word_count = len(text.split())
        if 10 <= word_count <= 40:
            content_score += 0.3
        elif word_count >= 5:
            content_score += 0.2
        
        # Check for descriptive content
        descriptive_indicators = ['provides', 'offers', 'features', 'includes', 'helps', 'enables', 'allows']
        if any(indicator in text.lower() for indicator in descriptive_indicators):
            content_score += 0.2
        
        # Check for specific information
        if re.search(r'\d+', text):
            content_score += 0.1
        
        # Penalty for generic content
        generic_phrases = ['welcome to', 'home page', 'main page', 'click here', 'read more']
        for phrase in generic_phrases:
            if phrase in text.lower():
                content_score -= 0.2
        
        content_score = max(0.0, min(1.0, content_score))
        score += content_score * 0.3
        
        # Title relevance (0.2 weight)
        if title:
            title_words = set(title.lower().split())
            text_words = set(text.lower().split())
            if title_words and text_words:
                overlap = len(title_words & text_words) / len(title_words)
                relevance_score = min(1.0, overlap * 2)  # Scale up since snippets are shorter
            else:
                relevance_score = 0.5
        else:
            relevance_score = 0.5
        
        score += relevance_score * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_confidence(self, text: str, source: str, quality_score: float) -> float:
        """Calculate confidence in the extracted snippet"""
        
        # Base confidence by source
        base_confidence = {
            'first_paragraph': 0.8,
            'content_area': 0.7,
            'any_paragraph': 0.6,
            'text_blocks': 0.4,
            'url_analysis': 0.3,
            'title_expansion': 0.2
        }.get(source, 0.3)
        
        # Adjust by quality score
        quality_factor = quality_score * 0.3
        
        # Adjust by length appropriateness
        length = len(text)
        if self.min_snippet_length <= length <= self.max_snippet_length:
            length_factor = 0.2
        else:
            length_factor = 0.0
        
        confidence = base_confidence + quality_factor + length_factor
        return max(0.0, min(1.0, confidence))
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics and configuration"""
        
        return {
            'min_snippet_length': self.min_snippet_length,
            'max_snippet_length': self.max_snippet_length,
            'ideal_snippet_length': self.ideal_snippet_length,
            'content_selectors': len(self.content_selectors),
            'paragraph_selectors': len(self.paragraph_selectors),
            'exclude_selectors': len(self.exclude_selectors),
            'boilerplate_patterns': len(self.boilerplate_patterns),
            'supported_domains': list(self.url_patterns.keys()),
            'extraction_methods': [
                'first_paragraph', 'content_area', 'any_paragraph', 
                'text_blocks', 'url_analysis', 'title_expansion'
            ]
        }