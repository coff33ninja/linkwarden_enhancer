"""Meta description extraction system for bookmark enhancement"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

from utils.logging_utils import get_logger
from utils.text_utils import TextUtils

logger = get_logger(__name__)


@dataclass
class MetaDescriptionResult:
    """Result from meta description extraction"""
    description: Optional[str] = None
    source: str = ""  # 'meta', 'og', 'twitter', 'summary', 'first_paragraph'
    confidence: float = 0.0
    length: int = 0
    quality_score: float = 0.0
    fallback_used: bool = False


class MetaDescriptionExtractor:
    """Extract meta descriptions from web pages with fallback hierarchy"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize meta description extractor"""
        self.config = config
        self.enhancement_config = config.get('enhancement', {})
        
        # Meta tag selectors in priority order
        self.meta_selectors = [
            ('meta[name="description"]', 'meta'),
            ('meta[property="og:description"]', 'og'),
            ('meta[name="twitter:description"]', 'twitter'),
            ('meta[name="summary"]', 'summary'),
            ('meta[property="description"]', 'meta_property'),
            ('meta[name="twitter:card"]', 'twitter_card'),
        ]
        
        # Quality thresholds
        self.min_length = self.enhancement_config.get('min_description_length', 20)
        self.max_length = self.enhancement_config.get('max_description_length', 300)
        self.ideal_length_min = 100
        self.ideal_length_max = 200
        
        # Common patterns to filter out
        self.generic_patterns = [
            r'^page \d+',
            r'^untitled',
            r'^default',
            r'^home\s*$',
            r'^index\s*$',
            r'^\s*$',
            r'^loading\.\.\.?',
            r'^please wait',
            r'^redirecting',
            r'^error \d+',
            r'^not found',
            r'^access denied',
        ]
        
        logger.info("Meta description extractor initialized")
    
    def extract_from_html(self, html_content: str, url: str = "") -> MetaDescriptionResult:
        """Extract meta description from HTML content"""
        
        if not html_content:
            return MetaDescriptionResult()
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try each meta selector in priority order
            for selector, source_type in self.meta_selectors:
                result = self._extract_from_selector(soup, selector, source_type)
                if result.description:
                    logger.debug(f"Found meta description from {source_type}: {len(result.description)} chars")
                    return result
            
            # Fallback to first paragraph extraction
            fallback_result = self._extract_first_paragraph(soup, url)
            if fallback_result.description:
                logger.debug(f"Using first paragraph fallback: {len(fallback_result.description)} chars")
                return fallback_result
            
            logger.debug("No meta description found")
            return MetaDescriptionResult()
            
        except Exception as e:
            logger.error(f"Failed to extract meta description: {e}")
            return MetaDescriptionResult()
    
    def extract_from_url(self, url: str, timeout: int = 10) -> MetaDescriptionResult:
        """Extract meta description by fetching URL"""
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            return self.extract_from_html(response.text, url)
            
        except Exception as e:
            logger.warning(f"Failed to fetch URL for meta description extraction: {e}")
            return MetaDescriptionResult()
    
    def _extract_from_selector(self, soup: BeautifulSoup, selector: str, source_type: str) -> MetaDescriptionResult:
        """Extract description from specific meta tag selector"""
        
        try:
            meta_tag = soup.select_one(selector)
            if not meta_tag:
                return MetaDescriptionResult()
            
            # Get content from different attributes
            content = (
                meta_tag.get('content') or 
                meta_tag.get('value') or 
                meta_tag.get_text(strip=True)
            )
            
            if not content:
                return MetaDescriptionResult()
            
            # Clean and validate the description
            cleaned_description = self._clean_description(content)
            if not cleaned_description:
                return MetaDescriptionResult()
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(cleaned_description, source_type)
            
            # Calculate confidence based on source type and quality
            confidence = self._calculate_confidence(source_type, quality_score, len(cleaned_description))
            
            return MetaDescriptionResult(
                description=cleaned_description,
                source=source_type,
                confidence=confidence,
                length=len(cleaned_description),
                quality_score=quality_score,
                fallback_used=False
            )
            
        except Exception as e:
            logger.debug(f"Failed to extract from selector {selector}: {e}")
            return MetaDescriptionResult()
    
    def _extract_first_paragraph(self, soup: BeautifulSoup, url: str) -> MetaDescriptionResult:
        """Extract first meaningful paragraph as fallback"""
        
        try:
            # Look for main content areas first
            content_selectors = [
                'main p',
                'article p',
                '.content p',
                '.main p',
                '#content p',
                '#main p',
                'p'
            ]
            
            for selector in content_selectors:
                paragraphs = soup.select(selector)
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    
                    # Skip short paragraphs or navigation text
                    if len(text) < self.min_length:
                        continue
                    
                    # Skip paragraphs that look like navigation or boilerplate
                    if self._is_boilerplate_text(text):
                        continue
                    
                    # Clean and validate
                    cleaned_text = self._clean_description(text)
                    if cleaned_text and len(cleaned_text) >= self.min_length:
                        
                        # Truncate if too long
                        if len(cleaned_text) > self.max_length:
                            cleaned_text = cleaned_text[:self.max_length-3] + "..."
                        
                        quality_score = self._calculate_quality_score(cleaned_text, 'first_paragraph')
                        confidence = self._calculate_confidence('first_paragraph', quality_score, len(cleaned_text))
                        
                        return MetaDescriptionResult(
                            description=cleaned_text,
                            source='first_paragraph',
                            confidence=confidence,
                            length=len(cleaned_text),
                            quality_score=quality_score,
                            fallback_used=True
                        )
            
            return MetaDescriptionResult()
            
        except Exception as e:
            logger.debug(f"Failed to extract first paragraph: {e}")
            return MetaDescriptionResult()
    
    def _clean_description(self, text: str) -> str:
        """Clean and normalize description text"""
        
        if not text:
            return ""
        
        # Basic text cleaning
        cleaned = TextUtils.clean_text(text)
        
        # Remove HTML entities and tags
        cleaned = re.sub(r'&[a-zA-Z0-9#]+;', ' ', cleaned)
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        
        # Remove extra whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Check against generic patterns
        for pattern in self.generic_patterns:
            if re.match(pattern, cleaned.lower()):
                return ""
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = [
            'description:', 'summary:', 'about:', 'overview:',
            'read more:', 'learn more:', 'click here:'
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove trailing punctuation repetition
        cleaned = re.sub(r'[.!?]{2,}$', '.', cleaned)
        
        # Ensure proper sentence ending
        if cleaned and not cleaned.endswith(('.', '!', '?', '…')):
            cleaned += '.'
        
        return cleaned.strip()
    
    def _calculate_quality_score(self, description: str, source_type: str) -> float:
        """Calculate quality score for description (0-1)"""
        
        if not description:
            return 0.0
        
        score = 0.0
        
        # Length scoring (ideal range gets highest score)
        length = len(description)
        if self.ideal_length_min <= length <= self.ideal_length_max:
            length_score = 1.0
        elif length < self.ideal_length_min:
            length_score = length / self.ideal_length_min
        else:
            # Penalty for being too long
            length_score = max(0.3, 1.0 - (length - self.ideal_length_max) / 200)
        
        score += length_score * 0.3
        
        # Source type scoring
        source_scores = {
            'meta': 1.0,
            'og': 0.9,
            'twitter': 0.8,
            'summary': 0.7,
            'meta_property': 0.6,
            'first_paragraph': 0.4,
            'twitter_card': 0.3
        }
        
        source_score = source_scores.get(source_type, 0.5)
        score += source_score * 0.2
        
        # Content quality indicators
        content_score = 0.0
        
        # Check for complete sentences
        sentences = description.split('.')
        complete_sentences = sum(1 for s in sentences if len(s.strip()) > 5)
        if complete_sentences > 0:
            content_score += 0.3
        
        # Check for descriptive words
        descriptive_words = [
            'provides', 'offers', 'features', 'includes', 'contains',
            'helps', 'enables', 'allows', 'supports', 'delivers'
        ]
        if any(word in description.lower() for word in descriptive_words):
            content_score += 0.2
        
        # Check for specific information (numbers, proper nouns)
        if re.search(r'\d+', description):
            content_score += 0.1
        
        if re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', description):
            content_score += 0.1
        
        # Penalty for generic phrases
        generic_phrases = [
            'welcome to', 'home page', 'main page', 'default page',
            'coming soon', 'under construction', 'page not found'
        ]
        
        for phrase in generic_phrases:
            if phrase in description.lower():
                content_score -= 0.2
        
        content_score = max(0.0, min(1.0, content_score))
        score += content_score * 0.3
        
        # Readability scoring (simple)
        words = description.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            # Prefer moderate word length (4-8 characters)
            if 4 <= avg_word_length <= 8:
                readability_score = 1.0
            else:
                readability_score = max(0.3, 1.0 - abs(avg_word_length - 6) / 10)
        else:
            readability_score = 0.0
        
        score += readability_score * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _calculate_confidence(self, source_type: str, quality_score: float, length: int) -> float:
        """Calculate confidence in the extracted description"""
        
        # Base confidence by source type
        base_confidence = {
            'meta': 0.9,
            'og': 0.85,
            'twitter': 0.8,
            'summary': 0.75,
            'meta_property': 0.7,
            'first_paragraph': 0.5,
            'twitter_card': 0.4
        }.get(source_type, 0.3)
        
        # Adjust by quality score
        quality_factor = quality_score * 0.3
        
        # Adjust by length appropriateness
        if self.ideal_length_min <= length <= self.ideal_length_max:
            length_factor = 0.2
        elif length >= self.min_length:
            length_factor = 0.1
        else:
            length_factor = -0.2
        
        confidence = base_confidence + quality_factor + length_factor
        return max(0.0, min(1.0, confidence))
    
    def _is_boilerplate_text(self, text: str) -> bool:
        """Check if text appears to be boilerplate/navigation content"""
        
        text_lower = text.lower()
        
        # Common boilerplate patterns
        boilerplate_patterns = [
            r'copyright \d{4}',
            r'all rights reserved',
            r'privacy policy',
            r'terms of service',
            r'cookie policy',
            r'sign up',
            r'log in',
            r'subscribe',
            r'follow us',
            r'share this',
            r'read more',
            r'click here',
            r'menu',
            r'navigation',
            r'breadcrumb',
            r'skip to',
            r'back to top'
        ]
        
        for pattern in boilerplate_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for navigation-like structure (lots of links)
        if text.count('|') > 3 or text.count('»') > 2:
            return True
        
        # Check for very short sentences (likely navigation)
        sentences = text.split('.')
        short_sentences = sum(1 for s in sentences if len(s.strip()) < 10)
        if len(sentences) > 1 and short_sentences / len(sentences) > 0.7:
            return True
        
        return False
    
    def extract_multiple_sources(self, html_content: str, url: str = "") -> List[MetaDescriptionResult]:
        """Extract descriptions from all available sources for comparison"""
        
        results = []
        
        if not html_content:
            return results
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract from all meta selectors
            for selector, source_type in self.meta_selectors:
                result = self._extract_from_selector(soup, selector, source_type)
                if result.description:
                    results.append(result)
            
            # Add first paragraph fallback
            fallback_result = self._extract_first_paragraph(soup, url)
            if fallback_result.description:
                results.append(fallback_result)
            
            # Sort by confidence score
            results.sort(key=lambda r: r.confidence, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to extract multiple sources: {e}")
            return results
    
    def get_best_description(self, html_content: str, url: str = "", 
                           existing_description: str = "") -> MetaDescriptionResult:
        """Get the best description considering existing content"""
        
        # If we have a good existing description, prefer it
        if existing_description:
            existing_quality = self._calculate_quality_score(existing_description, 'existing')
            if existing_quality > 0.7:
                return MetaDescriptionResult(
                    description=existing_description,
                    source='existing',
                    confidence=0.95,
                    length=len(existing_description),
                    quality_score=existing_quality,
                    fallback_used=False
                )
        
        # Extract all available descriptions
        candidates = self.extract_multiple_sources(html_content, url)
        
        if not candidates:
            return MetaDescriptionResult()
        
        # If we have existing description, compare with best candidate
        if existing_description and candidates:
            best_candidate = candidates[0]
            existing_quality = self._calculate_quality_score(existing_description, 'existing')
            
            # Use existing if it's significantly better or comparable
            if existing_quality >= best_candidate.quality_score - 0.1:
                return MetaDescriptionResult(
                    description=existing_description,
                    source='existing',
                    confidence=0.9,
                    length=len(existing_description),
                    quality_score=existing_quality,
                    fallback_used=False
                )
        
        # Return best candidate
        return candidates[0]
    
    def validate_description(self, description: str) -> Tuple[bool, List[str]]:
        """Validate description quality and return issues"""
        
        issues = []
        
        if not description:
            issues.append("Description is empty")
            return False, issues
        
        # Length validation
        if len(description) < self.min_length:
            issues.append(f"Description too short (minimum {self.min_length} characters)")
        
        if len(description) > self.max_length:
            issues.append(f"Description too long (maximum {self.max_length} characters)")
        
        # Generic pattern check
        for pattern in self.generic_patterns:
            if re.match(pattern, description.lower()):
                issues.append(f"Description matches generic pattern: {pattern}")
        
        # Quality checks
        if not re.search(r'[.!?]$', description):
            issues.append("Description doesn't end with proper punctuation")
        
        words = description.split()
        if len(words) < 5:
            issues.append("Description has too few words")
        
        # Check for repetitive content
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        repeated_words = [word for word, count in word_counts.items() if count > 3 and len(word) > 3]
        if repeated_words:
            issues.append(f"Repetitive words found: {', '.join(repeated_words)}")
        
        return len(issues) == 0, issues
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics and configuration"""
        
        return {
            'meta_selectors': len(self.meta_selectors),
            'min_length': self.min_length,
            'max_length': self.max_length,
            'ideal_length_range': f"{self.ideal_length_min}-{self.ideal_length_max}",
            'generic_patterns': len(self.generic_patterns),
            'supported_sources': [source for _, source in self.meta_selectors] + ['first_paragraph']
        }