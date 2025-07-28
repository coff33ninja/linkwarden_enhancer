"""
Title Enhancement System - Dedicated title improvement using existing scrapers

This module provides comprehensive title enhancement capabilities by orchestrating
existing web scrapers and implementing quality assessment algorithms. It focuses
specifically on improving bookmark titles through scraping, cleaning, and quality scoring.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    from enhancement.link_enhancement_engine import LinkEnhancementEngine
    from utils.logging_utils import get_logger
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available for TitleEnhancer: {e}")
    COMPONENTS_AVAILABLE = False
    
    # Mock components for testing
    class MockScrapingResult:
        def __init__(self):
            self.success = False
            self.title = None
            self.error_message = None
    
    class MockLinkEnhancementEngine:
        def enhance_bookmark(self, url):
            return MockScrapingResult()
    
    LinkEnhancementEngine = MockLinkEnhancementEngine
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger(__name__)


@dataclass
class TitleQualityScore:
    """Comprehensive title quality assessment"""
    length_score: float = 0.0
    descriptiveness_score: float = 0.0
    generic_penalty: float = 0.0
    url_similarity_penalty: float = 0.0
    formatting_score: float = 0.0
    overall_score: float = 0.0
    
    def calculate_overall(self) -> float:
        """Calculate weighted overall quality score"""
        self.overall_score = (
            self.length_score * 0.25 +
            self.descriptiveness_score * 0.35 +
            (1 - self.generic_penalty) * 0.20 +
            (1 - self.url_similarity_penalty) * 0.10 +
            self.formatting_score * 0.10
        )
        return self.overall_score


@dataclass
class TitleEnhancementResult:
    """Result of title enhancement operation"""
    original_title: str
    enhanced_title: Optional[str]
    quality_improvement: float
    enhancement_method: str  # 'scraped', 'cleaned', 'generated', 'unchanged'
    confidence_score: float
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class TitleEnhancementConfig:
    """Configuration for title enhancement"""
    # Quality thresholds
    min_quality_threshold: float = 0.3
    improvement_threshold: float = 0.2
    
    # Length constraints
    min_title_length: int = 5
    max_title_length: int = 100
    optimal_length_range: Tuple[int, int] = (20, 80)
    
    # Processing settings
    enable_scraping: bool = True
    enable_cleaning: bool = True
    enable_generation: bool = True
    scraping_timeout: int = 10
    
    # Fallback settings
    fallback_to_url_analysis: bool = True
    preserve_user_titles: bool = True
    
    # Cache settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24


class TitleQualityAssessor:
    """Advanced title quality assessment system"""
    
    def __init__(self):
        """Initialize quality assessor with patterns and rules"""
        # Generic title patterns (low quality indicators)
        self.generic_patterns = [
            r'^untitled\d*$',
            r'^page\d*$',
            r'^document\d*$',
            r'^home\d*$',
            r'^index\d*$',
            r'^default\d*$',
            r'^\d+$',
            r'^[a-z]+\.[a-z]+$',  # domain.com
            r'^new\s+(page|document|tab)$',
            r'^loading\.\.\.?$',
            r'^please\s+wait\.\.\.?$',
            r'^error\d*$',
            r'^not\s+found$',
            r'^404$',
            r'^403$',
            r'^500$'
        ]
        
        # Site name patterns (should be removed)
        self.site_name_patterns = [
            r'\s*[-|:•]\s*[^-|:•]+\s*$',  # " - Site Name", " | Site Name", etc.
            r'\s*—\s*[^—]+\s*$',          # " — Site Name"
            r'\s*::\s*[^:]+\s*$',         # " :: Site Name"
            r'\s*\|\s*[^|]+\s*$',         # " | Site Name"
            r'\s*•\s*[^•]+\s*$',          # " • Site Name"
        ]
        
        # Quality indicators (positive signals)
        self.quality_indicators = [
            r'\b(guide|tutorial|how\s+to|tips?|tricks?)\b',
            r'\b(complete|comprehensive|ultimate|definitive)\b',
            r'\b(introduction|getting\s+started|beginner)\b',
            r'\b(advanced|expert|professional|master)\b',
            r'\b(review|comparison|vs\.?|versus)\b',
            r'\b(best|top\s+\d+|essential|must-have)\b',
            r'\b(free|open\s+source|download)\b',
            r'\b(2024|2023|latest|new|updated)\b'
        ]
        
        # Descriptive word patterns
        self.descriptive_words = [
            r'\b(analysis|overview|summary|explanation)\b',
            r'\b(features|benefits|advantages|pros|cons)\b',
            r'\b(implementation|development|design|architecture)\b',
            r'\b(performance|optimization|efficiency|speed)\b',
            r'\b(security|privacy|safety|protection)\b',
            r'\b(integration|compatibility|support|documentation)\b'
        ]
    
    def assess_title_quality(self, title: str, url: str = "") -> TitleQualityScore:
        """
        Perform comprehensive title quality assessment.
        
        Args:
            title: The title to assess
            url: Optional URL for context
            
        Returns:
            TitleQualityScore with detailed scoring breakdown
        """
        if not title:
            return TitleQualityScore()
        
        score = TitleQualityScore()
        title_lower = title.lower().strip()
        
        # 1. Length scoring
        score.length_score = self._assess_length_quality(title)
        
        # 2. Descriptiveness scoring
        score.descriptiveness_score = self._assess_descriptiveness(title_lower)
        
        # 3. Generic pattern penalty
        score.generic_penalty = self._assess_generic_penalty(title_lower)
        
        # 4. URL similarity penalty
        if url:
            score.url_similarity_penalty = self._assess_url_similarity(title_lower, url)
        
        # 5. Formatting quality
        score.formatting_score = self._assess_formatting_quality(title)
        
        # Calculate overall score
        score.calculate_overall()
        
        return score
    
    def _assess_length_quality(self, title: str) -> float:
        """Assess title length quality (0.0 to 1.0)"""
        length = len(title)
        
        if length == 0:
            return 0.0
        elif length < 5:
            return 0.1
        elif length < 10:
            return 0.3
        elif 20 <= length <= 80:  # Optimal range
            return 1.0
        elif 10 <= length < 20:
            return 0.7
        elif 80 < length <= 100:
            return 0.8
        elif length > 100:
            return 0.5  # Too long
        else:
            return 0.6
    
    def _assess_descriptiveness(self, title_lower: str) -> float:
        """Assess how descriptive and informative the title is"""
        score = 0.0
        
        # Check for quality indicators
        quality_matches = sum(1 for pattern in self.quality_indicators 
                            if re.search(pattern, title_lower, re.IGNORECASE))
        score += min(0.4, quality_matches * 0.1)
        
        # Check for descriptive words
        descriptive_matches = sum(1 for pattern in self.descriptive_words 
                                if re.search(pattern, title_lower, re.IGNORECASE))
        score += min(0.3, descriptive_matches * 0.1)
        
        # Word diversity (more unique words = more descriptive)
        words = title_lower.split()
        if len(words) > 1:
            unique_words = len(set(words))
            diversity_score = min(0.3, unique_words / len(words))
            score += diversity_score
        
        return min(1.0, score)
    
    def _assess_generic_penalty(self, title_lower: str) -> float:
        """Calculate penalty for generic/poor quality titles"""
        penalty = 0.0
        
        # Check against generic patterns
        for pattern in self.generic_patterns:
            if re.match(pattern, title_lower, re.IGNORECASE):
                penalty += 0.8
                break
        
        # Additional generic indicators
        generic_words = ['untitled', 'page', 'document', 'home', 'index', 'default']
        if any(word in title_lower for word in generic_words):
            penalty += 0.3
        
        # Very short titles are often generic
        if len(title_lower) < 5:
            penalty += 0.4
        
        # All caps or all lowercase (poor formatting)
        if title_lower.isupper() or (title_lower.islower() and len(title_lower) > 10):
            penalty += 0.2
        
        return min(1.0, penalty)
    
    def _assess_url_similarity(self, title_lower: str, url: str) -> float:
        """Calculate penalty for titles too similar to URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            penalty = 0.0
            
            # Title is just the domain
            if title_lower == domain or title_lower == domain.replace('www.', ''):
                penalty += 0.9
            
            # Title contains mostly domain parts
            domain_parts = domain.replace('www.', '').split('.')
            domain_words = set(part for part in domain_parts if len(part) > 2)
            title_words = set(title_lower.split())
            
            if domain_words and title_words:
                overlap = len(domain_words.intersection(title_words))
                if overlap / len(title_words) > 0.7:
                    penalty += 0.5
            
            # Title is very similar to path
            if path and len(path) > 1:
                path_clean = path.strip('/').replace('-', ' ').replace('_', ' ')
                if title_lower in path_clean or path_clean in title_lower:
                    penalty += 0.3
            
            return min(1.0, penalty)
            
        except Exception:
            return 0.0
    
    def _assess_formatting_quality(self, title: str) -> float:
        """Assess title formatting quality"""
        score = 0.0
        
        # Proper capitalization (title case or sentence case)
        if title and title[0].isupper():
            score += 0.3
        
        # Not all caps or all lowercase
        if not title.isupper() and not (title.islower() and len(title) > 10):
            score += 0.3
        
        # Reasonable punctuation
        if not re.search(r'[!]{2,}|[?]{2,}|[.]{3,}', title):
            score += 0.2
        
        # No excessive whitespace
        if not re.search(r'\s{2,}', title):
            score += 0.2
        
        return score


class TitleCleaner:
    """Advanced title cleaning and standardization"""
    
    def __init__(self):
        """Initialize title cleaner with patterns"""
        # Site name removal patterns
        self.site_name_patterns = [
            r'\s*[-–—]\s*[^-–—]+\s*$',     # " - Site Name", " – Site Name", " — Site Name"
            r'\s*\|\s*[^|]+\s*$',          # " | Site Name"
            r'\s*::\s*[^:]+\s*$',          # " :: Site Name"
            r'\s*•\s*[^•]+\s*$',           # " • Site Name"
            r'\s*›\s*[^›]+\s*$',           # " › Site Name"
            r'\s*»\s*[^»]+\s*$',           # " » Site Name"
            r'\s*>\s*[^>]+\s*$',           # " > Site Name"
        ]
        
        # Common prefixes to remove
        self.prefix_patterns = [
            r'^(welcome\s+to\s+)',
            r'^(home\s*[-|]\s*)',
            r'^(index\s*[-|]\s*)',
            r'^(main\s+page\s*[-|]\s*)',
        ]
        
        # Common suffixes to remove
        self.suffix_patterns = [
            r'(\s*[-|]\s*home\s*)$',
            r'(\s*[-|]\s*main\s+page\s*)$',
            r'(\s*[-|]\s*index\s*)$',
        ]
        
        # Formatting cleanup patterns
        self.cleanup_patterns = [
            (r'\s+', ' '),                 # Multiple spaces to single space
            (r'^\s+|\s+$', ''),            # Trim whitespace
            (r'([.!?])\1+', r'\1'),        # Multiple punctuation to single
            (r'([.!?])\s*$', r'\1'),       # Clean ending punctuation
        ]
    
    def clean_title(self, title: str, url: str = "", preserve_user_content: bool = True) -> str:
        """
        Clean and standardize title.
        
        Args:
            title: Original title to clean
            url: Optional URL for context
            preserve_user_content: Whether to preserve user-generated content
            
        Returns:
            Cleaned and standardized title
        """
        if not title:
            return ""
        
        cleaned = title.strip()
        
        # Skip cleaning if it looks like user-generated content and preservation is enabled
        if preserve_user_content and self._looks_like_user_content(cleaned):
            return cleaned
        
        # Remove site names
        cleaned = self._remove_site_names(cleaned, url)
        
        # Remove common prefixes and suffixes
        cleaned = self._remove_prefixes_suffixes(cleaned)
        
        # Apply formatting cleanup
        cleaned = self._apply_formatting_cleanup(cleaned)
        
        # Smart capitalization
        cleaned = self._apply_smart_capitalization(cleaned)
        
        return cleaned.strip()
    
    def _looks_like_user_content(self, title: str) -> bool:
        """Check if title looks like user-generated content that should be preserved"""
        # Very short titles are likely not user content
        if len(title) < 10:
            return False
        
        # Contains personal indicators
        personal_indicators = [
            r'\bmy\b', r'\bi\b', r'\bme\b', r'\bour\b', r'\bwe\b',
            r'\bpersonal\b', r'\bblog\b', r'\bdiary\b', r'\bjournal\b'
        ]
        
        if any(re.search(pattern, title, re.IGNORECASE) for pattern in personal_indicators):
            return True
        
        # Contains specific dates or numbers that suggest user organization
        if re.search(r'\b(20\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', title, re.IGNORECASE):
            return True
        
        # Contains project or version numbers
        if re.search(r'\bv?\d+\.\d+', title):
            return True
        
        return False
    
    def _remove_site_names(self, title: str, url: str = "") -> str:
        """Remove site names from title"""
        cleaned = title
        
        # Apply site name removal patterns
        for pattern in self.site_name_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # If URL is provided, try to remove domain-specific patterns
        if url:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower().replace('www.', '')
                
                # Remove domain name if it appears at the end
                domain_pattern = rf'\s*[-|:•]\s*{re.escape(domain)}\s*$'
                cleaned = re.sub(domain_pattern, '', cleaned, flags=re.IGNORECASE)
                
                # Remove site name if it's just the domain without extension
                site_name = domain.split('.')[0]
                if len(site_name) > 3:
                    site_pattern = rf'\s*[-|:•]\s*{re.escape(site_name)}\s*$'
                    cleaned = re.sub(site_pattern, '', cleaned, flags=re.IGNORECASE)
                
            except Exception:
                pass
        
        return cleaned.strip()
    
    def _remove_prefixes_suffixes(self, title: str) -> str:
        """Remove common prefixes and suffixes"""
        cleaned = title
        
        # Remove prefixes
        for pattern in self.prefix_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove suffixes
        for pattern in self.suffix_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()
    
    def _apply_formatting_cleanup(self, title: str) -> str:
        """Apply general formatting cleanup"""
        cleaned = title
        
        for pattern, replacement in self.cleanup_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        return cleaned
    
    def _apply_smart_capitalization(self, title: str) -> str:
        """Apply smart capitalization rules"""
        if not title:
            return title
        
        # If title is all caps, convert to title case
        if title.isupper() and len(title) > 5:
            return title.title()
        
        # If title is all lowercase and longer than 10 chars, apply sentence case
        if title.islower() and len(title) > 10:
            return title.capitalize()
        
        # If first character is lowercase, capitalize it
        if title[0].islower():
            return title[0].upper() + title[1:]
        
        return title


class TitleGenerator:
    """Generate titles when scraping fails or returns poor results"""
    
    def __init__(self):
        """Initialize title generator"""
        self.url_patterns = {
            'github.com': self._generate_github_title,
            'stackoverflow.com': self._generate_stackoverflow_title,
            'reddit.com': self._generate_reddit_title,
            'youtube.com': self._generate_youtube_title,
            'medium.com': self._generate_medium_title,
        }
    
    def generate_title_from_url(self, url: str) -> Optional[str]:
        """
        Generate a meaningful title from URL structure.
        
        Args:
            url: URL to analyze
            
        Returns:
            Generated title or None if generation fails
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Try domain-specific generators
            for pattern, generator in self.url_patterns.items():
                if pattern in domain:
                    title = generator(parsed)
                    if title:
                        return title
            
            # Fallback to generic URL analysis
            return self._generate_generic_title(parsed)
            
        except Exception as e:
            logger.warning(f"Title generation failed for {url}: {e}")
            return None
    
    def _generate_github_title(self, parsed) -> Optional[str]:
        """Generate title for GitHub URLs"""
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) >= 2:
            owner, repo = path_parts[0], path_parts[1]
            
            if len(path_parts) == 2:
                return f"{owner}/{repo} - GitHub Repository"
            elif len(path_parts) > 2:
                section = path_parts[2]
                if section == 'issues':
                    return f"{owner}/{repo} Issues - GitHub"
                elif section == 'pulls':
                    return f"{owner}/{repo} Pull Requests - GitHub"
                elif section == 'wiki':
                    return f"{owner}/{repo} Wiki - GitHub"
                else:
                    return f"{owner}/{repo} - GitHub"
        
        return "GitHub Repository"
    
    def _generate_stackoverflow_title(self, parsed) -> Optional[str]:
        """Generate title for Stack Overflow URLs"""
        path = parsed.path.lower()
        
        if '/questions/' in path:
            return "Stack Overflow Question"
        elif '/tags/' in path:
            tag = parsed.path.split('/tags/')[-1].split('/')[0]
            return f"Stack Overflow - {tag} Questions"
        elif '/users/' in path:
            return "Stack Overflow User Profile"
        
        return "Stack Overflow"
    
    def _generate_reddit_title(self, parsed) -> Optional[str]:
        """Generate title for Reddit URLs"""
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) >= 2 and path_parts[0] == 'r':
            subreddit = path_parts[1]
            if len(path_parts) > 4 and path_parts[2] == 'comments':
                return f"Reddit Post - r/{subreddit}"
            else:
                return f"r/{subreddit} - Reddit"
        
        return "Reddit"
    
    def _generate_youtube_title(self, parsed) -> Optional[str]:
        """Generate title for YouTube URLs"""
        if '/watch' in parsed.path:
            return "YouTube Video"
        elif '/channel/' in parsed.path or '/c/' in parsed.path or '/user/' in parsed.path:
            return "YouTube Channel"
        elif '/playlist' in parsed.path:
            return "YouTube Playlist"
        
        return "YouTube"
    
    def _generate_medium_title(self, parsed) -> Optional[str]:
        """Generate title for Medium URLs"""
        path_parts = parsed.path.strip('/').split('/')
        
        if len(path_parts) >= 1:
            if path_parts[0].startswith('@'):
                author = path_parts[0][1:]
                return f"Medium Article by {author}"
            elif len(path_parts) > 1:
                return "Medium Article"
        
        return "Medium"
    
    def _generate_generic_title(self, parsed) -> Optional[str]:
        """Generate generic title from URL structure"""
        domain = parsed.netloc.replace('www.', '')
        path = parsed.path.strip('/')
        
        if not path:
            return f"{domain.title()} - Home"
        
        # Convert path to readable title
        path_parts = path.split('/')
        if path_parts:
            # Take the last meaningful part
            last_part = path_parts[-1]
            
            # Remove file extensions
            if '.' in last_part:
                last_part = last_part.split('.')[0]
            
            # Convert dashes and underscores to spaces
            readable = last_part.replace('-', ' ').replace('_', ' ')
            
            # Capitalize
            readable = readable.title()
            
            if readable and len(readable) > 2:
                return f"{readable} - {domain.title()}"
        
        return f"{domain.title()}"


class TitleEnhancer:
    """
    Main title enhancement system that orchestrates existing scrapers
    for comprehensive title improvement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize title enhancer with configuration"""
        self.config = config
        self.enhancement_config = TitleEnhancementConfig()
        
        # Initialize components
        if COMPONENTS_AVAILABLE:
            self.link_enhancer = LinkEnhancementEngine(config)
        else:
            self.link_enhancer = LinkEnhancementEngine()
        
        self.quality_assessor = TitleQualityAssessor()
        self.title_cleaner = TitleCleaner()
        self.title_generator = TitleGenerator()
        
        # Cache for title enhancements
        self.title_cache = {}
        
        logger.info("TitleEnhancer initialized")
    
    async def enhance_title(self, 
                          url: str, 
                          current_title: str = "", 
                          force_enhancement: bool = False) -> TitleEnhancementResult:
        """
        Enhance a single bookmark title.
        
        Args:
            url: URL of the bookmark
            current_title: Current title (if any)
            force_enhancement: Force enhancement even if current title is good
            
        Returns:
            TitleEnhancementResult with enhancement details
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{url}:{current_title}"
            if self.enhancement_config.enable_caching and cache_key in self.title_cache:
                cached_result = self.title_cache[cache_key]
                logger.debug(f"Using cached title for {url}")
                return cached_result
            
            # Assess current title quality
            current_quality = self.quality_assessor.assess_title_quality(current_title, url)
            
            # Skip enhancement if current title is good enough and not forced
            if (not force_enhancement and 
                current_quality.overall_score >= self.enhancement_config.min_quality_threshold):
                
                result = TitleEnhancementResult(
                    original_title=current_title,
                    enhanced_title=current_title,
                    quality_improvement=0.0,
                    enhancement_method='unchanged',
                    confidence_score=current_quality.overall_score,
                    processing_time=time.time() - start_time
                )
                
                # Cache result
                if self.enhancement_config.enable_caching:
                    self.title_cache[cache_key] = result
                
                return result
            
            # Try to get better title through scraping
            enhanced_title = None
            enhancement_method = 'unchanged'
            
            if self.enhancement_config.enable_scraping:
                scraped_title = await self._scrape_title(url)
                if scraped_title:
                    enhanced_title = scraped_title
                    enhancement_method = 'scraped'
            
            # If scraping failed or didn't improve quality, try cleaning current title
            if (not enhanced_title and 
                current_title and 
                self.enhancement_config.enable_cleaning):
                
                cleaned_title = self.title_cleaner.clean_title(
                    current_title, 
                    url, 
                    self.enhancement_config.preserve_user_titles
                )
                
                if cleaned_title != current_title:
                    cleaned_quality = self.quality_assessor.assess_title_quality(cleaned_title, url)
                    if cleaned_quality.overall_score > current_quality.overall_score:
                        enhanced_title = cleaned_title
                        enhancement_method = 'cleaned'
            
            # If still no improvement, try generating from URL
            if (not enhanced_title and 
                self.enhancement_config.enable_generation and
                self.enhancement_config.fallback_to_url_analysis):
                
                generated_title = self.title_generator.generate_title_from_url(url)
                if generated_title:
                    generated_quality = self.quality_assessor.assess_title_quality(generated_title, url)
                    if generated_quality.overall_score > current_quality.overall_score:
                        enhanced_title = generated_title
                        enhancement_method = 'generated'
            
            # Calculate final results
            if enhanced_title:
                enhanced_quality = self.quality_assessor.assess_title_quality(enhanced_title, url)
                quality_improvement = enhanced_quality.overall_score - current_quality.overall_score
                confidence_score = enhanced_quality.overall_score
            else:
                enhanced_title = current_title
                quality_improvement = 0.0
                confidence_score = current_quality.overall_score
            
            result = TitleEnhancementResult(
                original_title=current_title,
                enhanced_title=enhanced_title,
                quality_improvement=quality_improvement,
                enhancement_method=enhancement_method,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time
            )
            
            # Cache result
            if self.enhancement_config.enable_caching:
                self.title_cache[cache_key] = result
            
            logger.debug(f"Title enhancement completed for {url}: {enhancement_method}")
            return result
            
        except Exception as e:
            error_msg = f"Title enhancement failed: {e}"
            logger.error(f"Error enhancing title for {url}: {error_msg}")
            
            return TitleEnhancementResult(
                original_title=current_title,
                enhanced_title=current_title,
                quality_improvement=0.0,
                enhancement_method='error',
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                error_message=error_msg
            )
    
    async def enhance_titles_batch(self, 
                                 bookmarks: List[Dict[str, Any]], 
                                 force_enhancement: bool = False) -> List[TitleEnhancementResult]:
        """
        Enhance titles for multiple bookmarks.
        
        Args:
            bookmarks: List of bookmark dictionaries
            force_enhancement: Force enhancement even if current titles are good
            
        Returns:
            List of TitleEnhancementResult objects
        """
        results = []
        
        for bookmark in bookmarks:
            url = bookmark.get('url', '')
            current_title = bookmark.get('name', '')
            
            if url:
                result = await self.enhance_title(url, current_title, force_enhancement)
                results.append(result)
            else:
                # Create error result for invalid bookmark
                results.append(TitleEnhancementResult(
                    original_title=current_title,
                    enhanced_title=current_title,
                    quality_improvement=0.0,
                    enhancement_method='error',
                    confidence_score=0.0,
                    processing_time=0.0,
                    error_message="No URL provided"
                ))
        
        return results
    
    async def _scrape_title(self, url: str) -> Optional[str]:
        """Scrape title using existing link enhancement engine"""
        try:
            # Use existing link enhancement engine
            scraping_result = self.link_enhancer.enhance_bookmark(url)
            
            if scraping_result.success and scraping_result.title:
                # Clean the scraped title
                cleaned_title = self.title_cleaner.clean_title(
                    scraping_result.title, 
                    url, 
                    preserve_user_content=False  # Scraped content can be cleaned
                )
                
                return cleaned_title if cleaned_title else scraping_result.title
            
            return None
            
        except Exception as e:
            logger.warning(f"Title scraping failed for {url}: {e}")
            return None
    
    def assess_title_quality(self, title: str, url: str = "") -> TitleQualityScore:
        """
        Assess the quality of a title.
        
        Args:
            title: Title to assess
            url: Optional URL for context
            
        Returns:
            TitleQualityScore with detailed assessment
        """
        return self.quality_assessor.assess_title_quality(title, url)
    
    def clean_title(self, title: str, url: str = "") -> str:
        """
        Clean and standardize a title.
        
        Args:
            title: Title to clean
            url: Optional URL for context
            
        Returns:
            Cleaned title
        """
        return self.title_cleaner.clean_title(title, url, self.enhancement_config.preserve_user_titles)
    
    def generate_title_from_url(self, url: str) -> Optional[str]:
        """
        Generate a title from URL structure.
        
        Args:
            url: URL to analyze
            
        Returns:
            Generated title or None
        """
        return self.title_generator.generate_title_from_url(url)
    
    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get title enhancement statistics"""
        return {
            'cache_size': len(self.title_cache),
            'components_available': COMPONENTS_AVAILABLE,
            'configuration': {
                'min_quality_threshold': self.enhancement_config.min_quality_threshold,
                'enable_scraping': self.enhancement_config.enable_scraping,
                'enable_cleaning': self.enhancement_config.enable_cleaning,
                'enable_generation': self.enhancement_config.enable_generation,
                'preserve_user_titles': self.enhancement_config.preserve_user_titles,
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the title enhancement cache"""
        self.title_cache.clear()
        logger.info("Title enhancement cache cleared")