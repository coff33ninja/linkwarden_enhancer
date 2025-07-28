"""
Auto-Tagging System - Comprehensive tag generation using existing AI systems

This module provides intelligent auto-tagging capabilities by orchestrating existing
AI systems including TagPredictor, SpecializedAnalyzers, ContentAnalyzer, and
SmartDictionaryManager. It combines multiple tagging approaches for comprehensive
and accurate tag generation.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse
import re

try:
    from ai.tag_predictor import TagPredictor, TagPrediction
    from ai.content_analyzer import ContentAnalyzer
    from ai.specialized_analyzers import GamingAnalyzer, DevelopmentAnalyzer, SpecializedAnalysisResult
    from intelligence.dictionary_manager import SmartDictionaryManager
    from utils.logging_utils import get_logger
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available for AutoTagger: {e}")
    COMPONENTS_AVAILABLE = False
    
    # Mock components for testing
    @dataclass
    class MockTagPrediction:
        tag: str
        confidence: float
        source: str
    
    @dataclass 
    class MockSpecializedResult:
        specialized_tags: List[str] = field(default_factory=list)
        confidence_score: float = 0.5
        domain: str = "general"
    
    class MockComponent:
        def __init__(self, *args, **kwargs):
            self.is_trained = False
        
        def predict_tags(self, *args, **kwargs):
            return []
        
        def analyze_content(self, *args, **kwargs):
            return type('MockResult', (), {'keywords': []})()
        
        def can_analyze(self, *args, **kwargs):
            return False
        
        def analyze(self, *args, **kwargs):
            return MockSpecializedResult()
        
        def suggest_tags_for_bookmark(self, *args, **kwargs):
            return []
    
    TagPredictor = MockComponent
    ContentAnalyzer = MockComponent
    GamingAnalyzer = MockComponent
    DevelopmentAnalyzer = MockComponent
    SmartDictionaryManager = MockComponent
    TagPrediction = MockTagPrediction
    SpecializedAnalysisResult = MockSpecializedResult
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger(__name__)


@dataclass
class TagSuggestion:
    """Individual tag suggestion with metadata"""
    tag: str
    confidence: float
    source: str  # 'ml_model', 'specialized', 'content_analysis', 'url_analysis', 'dictionary'
    reasoning: str = ""
    domain_specific: bool = False


@dataclass
class TaggingResult:
    """Result of auto-tagging operation"""
    original_tags: List[str]
    suggested_tags: List[TagSuggestion]
    final_tags: List[str]
    tags_added: int
    confidence_scores: Dict[str, float]
    processing_time: float
    sources_used: List[str]
    error_message: Optional[str] = None


@dataclass
class AutoTaggingConfig:
    """Configuration for auto-tagging system"""
    # Confidence thresholds
    min_confidence_threshold: float = 0.3
    high_confidence_threshold: float = 0.7
    specialized_confidence_boost: float = 0.1
    
    # Tag limits
    max_tags_per_bookmark: int = 10
    max_new_tags: int = 8
    min_tag_length: int = 2
    max_tag_length: int = 30
    
    # Source weights
    ml_model_weight: float = 1.0
    specialized_analyzer_weight: float = 1.2
    content_analysis_weight: float = 0.8
    url_analysis_weight: float = 0.6
    dictionary_weight: float = 0.9
    
    # Processing options
    enable_ml_tagging: bool = True
    enable_specialized_analysis: bool = True
    enable_content_analysis: bool = True
    enable_url_analysis: bool = True
    enable_dictionary_suggestions: bool = True
    
    # Quality control
    enable_tag_deduplication: bool = True
    enable_similarity_merging: bool = True
    enable_generic_filtering: bool = True
    similarity_threshold: float = 0.8
    
    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24


class URLAnalyzer:
    """Extract tags from URL structure and patterns"""
    
    def __init__(self):
        """Initialize URL analyzer with domain patterns"""
        # Domain-specific tag mappings
        self.domain_tags = {
            'github.com': ['development', 'code', 'repository', 'programming'],
            'gitlab.com': ['development', 'code', 'repository', 'programming'],
            'stackoverflow.com': ['programming', 'development', 'help', 'q&a'],
            'stackexchange.com': ['programming', 'development', 'help', 'q&a'],
            'youtube.com': ['video', 'entertainment', 'media'],
            'reddit.com': ['social', 'discussion', 'community'],
            'medium.com': ['article', 'blog', 'writing'],
            'dev.to': ['development', 'programming', 'blog'],
            'hackernews.com': ['news', 'technology', 'programming'],
            'producthunt.com': ['product', 'startup', 'technology'],
            'dribbble.com': ['design', 'ui', 'ux', 'creative'],
            'behance.net': ['design', 'creative', 'portfolio'],
            'figma.com': ['design', 'ui', 'ux', 'tool'],
            'notion.so': ['productivity', 'tool', 'organization'],
            'trello.com': ['productivity', 'project-management', 'tool'],
            'slack.com': ['communication', 'team', 'productivity'],
            'discord.com': ['communication', 'gaming', 'community'],
            'twitch.tv': ['gaming', 'streaming', 'entertainment'],
            'steam': ['gaming', 'games', 'platform'],
            'itch.io': ['gaming', 'indie', 'games'],
            'docs.': ['documentation', 'reference'],
            'wiki': ['wiki', 'reference', 'information'],
        }
        
        # Path-based tag patterns
        self.path_patterns = {
            r'/docs?/': ['documentation', 'reference'],
            r'/api/': ['api', 'development', 'reference'],
            r'/tutorial/': ['tutorial', 'learning', 'guide'],
            r'/guide/': ['guide', 'tutorial', 'help'],
            r'/blog/': ['blog', 'article', 'writing'],
            r'/news/': ['news', 'article'],
            r'/download/': ['download', 'software'],
            r'/pricing/': ['pricing', 'business', 'saas'],
            r'/about/': ['about', 'company', 'information'],
            r'/contact/': ['contact', 'support'],
            r'/support/': ['support', 'help'],
            r'/faq/': ['faq', 'help', 'support'],
            r'/forum/': ['forum', 'discussion', 'community'],
            r'/community/': ['community', 'discussion'],
            r'/dashboard/': ['dashboard', 'app', 'tool'],
            r'/admin/': ['admin', 'management', 'tool'],
            r'/settings/': ['settings', 'configuration'],
            r'/profile/': ['profile', 'user', 'account'],
            r'/login/': ['login', 'authentication'],
            r'/signup/': ['signup', 'registration'],
        }
        
        # Technology detection patterns
        self.tech_patterns = {
            r'\b(react|vue|angular|svelte)\b': 'frontend',
            r'\b(node|express|fastapi|django|flask)\b': 'backend',
            r'\b(python|javascript|java|go|rust|php)\b': 'programming',
            r'\b(docker|kubernetes|k8s)\b': 'devops',
            r'\b(aws|azure|gcp|cloud)\b': 'cloud',
            r'\b(mysql|postgres|mongodb|redis)\b': 'database',
            r'\b(git|github|gitlab)\b': 'version-control',
            r'\b(ai|ml|machine-learning|neural)\b': 'artificial-intelligence',
            r'\b(blockchain|crypto|bitcoin|ethereum)\b': 'blockchain',
            r'\b(mobile|ios|android|flutter|react-native)\b': 'mobile',
        }
    
    def analyze_url(self, url: str) -> List[TagSuggestion]:
        """
        Extract tags from URL structure.
        
        Args:
            url: URL to analyze
            
        Returns:
            List of TagSuggestion objects
        """
        suggestions = []
        
        try:
            parsed = urlparse(url.lower())
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            
            # Domain-based tags
            domain_suggestions = self._analyze_domain(domain)
            suggestions.extend(domain_suggestions)
            
            # Path-based tags
            path_suggestions = self._analyze_path(path)
            suggestions.extend(path_suggestions)
            
            # Technology detection in URL
            tech_suggestions = self._analyze_technology_patterns(url)
            suggestions.extend(tech_suggestions)
            
            # Query parameter analysis
            query_suggestions = self._analyze_query_parameters(query)
            suggestions.extend(query_suggestions)
            
        except Exception as e:
            logger.warning(f"URL analysis failed for {url}: {e}")
        
        return suggestions
    
    def _analyze_domain(self, domain: str) -> List[TagSuggestion]:
        """Analyze domain for tag suggestions"""
        suggestions = []
        
        # Direct domain matches
        for domain_pattern, tags in self.domain_tags.items():
            if domain_pattern in domain:
                for tag in tags:
                    suggestions.append(TagSuggestion(
                        tag=tag,
                        confidence=0.8,
                        source='url_analysis',
                        reasoning=f"Domain pattern: {domain_pattern}",
                        domain_specific=True
                    ))
                break
        
        # Subdomain analysis
        if domain.startswith('docs.') or domain.startswith('api.'):
            suggestions.append(TagSuggestion(
                tag='documentation' if 'docs.' in domain else 'api',
                confidence=0.9,
                source='url_analysis',
                reasoning="Subdomain pattern",
                domain_specific=True
            ))
        
        return suggestions
    
    def _analyze_path(self, path: str) -> List[TagSuggestion]:
        """Analyze URL path for tag suggestions"""
        suggestions = []
        
        for pattern, tags in self.path_patterns.items():
            if re.search(pattern, path, re.IGNORECASE):
                for tag in (tags if isinstance(tags, list) else [tags]):
                    suggestions.append(TagSuggestion(
                        tag=tag,
                        confidence=0.7,
                        source='url_analysis',
                        reasoning=f"Path pattern: {pattern}"
                    ))
        
        return suggestions
    
    def _analyze_technology_patterns(self, url: str) -> List[TagSuggestion]:
        """Detect technology mentions in URL"""
        suggestions = []
        
        for pattern, category in self.tech_patterns.items():
            matches = re.findall(pattern, url, re.IGNORECASE)
            for match in matches:
                suggestions.append(TagSuggestion(
                    tag=match.lower(),
                    confidence=0.6,
                    source='url_analysis',
                    reasoning=f"Technology pattern: {match}"
                ))
                
                # Add category tag
                suggestions.append(TagSuggestion(
                    tag=category,
                    confidence=0.5,
                    source='url_analysis',
                    reasoning=f"Technology category: {category}"
                ))
        
        return suggestions
    
    def _analyze_query_parameters(self, query: str) -> List[TagSuggestion]:
        """Analyze query parameters for tag hints"""
        suggestions = []
        
        if not query:
            return suggestions
        
        # Common parameter patterns
        param_patterns = {
            r'tag=([^&]+)': 'explicit_tag',
            r'category=([^&]+)': 'category',
            r'type=([^&]+)': 'type',
            r'lang=([^&]+)': 'language',
            r'framework=([^&]+)': 'framework',
        }
        
        for pattern, tag_type in param_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                # Clean and validate the parameter value
                tag_value = match.replace('%20', ' ').replace('+', ' ').strip()
                if len(tag_value) > 1 and len(tag_value) < 20:
                    suggestions.append(TagSuggestion(
                        tag=tag_value.lower(),
                        confidence=0.8,
                        source='url_analysis',
                        reasoning=f"Query parameter: {tag_type}"
                    ))
        
        return suggestions


class TagQualityController:
    """Control tag quality through deduplication, merging, and filtering"""
    
    def __init__(self, config: AutoTaggingConfig):
        """Initialize quality controller"""
        self.config = config
        
        # Generic tags to filter out (too broad to be useful)
        self.generic_tags = {
            'web', 'site', 'website', 'page', 'link', 'url', 'internet',
            'online', 'digital', 'content', 'information', 'data', 'stuff',
            'thing', 'item', 'resource', 'material', 'general', 'misc',
            'other', 'various', 'multiple', 'different', 'new', 'old',
            'good', 'bad', 'best', 'top', 'popular', 'common', 'basic'
        }
        
        # Tag similarity patterns for merging
        self.similarity_patterns = [
            # Programming languages
            (['js', 'javascript'], 'javascript'),
            (['py', 'python'], 'python'),
            (['ts', 'typescript'], 'typescript'),
            (['cpp', 'c++', 'cplusplus'], 'c++'),
            (['csharp', 'c#'], 'c#'),
            
            # Frameworks
            (['reactjs', 'react.js'], 'react'),
            (['vuejs', 'vue.js'], 'vue'),
            (['nodejs', 'node.js'], 'node'),
            (['nextjs', 'next.js'], 'next'),
            
            # Technologies
            (['ai', 'artificial-intelligence', 'machine-learning', 'ml'], 'artificial-intelligence'),
            (['devops', 'dev-ops'], 'devops'),
            (['frontend', 'front-end'], 'frontend'),
            (['backend', 'back-end'], 'backend'),
            (['fullstack', 'full-stack'], 'fullstack'),
            
            # Platforms
            (['github', 'gh'], 'github'),
            (['stackoverflow', 'stack-overflow', 'so'], 'stackoverflow'),
            
            # Content types
            (['tutorial', 'guide', 'how-to'], 'tutorial'),
            (['documentation', 'docs', 'reference'], 'documentation'),
            (['article', 'blog-post', 'post'], 'article'),
        ]
    
    def process_tag_suggestions(self, 
                              suggestions: List[TagSuggestion], 
                              existing_tags: List[str]) -> List[str]:
        """
        Process tag suggestions through quality control pipeline.
        
        Args:
            suggestions: List of tag suggestions
            existing_tags: Already existing tags
            
        Returns:
            List of final processed tags
        """
        # Convert existing tags to lowercase for comparison
        existing_lower = {tag.lower() for tag in existing_tags}
        
        # Step 1: Filter by confidence threshold
        filtered_suggestions = [
            s for s in suggestions 
            if s.confidence >= self.config.min_confidence_threshold
        ]
        
        # Step 2: Remove duplicates with existing tags
        new_suggestions = [
            s for s in filtered_suggestions 
            if s.tag.lower() not in existing_lower
        ]
        
        # Step 3: Apply generic tag filtering
        if self.config.enable_generic_filtering:
            new_suggestions = [
                s for s in new_suggestions 
                if s.tag.lower() not in self.generic_tags
            ]
        
        # Step 4: Apply tag deduplication
        if self.config.enable_tag_deduplication:
            new_suggestions = self._deduplicate_suggestions(new_suggestions)
        
        # Step 5: Apply similarity merging
        if self.config.enable_similarity_merging:
            new_suggestions = self._merge_similar_tags(new_suggestions)
        
        # Step 6: Sort by confidence and apply limits
        new_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        # Apply tag limits
        final_suggestions = new_suggestions[:self.config.max_new_tags]
        
        # Extract final tag names
        final_tags = [s.tag for s in final_suggestions]
        
        # Apply length constraints
        final_tags = [
            tag for tag in final_tags 
            if self.config.min_tag_length <= len(tag) <= self.config.max_tag_length
        ]
        
        return final_tags
    
    def _deduplicate_suggestions(self, suggestions: List[TagSuggestion]) -> List[TagSuggestion]:
        """Remove duplicate tag suggestions, keeping highest confidence"""
        tag_map = {}
        
        for suggestion in suggestions:
            tag_lower = suggestion.tag.lower()
            if tag_lower not in tag_map or suggestion.confidence > tag_map[tag_lower].confidence:
                tag_map[tag_lower] = suggestion
        
        return list(tag_map.values())
    
    def _merge_similar_tags(self, suggestions: List[TagSuggestion]) -> List[TagSuggestion]:
        """Merge similar tags based on predefined patterns"""
        # Create mapping from suggestion tags to canonical forms
        tag_to_canonical = {}
        suggestion_map = {s.tag.lower(): s for s in suggestions}
        
        # Apply similarity patterns
        for similar_tags, canonical in self.similarity_patterns:
            found_tags = [tag for tag in similar_tags if tag in suggestion_map]
            
            if found_tags:
                # Find the suggestion with highest confidence
                best_suggestion = max(
                    (suggestion_map[tag] for tag in found_tags),
                    key=lambda x: x.confidence
                )
                
                # Map all similar tags to canonical form
                for tag in found_tags:
                    tag_to_canonical[tag] = TagSuggestion(
                        tag=canonical,
                        confidence=best_suggestion.confidence,
                        source=best_suggestion.source,
                        reasoning=f"Merged from: {', '.join(found_tags)}",
                        domain_specific=best_suggestion.domain_specific
                    )
        
        # Build final list with merged tags
        final_suggestions = []
        processed_tags = set()
        
        for suggestion in suggestions:
            tag_lower = suggestion.tag.lower()
            
            if tag_lower in tag_to_canonical:
                canonical_suggestion = tag_to_canonical[tag_lower]
                canonical_tag = canonical_suggestion.tag.lower()
                
                if canonical_tag not in processed_tags:
                    final_suggestions.append(canonical_suggestion)
                    processed_tags.add(canonical_tag)
            else:
                if tag_lower not in processed_tags:
                    final_suggestions.append(suggestion)
                    processed_tags.add(tag_lower)
        
        return final_suggestions


class AutoTagger:
    """
    Main auto-tagging system that orchestrates existing AI systems
    for comprehensive and intelligent tag generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize auto-tagger with configuration"""
        self.config = config
        self.tagging_config = AutoTaggingConfig()
        
        # Initialize existing AI components
        if COMPONENTS_AVAILABLE:
            self.tag_predictor = TagPredictor(config)
            self.content_analyzer = ContentAnalyzer(config)
            self.smart_dictionary = SmartDictionaryManager(config)
            self.gaming_analyzer = GamingAnalyzer()
            self.development_analyzer = DevelopmentAnalyzer()
        else:
            self.tag_predictor = TagPredictor()
            self.content_analyzer = ContentAnalyzer()
            self.smart_dictionary = SmartDictionaryManager()
            self.gaming_analyzer = GamingAnalyzer()
            self.development_analyzer = DevelopmentAnalyzer()
        
        # Initialize custom components
        self.url_analyzer = URLAnalyzer()
        self.quality_controller = TagQualityController(self.tagging_config)
        
        # Caching
        self.tagging_cache = {}
        
        logger.info("AutoTagger initialized with all AI systems")
    
    async def generate_tags(self, 
                          url: str, 
                          title: str = "", 
                          content: str = "", 
                          existing_tags: List[str] = None) -> TaggingResult:
        """
        Generate comprehensive tags using all available AI systems.
        
        Args:
            url: URL of the bookmark
            title: Title of the bookmark
            content: Content/description of the bookmark
            existing_tags: Already existing tags
            
        Returns:
            TaggingResult with comprehensive tagging information
        """
        if existing_tags is None:
            existing_tags = []
        
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = f"{url}:{title}:{content}:{','.join(sorted(existing_tags))}"
            if self.tagging_config.enable_caching and cache_key in self.tagging_cache:
                cached_result = self.tagging_cache[cache_key]
                logger.debug(f"Using cached tags for {url}")
                return cached_result
            
            all_suggestions = []
            sources_used = []
            
            # 1. ML-based tag prediction
            if self.tagging_config.enable_ml_tagging and hasattr(self.tag_predictor, 'is_trained'):
                ml_suggestions = await self._get_ml_predictions(url, title, content, existing_tags)
                all_suggestions.extend(ml_suggestions)
                if ml_suggestions:
                    sources_used.append('ml_model')
            
            # 2. Specialized domain analysis
            if self.tagging_config.enable_specialized_analysis:
                specialized_suggestions = await self._get_specialized_analysis(url, title, content)
                all_suggestions.extend(specialized_suggestions)
                if specialized_suggestions:
                    sources_used.append('specialized_analyzer')
            
            # 3. Content analysis
            if self.tagging_config.enable_content_analysis:
                content_suggestions = await self._get_content_analysis(title, content, url)
                all_suggestions.extend(content_suggestions)
                if content_suggestions:
                    sources_used.append('content_analysis')
            
            # 4. URL analysis
            if self.tagging_config.enable_url_analysis:
                url_suggestions = self.url_analyzer.analyze_url(url)
                all_suggestions.extend(url_suggestions)
                if url_suggestions:
                    sources_used.append('url_analysis')
            
            # 5. Smart dictionary suggestions
            if self.tagging_config.enable_dictionary_suggestions:
                dictionary_suggestions = await self._get_dictionary_suggestions(url, title, content, existing_tags)
                all_suggestions.extend(dictionary_suggestions)
                if dictionary_suggestions:
                    sources_used.append('dictionary')
            
            # Process suggestions through quality control
            final_tags = self.quality_controller.process_tag_suggestions(all_suggestions, existing_tags)
            
            # Calculate confidence scores
            confidence_scores = {}
            for suggestion in all_suggestions:
                if suggestion.tag in final_tags:
                    confidence_scores[suggestion.tag] = suggestion.confidence
            
            # Create result
            result = TaggingResult(
                original_tags=existing_tags.copy(),
                suggested_tags=all_suggestions,
                final_tags=final_tags,
                tags_added=len(final_tags),
                confidence_scores=confidence_scores,
                processing_time=time.time() - start_time,
                sources_used=sources_used
            )
            
            # Cache result
            if self.tagging_config.enable_caching:
                self.tagging_cache[cache_key] = result
            
            logger.debug(f"Generated {len(final_tags)} tags for {url} using {len(sources_used)} sources")
            return result
            
        except Exception as e:
            error_msg = f"Tag generation failed: {e}"
            logger.error(f"Error generating tags for {url}: {error_msg}")
            
            return TaggingResult(
                original_tags=existing_tags.copy(),
                suggested_tags=[],
                final_tags=[],
                tags_added=0,
                confidence_scores={},
                processing_time=time.time() - start_time,
                sources_used=[],
                error_message=error_msg
            )
    
    async def generate_tags_batch(self, 
                                bookmarks: List[Dict[str, Any]]) -> List[TaggingResult]:
        """
        Generate tags for multiple bookmarks.
        
        Args:
            bookmarks: List of bookmark dictionaries
            
        Returns:
            List of TaggingResult objects
        """
        results = []
        
        for bookmark in bookmarks:
            url = bookmark.get('url', '')
            title = bookmark.get('name', '')
            content = bookmark.get('description', '')
            
            # Extract existing tags
            existing_tags = []
            for tag in bookmark.get('tags', []):
                if isinstance(tag, dict):
                    tag_name = tag.get('name', '')
                else:
                    tag_name = str(tag)
                
                if tag_name:
                    existing_tags.append(tag_name)
            
            if url:
                result = await self.generate_tags(url, title, content, existing_tags)
                results.append(result)
            else:
                # Create error result for invalid bookmark
                results.append(TaggingResult(
                    original_tags=existing_tags,
                    suggested_tags=[],
                    final_tags=[],
                    tags_added=0,
                    confidence_scores={},
                    processing_time=0.0,
                    sources_used=[],
                    error_message="No URL provided"
                ))
        
        return results
    
    async def _get_ml_predictions(self, 
                                url: str, 
                                title: str, 
                                content: str, 
                                existing_tags: List[str]) -> List[TagSuggestion]:
        """Get predictions from ML-based tag predictor"""
        suggestions = []
        
        try:
            if hasattr(self.tag_predictor, 'is_trained') and self.tag_predictor.is_trained:
                predictions = self.tag_predictor.predict_tags(title, content, url, existing_tags)
                
                for prediction in predictions:
                    # Apply weight adjustment
                    adjusted_confidence = prediction.confidence * self.tagging_config.ml_model_weight
                    
                    suggestions.append(TagSuggestion(
                        tag=prediction.tag,
                        confidence=min(1.0, adjusted_confidence),
                        source='ml_model',
                        reasoning=f"ML prediction (source: {prediction.source})"
                    ))
            
        except Exception as e:
            logger.warning(f"ML tag prediction failed: {e}")
        
        return suggestions
    
    async def _get_specialized_analysis(self, 
                                      url: str, 
                                      title: str, 
                                      content: str) -> List[TagSuggestion]:
        """Get tags from specialized domain analyzers"""
        suggestions = []
        
        try:
            # Gaming content analysis
            if self.gaming_analyzer.can_analyze(url, title, content):
                gaming_result = self.gaming_analyzer.analyze(url, title, content)
                
                for tag in gaming_result.specialized_tags:
                    # Apply specialized analyzer weight and boost
                    confidence = (gaming_result.confidence_score * 
                                self.tagging_config.specialized_analyzer_weight + 
                                self.tagging_config.specialized_confidence_boost)
                    
                    suggestions.append(TagSuggestion(
                        tag=tag,
                        confidence=min(1.0, confidence),
                        source='specialized_analyzer',
                        reasoning=f"Gaming analysis (domain: {gaming_result.domain})",
                        domain_specific=True
                    ))
            
            # Development content analysis
            if self.development_analyzer.can_analyze(url, title, content):
                dev_result = self.development_analyzer.analyze(url, title, content)
                
                for tag in dev_result.specialized_tags:
                    confidence = (dev_result.confidence_score * 
                                self.tagging_config.specialized_analyzer_weight + 
                                self.tagging_config.specialized_confidence_boost)
                    
                    suggestions.append(TagSuggestion(
                        tag=tag,
                        confidence=min(1.0, confidence),
                        source='specialized_analyzer',
                        reasoning=f"Development analysis (domain: {dev_result.domain})",
                        domain_specific=True
                    ))
            
        except Exception as e:
            logger.warning(f"Specialized analysis failed: {e}")
        
        return suggestions
    
    async def _get_content_analysis(self, 
                                  title: str, 
                                  content: str, 
                                  url: str) -> List[TagSuggestion]:
        """Get tags from content analysis"""
        suggestions = []
        
        try:
            if hasattr(self.content_analyzer, 'is_trained') and self.content_analyzer.is_trained:
                analysis_result = self.content_analyzer.analyze_content(title, content, url)
                
                # Use keywords as tags
                for keyword in analysis_result.keywords:
                    confidence = 0.6 * self.tagging_config.content_analysis_weight
                    
                    suggestions.append(TagSuggestion(
                        tag=keyword,
                        confidence=confidence,
                        source='content_analysis',
                        reasoning="Content keyword extraction"
                    ))
                
                # Use content type as tag
                if analysis_result.content_type and analysis_result.content_type != 'general':
                    suggestions.append(TagSuggestion(
                        tag=analysis_result.content_type,
                        confidence=0.7 * self.tagging_config.content_analysis_weight,
                        source='content_analysis',
                        reasoning="Content type classification"
                    ))
            
        except Exception as e:
            logger.warning(f"Content analysis failed: {e}")
        
        return suggestions
    
    async def _get_dictionary_suggestions(self, 
                                        url: str, 
                                        title: str, 
                                        content: str, 
                                        existing_tags: List[str]) -> List[TagSuggestion]:
        """Get suggestions from smart dictionary manager"""
        suggestions = []
        
        try:
            dictionary_suggestions = self.smart_dictionary.suggest_tags_for_bookmark(
                url, title, content, existing_tags
            )
            
            for tag, confidence in dictionary_suggestions:
                adjusted_confidence = confidence * self.tagging_config.dictionary_weight
                
                suggestions.append(TagSuggestion(
                    tag=tag,
                    confidence=adjusted_confidence,
                    source='dictionary',
                    reasoning="Smart dictionary suggestion"
                ))
            
        except Exception as e:
            logger.warning(f"Dictionary suggestions failed: {e}")
        
        return suggestions
    
    def get_tagging_stats(self) -> Dict[str, Any]:
        """Get comprehensive tagging statistics"""
        return {
            'cache_size': len(self.tagging_cache),
            'components_available': COMPONENTS_AVAILABLE,
            'ai_components_status': {
                'tag_predictor_trained': getattr(self.tag_predictor, 'is_trained', False),
                'content_analyzer_trained': getattr(self.content_analyzer, 'is_trained', False),
                'smart_dictionary_available': self.smart_dictionary is not None,
                'specialized_analyzers_available': True,
            },
            'configuration': {
                'min_confidence_threshold': self.tagging_config.min_confidence_threshold,
                'max_tags_per_bookmark': self.tagging_config.max_tags_per_bookmark,
                'max_new_tags': self.tagging_config.max_new_tags,
                'enable_ml_tagging': self.tagging_config.enable_ml_tagging,
                'enable_specialized_analysis': self.tagging_config.enable_specialized_analysis,
                'enable_content_analysis': self.tagging_config.enable_content_analysis,
                'enable_url_analysis': self.tagging_config.enable_url_analysis,
                'enable_dictionary_suggestions': self.tagging_config.enable_dictionary_suggestions,
            },
            'quality_control': {
                'enable_tag_deduplication': self.tagging_config.enable_tag_deduplication,
                'enable_similarity_merging': self.tagging_config.enable_similarity_merging,
                'enable_generic_filtering': self.tagging_config.enable_generic_filtering,
                'generic_tags_filtered': len(self.quality_controller.generic_tags),
                'similarity_patterns': len(self.quality_controller.similarity_patterns),
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the tagging cache"""
        self.tagging_cache.clear()
        logger.info("Auto-tagging cache cleared")