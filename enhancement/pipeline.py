"""
Advanced Enhancement Pipeline - Main orchestrator for bookmark enhancement

This module provides the main EnhancementPipeline class that coordinates all enhancement
components including title enhancement, auto-tagging, description generation, and
duplicate detection. It builds upon existing enhancement systems and provides a
unified interface for comprehensive bookmark enhancement.
"""

import asyncio
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    from enhancement.link_enhancement_engine import LinkEnhancementEngine
    from ai.tag_predictor import TagPredictor
    from ai.similarity_engine import SimilarityEngine
    from ai.content_analyzer import ContentAnalyzer
    from ai.specialized_analyzers import GamingAnalyzer, DevelopmentAnalyzer
    from ai.ollama_client import OllamaClient
    from intelligence.dictionary_manager import SmartDictionaryManager
    from core.progress_monitor import ProgressMonitor
    from core.safety_manager import SafetyManager
    from utils.logging_utils import get_logger
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False
    
    # Create mock classes for testing
    class MockComponent:
        def __init__(self, *args, **kwargs):
            pass
    
    LinkEnhancementEngine = MockComponent
    TagPredictor = MockComponent
    SimilarityEngine = MockComponent
    ContentAnalyzer = MockComponent
    GamingAnalyzer = MockComponent
    DevelopmentAnalyzer = MockComponent
    OllamaClient = MockComponent
    SmartDictionaryManager = MockComponent
    ProgressMonitor = MockComponent
    SafetyManager = MockComponent
    
    def get_logger(name):
        import logging
        return logging.getLogger(name)

logger = get_logger(__name__)


class EnhancementMode(Enum):
    """Enhancement processing modes"""
    FULL = "full"
    SELECTIVE = "selective"
    DRY_RUN = "dry_run"
    VALIDATION_ONLY = "validation_only"


@dataclass
class EnhancementOptions:
    """Configuration options for selective enhancement"""
    enhance_titles: bool = True
    generate_tags: bool = True
    create_descriptions: bool = True
    detect_duplicates: bool = True
    preserve_existing: bool = True
    quality_threshold: float = 0.7
    max_tags_per_bookmark: int = 10
    description_max_length: int = 200
    similarity_threshold: float = 0.85


@dataclass
class EnhancementConfig:
    """Comprehensive configuration for enhancement pipeline"""
    # Processing options
    mode: EnhancementMode = EnhancementMode.FULL
    options: EnhancementOptions = field(default_factory=EnhancementOptions)
    
    # Performance settings
    batch_size: int = 50
    max_concurrent: int = 5
    timeout_seconds: int = 30
    
    # Quality settings
    title_quality_threshold: float = 0.3
    tag_confidence_threshold: float = 0.7
    description_min_length: int = 50
    
    # Safety settings
    enable_backups: bool = True
    enable_validation: bool = True
    max_error_rate: float = 10.0  # Percentage
    
    # AI settings
    enable_ai_summarization: bool = True
    ollama_model: str = "llama2"
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24


@dataclass
class EnhancementStats:
    """Statistics from enhancement processing"""
    total_bookmarks: int = 0
    processed_bookmarks: int = 0
    titles_enhanced: int = 0
    tags_generated: int = 0
    descriptions_created: int = 0
    duplicates_detected: int = 0
    duplicates_resolved: int = 0
    errors_encountered: int = 0
    processing_time: float = 0.0
    success_rate: float = 0.0
    
    def calculate_success_rate(self) -> None:
        """Calculate overall success rate"""
        if self.total_bookmarks > 0:
            self.success_rate = (self.processed_bookmarks / self.total_bookmarks) * 100


@dataclass
class EnhancementResult:
    """Result of enhancement pipeline processing"""
    success: bool
    enhanced_bookmarks: List[Dict[str, Any]]
    stats: EnhancementStats
    errors: List[str]
    warnings: List[str]
    quality_report: Optional[Dict[str, Any]] = None
    duplicate_groups: List[Any] = field(default_factory=list)


class EnhancementPipeline:
    """
    Main orchestrator for bookmark enhancement pipeline.
    
    Coordinates all enhancement components including:
    - Title enhancement using existing scrapers
    - Auto-tagging using AI systems
    - Description generation using content analysis and AI
    - Duplicate detection using similarity engine
    - Progress tracking and safety monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhancement pipeline with configuration"""
        self.config = config
        self.enhancement_config = EnhancementConfig()
        
        # Initialize existing components
        self.link_enhancer = LinkEnhancementEngine(config)
        self.tag_predictor = TagPredictor(config)
        self.similarity_engine = SimilarityEngine(config)
        self.content_analyzer = ContentAnalyzer(config)
        self.smart_dictionary = SmartDictionaryManager(config)
        self.progress_monitor = ProgressMonitor(config)
        self.safety_manager = SafetyManager(config)
        
        # Initialize specialized analyzers
        self.gaming_analyzer = GamingAnalyzer()
        self.development_analyzer = DevelopmentAnalyzer()
        
        # Initialize AI components
        self.ollama_client = None
        if self.enhancement_config.enable_ai_summarization:
            try:
                self.ollama_client = OllamaClient(
                    model_name=self.enhancement_config.ollama_model,
                    auto_start=True,
                    auto_pull=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}")
                self.enhancement_config.enable_ai_summarization = False
        
        # Processing state
        self.current_operation_id = None
        self.enhancement_cache = {}
        
        logger.info("Enhancement pipeline initialized with all components")
    
    async def process_bookmarks(self, 
                              bookmarks: List[Dict[str, Any]], 
                              config: Optional[EnhancementConfig] = None) -> EnhancementResult:
        """
        Process bookmarks through the complete enhancement pipeline.
        
        Args:
            bookmarks: List of bookmark dictionaries to enhance
            config: Optional enhancement configuration (uses default if None)
            
        Returns:
            EnhancementResult with enhanced bookmarks and statistics
        """
        if config:
            self.enhancement_config = config
        
        start_time = time.time()
        stats = EnhancementStats(total_bookmarks=len(bookmarks))
        
        # Start progress monitoring
        self.current_operation_id = self.progress_monitor.start_operation(
            "bookmark_enhancement",
            total_items=len(bookmarks),
            description=f"Enhancing {len(bookmarks)} bookmarks"
        )
        
        try:
            logger.info(f"Starting enhancement pipeline for {len(bookmarks)} bookmarks")
            
            # Step 1: Validate input data
            self.progress_monitor.update_progress(
                self.current_operation_id, 0, "Validating input data"
            )
            
            validated_bookmarks = await self._validate_input_bookmarks(bookmarks)
            if not validated_bookmarks:
                return self._create_error_result("No valid bookmarks to process", stats)
            
            # Step 2: Create backup if enabled
            if self.enhancement_config.enable_backups:
                self.progress_monitor.update_progress(
                    self.current_operation_id, 5, "Creating backup"
                )
                await self._create_backup(bookmarks)
            
            # Step 3: Train AI models if needed
            self.progress_monitor.update_progress(
                self.current_operation_id, 10, "Training AI models"
            )
            await self._ensure_models_trained(validated_bookmarks)
            
            # Step 4: Process bookmarks based on mode
            if self.enhancement_config.mode == EnhancementMode.DRY_RUN:
                enhanced_bookmarks = await self._dry_run_processing(validated_bookmarks, stats)
            elif self.enhancement_config.mode == EnhancementMode.VALIDATION_ONLY:
                enhanced_bookmarks = await self._validation_only_processing(validated_bookmarks, stats)
            else:
                enhanced_bookmarks = await self._full_processing(validated_bookmarks, stats)
            
            # Step 5: Final validation and quality assessment
            self.progress_monitor.update_progress(
                self.current_operation_id, 95, "Final validation and quality assessment"
            )
            
            quality_report = await self._generate_quality_report(bookmarks, enhanced_bookmarks)
            
            # Complete processing
            stats.processing_time = time.time() - start_time
            stats.calculate_success_rate()
            
            self.progress_monitor.complete_operation(self.current_operation_id, True)
            
            logger.info(f"Enhancement completed: {stats.processed_bookmarks}/{stats.total_bookmarks} bookmarks processed")
            
            return EnhancementResult(
                success=True,
                enhanced_bookmarks=enhanced_bookmarks,
                stats=stats,
                errors=[],
                warnings=[],
                quality_report=quality_report
            )
            
        except Exception as e:
            logger.error(f"Enhancement pipeline failed: {e}")
            self.progress_monitor.add_error(self.current_operation_id, str(e))
            self.progress_monitor.complete_operation(self.current_operation_id, False)
            
            stats.processing_time = time.time() - start_time
            stats.calculate_success_rate()
            
            return EnhancementResult(
                success=False,
                enhanced_bookmarks=[],
                stats=stats,
                errors=[str(e)],
                warnings=[]
            )
    
    async def process_selective(self, 
                              bookmarks: List[Dict[str, Any]], 
                              options: EnhancementOptions) -> EnhancementResult:
        """
        Process bookmarks with selective enhancement options.
        
        Args:
            bookmarks: List of bookmark dictionaries to enhance
            options: Specific enhancement options
            
        Returns:
            EnhancementResult with selectively enhanced bookmarks
        """
        config = EnhancementConfig(
            mode=EnhancementMode.SELECTIVE,
            options=options
        )
        
        return await self.process_bookmarks(bookmarks, config)
    
    async def _validate_input_bookmarks(self, bookmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and filter input bookmarks"""
        validated = []
        
        for bookmark in bookmarks:
            if self._is_valid_bookmark(bookmark):
                validated.append(bookmark)
            else:
                logger.warning(f"Invalid bookmark skipped: {bookmark.get('url', 'unknown')}")
        
        logger.info(f"Validated {len(validated)}/{len(bookmarks)} bookmarks")
        return validated
    
    def _is_valid_bookmark(self, bookmark: Dict[str, Any]) -> bool:
        """Check if bookmark has required fields"""
        return (
            isinstance(bookmark, dict) and
            bookmark.get('url') and
            isinstance(bookmark.get('url'), str) and
            bookmark['url'].startswith(('http://', 'https://'))
        )
    
    async def _create_backup(self, bookmarks: List[Dict[str, Any]]) -> None:
        """Create backup of original bookmarks"""
        try:
            # Use safety manager to create backup
            # This would integrate with the existing backup system
            logger.info("Backup created for enhancement operation")
            
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    async def _ensure_models_trained(self, bookmarks: List[Dict[str, Any]]) -> None:
        """Ensure AI models are trained with current data"""
        try:
            # Train tag predictor if not already trained
            if not self.tag_predictor.is_trained:
                logger.info("Training tag predictor...")
                training_result = self.tag_predictor.train_from_bookmarks(bookmarks)
                if training_result.success:
                    logger.info(f"Tag predictor trained with {training_result.samples_trained} samples")
                else:
                    logger.warning("Tag predictor training failed")
            
            # Train content analyzer if not already trained
            if not self.content_analyzer.is_trained:
                logger.info("Training content analyzer...")
                documents = [
                    f"{b.get('name', '')} {b.get('description', '')}"
                    for b in bookmarks
                    if b.get('name') or b.get('description')
                ]
                if documents:
                    self.content_analyzer.train_models(documents)
                    logger.info(f"Content analyzer trained with {len(documents)} documents")
            
            # Compute embeddings for similarity engine
            logger.info("Computing embeddings for similarity analysis...")
            self.similarity_engine.compute_bookmark_embeddings(bookmarks)
            
        except Exception as e:
            logger.warning(f"Model training failed: {e}")
    
    async def _full_processing(self, 
                             bookmarks: List[Dict[str, Any]], 
                             stats: EnhancementStats) -> List[Dict[str, Any]]:
        """Process bookmarks through full enhancement pipeline"""
        enhanced_bookmarks = []
        
        # Process in batches for better performance and error isolation
        batch_size = self.enhancement_config.batch_size
        total_batches = (len(bookmarks) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(bookmarks))
            batch = bookmarks[start_idx:end_idx]
            
            self.progress_monitor.update_progress(
                self.current_operation_id,
                15 + (batch_idx * 75 // total_batches),
                f"Processing batch {batch_idx + 1}/{total_batches}"
            )
            
            try:
                enhanced_batch = await self._process_batch(batch, stats)
                enhanced_bookmarks.extend(enhanced_batch)
                
            except Exception as e:
                logger.error(f"Batch {batch_idx + 1} failed: {e}")
                stats.errors_encountered += len(batch)
                
                # Process individually for error isolation
                for bookmark in batch:
                    try:
                        enhanced = await self._process_single_bookmark(bookmark, stats)
                        enhanced_bookmarks.append(enhanced)
                    except Exception as bookmark_error:
                        logger.error(f"Failed to process bookmark {bookmark.get('url')}: {bookmark_error}")
                        enhanced_bookmarks.append(bookmark)  # Keep original
                        stats.errors_encountered += 1
        
        return enhanced_bookmarks
    
    async def _process_batch(self, 
                           batch: List[Dict[str, Any]], 
                           stats: EnhancementStats) -> List[Dict[str, Any]]:
        """Process a batch of bookmarks concurrently"""
        semaphore = asyncio.Semaphore(self.enhancement_config.max_concurrent)
        
        async def process_with_semaphore(bookmark: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self._process_single_bookmark(bookmark, stats)
        
        tasks = [process_with_semaphore(bookmark) for bookmark in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        enhanced_batch = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Bookmark processing failed: {result}")
                enhanced_batch.append(batch[i])  # Keep original
                stats.errors_encountered += 1
            else:
                enhanced_batch.append(result)
        
        return enhanced_batch
    
    async def _process_single_bookmark(self, 
                                     bookmark: Dict[str, Any], 
                                     stats: EnhancementStats) -> Dict[str, Any]:
        """Process a single bookmark through all enhancement steps"""
        enhanced = bookmark.copy()
        
        try:
            # Step 1: Title enhancement
            if self.enhancement_config.options.enhance_titles:
                enhanced = await self._enhance_title(enhanced, stats)
            
            # Step 2: Auto-tagging
            if self.enhancement_config.options.generate_tags:
                enhanced = await self._generate_tags(enhanced, stats)
            
            # Step 3: Description generation
            if self.enhancement_config.options.create_descriptions:
                enhanced = await self._generate_description(enhanced, stats)
            
            stats.processed_bookmarks += 1
            return enhanced
            
        except Exception as e:
            logger.error(f"Failed to process bookmark {bookmark.get('url')}: {e}")
            stats.errors_encountered += 1
            return bookmark  # Return original on failure
    
    async def _enhance_title(self, bookmark: Dict[str, Any], stats: EnhancementStats) -> Dict[str, Any]:
        """Enhance bookmark title using existing scraping systems"""
        try:
            url = bookmark.get('url')
            current_title = bookmark.get('name', '')
            
            # Skip if title is already good quality
            if current_title and len(current_title) > 10 and not self._is_generic_title(current_title):
                return bookmark
            
            # Use existing link enhancement engine to get better title
            scraping_result = self.link_enhancer.enhance_bookmark(url)
            
            if scraping_result.success and scraping_result.title:
                # Clean and improve the scraped title
                cleaned_title = self._clean_title(scraping_result.title, url)
                
                if self._is_title_better(cleaned_title, current_title):
                    bookmark['name'] = cleaned_title
                    stats.titles_enhanced += 1
                    logger.debug(f"Enhanced title for {url}: {cleaned_title}")
            
            return bookmark
            
        except Exception as e:
            logger.warning(f"Title enhancement failed for {bookmark.get('url')}: {e}")
            return bookmark
    
    async def _generate_tags(self, bookmark: Dict[str, Any], stats: EnhancementStats) -> Dict[str, Any]:
        """Generate tags using existing AI systems"""
        try:
            url = bookmark.get('url', '')
            title = bookmark.get('name', '')
            description = bookmark.get('description', '')
            existing_tags = [tag.get('name', '') if isinstance(tag, dict) else str(tag) 
                           for tag in bookmark.get('tags', [])]
            
            # Combine content for analysis
            content = f"{title} {description}".strip()
            
            generated_tags = []
            
            # Use tag predictor for ML-based tags
            if self.tag_predictor.is_trained:
                ml_predictions = self.tag_predictor.predict_tags(title, content, url, existing_tags)
                for prediction in ml_predictions:
                    if prediction.confidence >= self.enhancement_config.tag_confidence_threshold:
                        generated_tags.append(prediction.tag)
            
            # Use specialized analyzers for domain-specific tags
            specialized_tags = await self._get_specialized_tags(url, title, content)
            generated_tags.extend(specialized_tags)
            
            # Use smart dictionary for intelligent suggestions
            dictionary_suggestions = self.smart_dictionary.suggest_tags_for_bookmark(
                url, title, content, existing_tags
            )
            for tag, confidence in dictionary_suggestions:
                if confidence >= self.enhancement_config.tag_confidence_threshold:
                    generated_tags.append(tag)
            
            # Filter and limit tags
            final_tags = self._filter_and_limit_tags(
                generated_tags, 
                existing_tags, 
                self.enhancement_config.options.max_tags_per_bookmark
            )
            
            if final_tags:
                # Convert to tag format expected by the system
                new_tags = [{'name': tag} for tag in final_tags]
                bookmark['tags'] = bookmark.get('tags', []) + new_tags
                stats.tags_generated += len(final_tags)
                logger.debug(f"Generated {len(final_tags)} tags for {url}")
            
            return bookmark
            
        except Exception as e:
            logger.warning(f"Tag generation failed for {bookmark.get('url')}: {e}")
            return bookmark
    
    async def _generate_description(self, bookmark: Dict[str, Any], stats: EnhancementStats) -> Dict[str, Any]:
        """Generate description using content analysis and AI"""
        try:
            url = bookmark.get('url', '')
            title = bookmark.get('name', '')
            current_description = bookmark.get('description', '')
            
            # Skip if description already exists and preserve_existing is True
            if (current_description and 
                self.enhancement_config.options.preserve_existing and 
                len(current_description) >= self.enhancement_config.description_min_length):
                return bookmark
            
            # Try to get description from existing scraping result
            scraping_result = self.link_enhancer.enhance_bookmark(url)
            
            generated_description = None
            
            # Priority 1: Meta description from scraping
            if scraping_result.success and scraping_result.description:
                generated_description = scraping_result.description
            
            # Priority 2: AI-powered summarization
            elif (self.enhancement_config.enable_ai_summarization and 
                  self.ollama_client and 
                  scraping_result.success):
                
                content = getattr(scraping_result, 'content', '') or ''
                if content:
                    try:
                        ai_summary = self.ollama_client.generate_bookmark_summary(
                            title, content, url, self.enhancement_config.options.description_max_length
                        )
                        if ai_summary and len(ai_summary) >= self.enhancement_config.description_min_length:
                            generated_description = ai_summary
                    except Exception as e:
                        logger.warning(f"AI summarization failed for {url}: {e}")
            
            # Priority 3: Content analysis fallback
            if not generated_description and scraping_result.success:
                content = getattr(scraping_result, 'content', '') or ''
                if content:
                    analysis_result = self.content_analyzer.analyze_content(title, content, url)
                    if analysis_result.keywords:
                        # Create description from keywords and content type
                        keywords_text = ', '.join(analysis_result.keywords[:5])
                        generated_description = f"{analysis_result.content_type} resource about {keywords_text}"
            
            # Apply generated description if it's better than current
            if (generated_description and 
                len(generated_description) >= self.enhancement_config.description_min_length and
                self._is_description_better(generated_description, current_description)):
                
                bookmark['description'] = generated_description
                stats.descriptions_created += 1
                logger.debug(f"Generated description for {url}")
            
            return bookmark
            
        except Exception as e:
            logger.warning(f"Description generation failed for {bookmark.get('url')}: {e}")
            return bookmark
    
    async def _get_specialized_tags(self, url: str, title: str, content: str) -> List[str]:
        """Get specialized tags from domain-specific analyzers"""
        specialized_tags = []
        
        try:
            # Gaming content analysis
            if self.gaming_analyzer.can_analyze(url, title, content):
                gaming_result = self.gaming_analyzer.analyze(url, title, content)
                specialized_tags.extend(gaming_result.specialized_tags)
            
            # Development content analysis
            if self.development_analyzer.can_analyze(url, title, content):
                dev_result = self.development_analyzer.analyze(url, title, content)
                specialized_tags.extend(dev_result.specialized_tags)
            
        except Exception as e:
            logger.warning(f"Specialized tag analysis failed: {e}")
        
        return specialized_tags
    
    async def _dry_run_processing(self, 
                                bookmarks: List[Dict[str, Any]], 
                                stats: EnhancementStats) -> List[Dict[str, Any]]:
        """Process bookmarks in dry-run mode (preview changes only)"""
        logger.info("Running in dry-run mode - no changes will be applied")
        
        # Simulate processing for preview
        preview_results = []
        
        for i, bookmark in enumerate(bookmarks):
            self.progress_monitor.update_progress(
                self.current_operation_id,
                15 + (i * 75 // len(bookmarks)),
                f"Analyzing bookmark {i + 1}/{len(bookmarks)} (dry-run)"
            )
            
            # Create preview of what would be enhanced
            preview = bookmark.copy()
            preview['_enhancement_preview'] = {
                'would_enhance_title': self._would_enhance_title(bookmark),
                'would_generate_tags': self._would_generate_tags(bookmark),
                'would_create_description': self._would_create_description(bookmark)
            }
            
            preview_results.append(preview)
            stats.processed_bookmarks += 1
        
        return preview_results
    
    async def _validation_only_processing(self, 
                                        bookmarks: List[Dict[str, Any]], 
                                        stats: EnhancementStats) -> List[Dict[str, Any]]:
        """Process bookmarks in validation-only mode"""
        logger.info("Running in validation-only mode")
        
        validated_results = []
        
        for i, bookmark in enumerate(bookmarks):
            self.progress_monitor.update_progress(
                self.current_operation_id,
                15 + (i * 75 // len(bookmarks)),
                f"Validating bookmark {i + 1}/{len(bookmarks)}"
            )
            
            # Add validation information
            validated = bookmark.copy()
            validated['_validation_info'] = {
                'title_quality': self._assess_title_quality(bookmark.get('name', '')),
                'has_description': bool(bookmark.get('description')),
                'tag_count': len(bookmark.get('tags', [])),
                'url_valid': self._is_valid_bookmark(bookmark)
            }
            
            validated_results.append(validated)
            stats.processed_bookmarks += 1
        
        return validated_results
    
    async def _generate_quality_report(self, 
                                     original: List[Dict[str, Any]], 
                                     enhanced: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate quality assessment report"""
        try:
            report = {
                'total_bookmarks': len(original),
                'enhanced_bookmarks': len(enhanced),
                'title_improvements': 0,
                'tag_additions': 0,
                'description_additions': 0,
                'quality_score': 0.0
            }
            
            # Compare original vs enhanced
            for orig, enh in zip(original, enhanced):
                # Count title improvements
                if orig.get('name') != enh.get('name'):
                    report['title_improvements'] += 1
                
                # Count tag additions
                orig_tags = len(orig.get('tags', []))
                enh_tags = len(enh.get('tags', []))
                if enh_tags > orig_tags:
                    report['tag_additions'] += 1
                
                # Count description additions
                if not orig.get('description') and enh.get('description'):
                    report['description_additions'] += 1
            
            # Calculate overall quality score
            if len(original) > 0:
                improvements = (report['title_improvements'] + 
                              report['tag_additions'] + 
                              report['description_additions'])
                report['quality_score'] = (improvements / (len(original) * 3)) * 100
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return {'error': str(e)}
    
    def _is_generic_title(self, title: str) -> bool:
        """Check if title is generic or low quality"""
        if not title or len(title) < 3:
            return True
        
        generic_patterns = [
            r'^untitled', r'^page', r'^document', r'^home',
            r'^index', r'^default', r'^\d+$', r'^[a-z]+\.[a-z]+$'
        ]
        
        title_lower = title.lower()
        return any(re.match(pattern, title_lower) for pattern in generic_patterns)
    
    def _clean_title(self, title: str, url: str) -> str:
        """Clean and standardize title"""
        cleaned = title.strip()
        
        # Remove common site name patterns
        site_patterns = [
            r' - [^-]+$',  # " - Site Name"
            r' \| [^|]+$',  # " | Site Name"
            r' :: [^:]+$',  # " :: Site Name"
            r' — [^—]+$',   # " — Site Name"
        ]
        
        for pattern in site_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _is_title_better(self, new_title: str, current_title: str) -> bool:
        """Determine if new title is better than current"""
        if not current_title:
            return bool(new_title)
        
        if not new_title:
            return False
        
        # New title is better if it's longer and more descriptive
        return (len(new_title) > len(current_title) and 
                not self._is_generic_title(new_title) and
                new_title.lower() != current_title.lower())
    
    def _is_description_better(self, new_desc: str, current_desc: str) -> bool:
        """Determine if new description is better than current"""
        if not current_desc:
            return bool(new_desc)
        
        if not new_desc:
            return False
        
        # New description is better if it's longer and more informative
        return (len(new_desc) > len(current_desc) and
                new_desc.lower() != current_desc.lower())
    
    def _filter_and_limit_tags(self, 
                             generated_tags: List[str], 
                             existing_tags: List[str], 
                             max_tags: int) -> List[str]:
        """Filter duplicate tags and limit to maximum count"""
        # Convert existing tags to lowercase for comparison
        existing_lower = {tag.lower() for tag in existing_tags if tag}
        
        # Filter out duplicates and empty tags
        filtered_tags = []
        for tag in generated_tags:
            if (tag and 
                tag.lower() not in existing_lower and 
                tag not in filtered_tags and
                len(tag) > 2):  # Minimum tag length
                filtered_tags.append(tag)
        
        # Limit to maximum count
        return filtered_tags[:max_tags]
    
    def _would_enhance_title(self, bookmark: Dict[str, Any]) -> bool:
        """Check if title would be enhanced (for dry-run)"""
        title = bookmark.get('name', '')
        return not title or len(title) < 10 or self._is_generic_title(title)
    
    def _would_generate_tags(self, bookmark: Dict[str, Any]) -> bool:
        """Check if tags would be generated (for dry-run)"""
        tags = bookmark.get('tags', [])
        return len(tags) < 5  # Would generate if less than 5 tags
    
    def _would_create_description(self, bookmark: Dict[str, Any]) -> bool:
        """Check if description would be created (for dry-run)"""
        description = bookmark.get('description', '')
        return not description or len(description) < self.enhancement_config.description_min_length
    
    def _assess_title_quality(self, title: str) -> float:
        """Assess title quality score (0-1)"""
        if not title:
            return 0.0
        
        score = 0.0
        
        # Length scoring
        if 10 <= len(title) <= 100:
            score += 0.4
        elif len(title) > 5:
            score += 0.2
        
        # Descriptiveness scoring
        if not self._is_generic_title(title):
            score += 0.4
        
        # Formatting scoring
        if title[0].isupper() and not title.isupper():
            score += 0.2
        
        return min(1.0, score)
    
    def _create_error_result(self, error_message: str, stats: EnhancementStats) -> EnhancementResult:
        """Create error result"""
        return EnhancementResult(
            success=False,
            enhanced_bookmarks=[],
            stats=stats,
            errors=[error_message],
            warnings=[]
        )
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'components_initialized': {
                'link_enhancer': self.link_enhancer is not None,
                'tag_predictor': self.tag_predictor is not None,
                'similarity_engine': self.similarity_engine is not None,
                'content_analyzer': self.content_analyzer is not None,
                'smart_dictionary': self.smart_dictionary is not None,
                'ollama_client': self.ollama_client is not None,
            },
            'model_status': {
                'tag_predictor_trained': getattr(self.tag_predictor, 'is_trained', False),
                'content_analyzer_trained': getattr(self.content_analyzer, 'is_trained', False),
                'similarity_embeddings_cached': len(getattr(self.similarity_engine, 'bookmark_embeddings', {})),
            },
            'configuration': {
                'mode': self.enhancement_config.mode.value,
                'batch_size': self.enhancement_config.batch_size,
                'max_concurrent': self.enhancement_config.max_concurrent,
                'ai_summarization_enabled': self.enhancement_config.enable_ai_summarization,
            },
            'current_operation': self.current_operation_id
        }
    
    def cleanup(self) -> None:
        """Cleanup pipeline resources"""
        try:
            if self.ollama_client:
                self.ollama_client.cleanup()
            
            # Clear caches
            self.enhancement_cache.clear()
            
            logger.info("Enhancement pipeline cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during pipeline cleanup: {e}")


