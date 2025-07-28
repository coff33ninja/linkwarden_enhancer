"""Graceful degradation framework for bookmark enhancement operations"""

import asyncio
import traceback
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json

from utils.logging_utils import get_logger, ComponentLogger
from data_models import SafetyResult, ChangeSet, EnhancementReport

logger = get_logger(__name__)


class ComponentStatus(Enum):
    """Status of enhancement components"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComponentError:
    """Error information for a component"""
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime
    traceback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_suggestion: Optional[str] = None


@dataclass
class ComponentResult:
    """Result of a component operation"""
    component: str
    status: ComponentStatus
    data: Any = None
    errors: List[ComponentError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    fallback_used: bool = False
    degraded_features: List[str] = field(default_factory=list)


@dataclass
class EnhancementContext:
    """Context for enhancement operations"""
    bookmark_id: int
    original_bookmark: Dict[str, Any]
    enhanced_bookmark: Dict[str, Any]
    component_results: Dict[str, ComponentResult] = field(default_factory=dict)
    global_errors: List[ComponentError] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    dry_run: bool = False


class GracefulDegradationFramework:
    """Framework for graceful degradation of enhancement operations"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize graceful degradation framework"""
        self.config = config
        self.degradation_config = config.get('graceful_degradation', {})
        
        # Component logger for detailed tracking
        self.component_logger = ComponentLogger('enhancement.graceful_degradation', 
                                               verbose=config.get('verbose', False))
        
        # Error handling configuration
        self.max_retries = self.degradation_config.get('max_retries', 3)
        self.retry_delay = self.degradation_config.get('retry_delay', 1.0)
        self.continue_on_error = self.degradation_config.get('continue_on_error', True)
        self.enable_fallbacks = self.degradation_config.get('enable_fallbacks', True)
        
        # Component timeout settings
        self.component_timeouts = self.degradation_config.get('component_timeouts', {
            'title_enhancement': 30.0,
            'auto_tagging': 45.0,
            'description_generation': 60.0,
            'duplicate_detection': 120.0
        })
        
        # Fallback strategies
        self.fallback_strategies = {
            'title_enhancement': self._title_fallback,
            'auto_tagging': self._tagging_fallback,
            'description_generation': self._description_fallback,
            'duplicate_detection': self._duplicate_fallback
        }
        
        # Error recovery suggestions
        self.recovery_suggestions = {
            'network_error': "Check internet connection and retry",
            'timeout_error': "Increase timeout settings or reduce batch size",
            'memory_error': "Reduce batch size or restart application",
            'api_error': "Check API credentials and rate limits",
            'parsing_error': "Validate input data format",
            'validation_error': "Check data integrity and schema compliance"
        }
        
        logger.info("Graceful degradation framework initialized")
    
    async def safe_enhance_bookmark(self, 
                                   bookmark: Dict[str, Any],
                                   enhancement_components: Dict[str, Callable],
                                   dry_run: bool = False) -> EnhancementContext:
        """Safely enhance a bookmark with comprehensive error handling"""
        
        context = EnhancementContext(
            bookmark_id=bookmark.get('id', 0),
            original_bookmark=bookmark.copy(),
            enhanced_bookmark=bookmark.copy(),
            dry_run=dry_run
        )
        
        self.component_logger.debug_operation(
            "safe_enhance_bookmark_start",
            {
                'bookmark_id': context.bookmark_id,
                'components': list(enhancement_components.keys()),
                'dry_run': dry_run
            }
        )
        
        try:
            # Process each enhancement component with isolation
            for component_name, component_func in enhancement_components.items():
                component_result = await self._execute_component_safely(
                    component_name, component_func, context
                )
                
                context.component_results[component_name] = component_result
                
                # Apply successful results to enhanced bookmark
                if component_result.status in [ComponentStatus.SUCCESS, ComponentStatus.PARTIAL_SUCCESS]:
                    if component_result.data:
                        self._apply_component_result(context, component_name, component_result.data)
                
                # Log component completion
                self.component_logger.debug_operation(
                    f"component_completed_{component_name}",
                    {
                        'status': component_result.status.value,
                        'processing_time': component_result.processing_time,
                        'errors': len(component_result.errors),
                        'warnings': len(component_result.warnings),
                        'fallback_used': component_result.fallback_used
                    }
                )
            
            # Calculate overall success
            total_time = (datetime.now() - context.start_time).total_seconds()
            
            self.component_logger.debug_performance(
                "safe_enhance_bookmark_complete",
                total_time,
                {
                    'components_processed': len(context.component_results),
                    'successful_components': sum(1 for r in context.component_results.values() 
                                               if r.status == ComponentStatus.SUCCESS),
                    'failed_components': sum(1 for r in context.component_results.values() 
                                           if r.status == ComponentStatus.FAILED),
                    'fallbacks_used': sum(1 for r in context.component_results.values() 
                                        if r.fallback_used)
                }
            )
            
            return context
            
        except Exception as e:
            # Global error handling
            error = ComponentError(
                component="global",
                error_type=type(e).__name__,
                message=str(e),
                severity=ErrorSeverity.CRITICAL,
                timestamp=datetime.now(),
                traceback=traceback.format_exc(),
                recovery_suggestion="Review system configuration and data integrity"
            )
            
            context.global_errors.append(error)
            
            logger.error(f"Critical error in safe_enhance_bookmark: {e}", exc_info=True)
            
            return context
    
    async def _execute_component_safely(self, 
                                       component_name: str,
                                       component_func: Callable,
                                       context: EnhancementContext) -> ComponentResult:
        """Execute a component with comprehensive error handling"""
        
        start_time = datetime.now()
        result = ComponentResult(component=component_name, status=ComponentStatus.FAILED)
        
        try:
            # Get component timeout
            timeout = self.component_timeouts.get(component_name, 60.0)
            
            # Execute with retries
            for attempt in range(self.max_retries + 1):
                try:
                    self.component_logger.debug_operation(
                        f"component_attempt_{component_name}",
                        {'attempt': attempt + 1, 'max_attempts': self.max_retries + 1}
                    )
                    
                    # Execute component with timeout
                    if asyncio.iscoroutinefunction(component_func):
                        component_result = await asyncio.wait_for(
                            component_func(context.enhanced_bookmark),
                            timeout=timeout
                        )
                    else:
                        component_result = await asyncio.wait_for(
                            asyncio.to_thread(component_func, context.enhanced_bookmark),
                            timeout=timeout
                        )
                    
                    # Success
                    result.status = ComponentStatus.SUCCESS
                    result.data = component_result
                    break
                    
                except asyncio.TimeoutError:
                    error = ComponentError(
                        component=component_name,
                        error_type="TimeoutError",
                        message=f"Component timed out after {timeout}s",
                        severity=ErrorSeverity.HIGH,
                        timestamp=datetime.now(),
                        recovery_suggestion=self.recovery_suggestions.get('timeout_error')
                    )
                    result.errors.append(error)
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        break
                
                except Exception as e:
                    error = ComponentError(
                        component=component_name,
                        error_type=type(e).__name__,
                        message=str(e),
                        severity=self._classify_error_severity(e),
                        timestamp=datetime.now(),
                        traceback=traceback.format_exc(),
                        recovery_suggestion=self._get_recovery_suggestion(e)
                    )
                    result.errors.append(error)
                    
                    # Retry for certain error types
                    if attempt < self.max_retries and self._should_retry_error(e):
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                    else:
                        break
            
            # Try fallback if main component failed
            if result.status == ComponentStatus.FAILED and self.enable_fallbacks:
                fallback_result = await self._try_fallback(component_name, context)
                if fallback_result:
                    result.status = ComponentStatus.DEGRADED
                    result.data = fallback_result
                    result.fallback_used = True
                    result.degraded_features.append(f"{component_name}_fallback_used")
            
            # Calculate processing time
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            # Unexpected error in error handling
            error = ComponentError(
                component=component_name,
                error_type="UnexpectedError",
                message=f"Unexpected error in component execution: {e}",
                severity=ErrorSeverity.CRITICAL,
                timestamp=datetime.now(),
                traceback=traceback.format_exc()
            )
            result.errors.append(error)
            result.processing_time = (datetime.now() - start_time).total_seconds()
            
            return result
    
    async def _try_fallback(self, component_name: str, context: EnhancementContext) -> Any:
        """Try fallback strategy for failed component"""
        
        if component_name not in self.fallback_strategies:
            return None
        
        try:
            self.component_logger.debug_operation(
                f"fallback_attempt_{component_name}",
                {'bookmark_id': context.bookmark_id}
            )
            
            fallback_func = self.fallback_strategies[component_name]
            return await fallback_func(context.enhanced_bookmark, context)
            
        except Exception as e:
            logger.warning(f"Fallback failed for {component_name}: {e}")
            return None
    
    async def _title_fallback(self, bookmark: Dict[str, Any], context: EnhancementContext) -> Dict[str, Any]:
        """Fallback strategy for title enhancement"""
        try:
            # Simple title cleaning fallback
            title = bookmark.get('name', '') or bookmark.get('title', '')
            url = bookmark.get('url', '')
            
            if not title and url:
                # Extract title from URL
                from urllib.parse import urlparse
                parsed = urlparse(url)
                path_parts = [part for part in parsed.path.split('/') if part]
                if path_parts:
                    title = path_parts[-1].replace('-', ' ').replace('_', ' ').title()
                else:
                    title = parsed.netloc.replace('www.', '').title()
            
            # Basic cleaning
            if title:
                title = title.strip()
                # Remove common suffixes
                for suffix in [' - Site Name', ' | Website', ' :: Portal']:
                    if title.endswith(suffix):
                        title = title[:-len(suffix)]
            
            return {'name': title or 'Untitled Bookmark'}
            
        except Exception as e:
            logger.warning(f"Title fallback failed: {e}")
            return {'name': bookmark.get('name', 'Untitled Bookmark')}
    
    async def _tagging_fallback(self, bookmark: Dict[str, Any], context: EnhancementContext) -> Dict[str, Any]:
        """Fallback strategy for auto-tagging"""
        try:
            # Simple URL-based tagging
            url = bookmark.get('url', '')
            tags = []
            
            if url:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                
                # Domain-based tags
                if 'github.com' in domain:
                    tags.extend(['development', 'code', 'github'])
                elif 'stackoverflow.com' in domain:
                    tags.extend(['development', 'programming', 'help'])
                elif 'youtube.com' in domain:
                    tags.extend(['video', 'entertainment'])
                elif 'reddit.com' in domain:
                    tags.extend(['social', 'discussion'])
                elif any(news in domain for news in ['news', 'bbc', 'cnn', 'reuters']):
                    tags.extend(['news', 'article'])
                
                # Path-based tags
                path_parts = [part.lower() for part in parsed.path.split('/') if part]
                for part in path_parts:
                    if part in ['tutorial', 'guide', 'docs', 'documentation']:
                        tags.append('tutorial')
                    elif part in ['blog', 'article', 'post']:
                        tags.append('article')
            
            # Keep existing tags
            existing_tags = bookmark.get('tags', [])
            if isinstance(existing_tags, list):
                tag_names = [tag.get('name', tag) if isinstance(tag, dict) else str(tag) 
                           for tag in existing_tags]
                tags.extend(tag_names)
            
            # Remove duplicates and limit
            unique_tags = list(dict.fromkeys(tags))[:10]
            
            return {'tags': [{'name': tag} for tag in unique_tags]}
            
        except Exception as e:
            logger.warning(f"Tagging fallback failed: {e}")
            return {'tags': bookmark.get('tags', [])}
    
    async def _description_fallback(self, bookmark: Dict[str, Any], context: EnhancementContext) -> Dict[str, Any]:
        """Fallback strategy for description generation"""
        try:
            # Use existing description or create simple one
            description = bookmark.get('description', '')
            
            if not description:
                title = bookmark.get('name', '')
                url = bookmark.get('url', '')
                
                if title and url:
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc.replace('www.', '')
                    description = f"Bookmark: {title} from {domain}"
                elif url:
                    description = f"Bookmarked link: {url}"
                else:
                    description = "Bookmarked content"
            
            return {'description': description}
            
        except Exception as e:
            logger.warning(f"Description fallback failed: {e}")
            return {'description': bookmark.get('description', '')}
    
    async def _duplicate_fallback(self, bookmark: Dict[str, Any], context: EnhancementContext) -> Dict[str, Any]:
        """Fallback strategy for duplicate detection"""
        try:
            # Simple URL-based duplicate check (placeholder)
            # In real implementation, this would check against a simple URL list
            return {'duplicate_status': 'unchecked', 'reason': 'fallback_mode'}
            
        except Exception as e:
            logger.warning(f"Duplicate detection fallback failed: {e}")
            return {'duplicate_status': 'error'}
    
    def _apply_component_result(self, context: EnhancementContext, component_name: str, result_data: Any):
        """Apply component result to enhanced bookmark"""
        try:
            if isinstance(result_data, dict):
                context.enhanced_bookmark.update(result_data)
            else:
                # Handle non-dict results
                context.enhanced_bookmark[f'{component_name}_result'] = result_data
                
        except Exception as e:
            logger.warning(f"Failed to apply result from {component_name}: {e}")
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity"""
        error_type = type(error).__name__
        
        if error_type in ['MemoryError', 'SystemExit', 'KeyboardInterrupt']:
            return ErrorSeverity.CRITICAL
        elif error_type in ['TimeoutError', 'ConnectionError', 'HTTPError']:
            return ErrorSeverity.HIGH
        elif error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _should_retry_error(self, error: Exception) -> bool:
        """Determine if error should trigger a retry"""
        retry_errors = [
            'ConnectionError', 'TimeoutError', 'HTTPError', 
            'TemporaryFailure', 'RateLimitError'
        ]
        return type(error).__name__ in retry_errors
    
    def _get_recovery_suggestion(self, error: Exception) -> str:
        """Get recovery suggestion for error"""
        error_type = type(error).__name__.lower()
        
        for pattern, suggestion in self.recovery_suggestions.items():
            if pattern in error_type:
                return suggestion
        
        return "Review error details and system logs for specific guidance"
    
    def create_enhancement_report(self, contexts: List[EnhancementContext]) -> EnhancementReport:
        """Create comprehensive enhancement report from contexts"""
        try:
            total_bookmarks = len(contexts)
            successful_enhancements = 0
            total_errors = 0
            total_warnings = 0
            component_stats = {}
            
            for context in contexts:
                # Count successful enhancements
                if any(r.status in [ComponentStatus.SUCCESS, ComponentStatus.PARTIAL_SUCCESS] 
                      for r in context.component_results.values()):
                    successful_enhancements += 1
                
                # Aggregate component statistics
                for component_name, result in context.component_results.items():
                    if component_name not in component_stats:
                        component_stats[component_name] = {
                            'success': 0, 'failed': 0, 'degraded': 0, 
                            'total_time': 0.0, 'fallbacks_used': 0
                        }
                    
                    stats = component_stats[component_name]
                    if result.status == ComponentStatus.SUCCESS:
                        stats['success'] += 1
                    elif result.status == ComponentStatus.FAILED:
                        stats['failed'] += 1
                    elif result.status == ComponentStatus.DEGRADED:
                        stats['degraded'] += 1
                    
                    stats['total_time'] += result.processing_time
                    if result.fallback_used:
                        stats['fallbacks_used'] += 1
                    
                    total_errors += len(result.errors)
                    total_warnings += len(result.warnings)
                
                total_errors += len(context.global_errors)
            
            # Create AI analysis report
            from data_models import AIAnalysisReport
            ai_report = AIAnalysisReport(
                total_bookmarks_analyzed=total_bookmarks,
                ai_tags_suggested=component_stats.get('auto_tagging', {}).get('success', 0),
                duplicates_detected=component_stats.get('duplicate_detection', {}).get('success', 0),
                processing_time=sum(stats.get('total_time', 0) for stats in component_stats.values()),
                model_accuracy_metrics={
                    'enhancement_success_rate': successful_enhancements / total_bookmarks if total_bookmarks > 0 else 0,
                    'error_rate': total_errors / total_bookmarks if total_bookmarks > 0 else 0,
                    'fallback_usage_rate': sum(stats.get('fallbacks_used', 0) for stats in component_stats.values()) / total_bookmarks if total_bookmarks > 0 else 0
                }
            )
            
            return EnhancementReport(
                bookmarks_enhanced=successful_enhancements,
                metadata_fields_added=sum(1 for context in contexts 
                                        if len(context.enhanced_bookmark) > len(context.original_bookmark)),
                scraping_failures=component_stats.get('title_enhancement', {}).get('failed', 0),
                scrapers_used={'graceful_degradation': total_bookmarks},
                average_scraping_time=sum(stats.get('total_time', 0) for stats in component_stats.values()) / total_bookmarks if total_bookmarks > 0 else 0,
                cache_hit_rate=0.0,  # Would be calculated from actual cache usage
                ai_analysis_report=ai_report
            )
            
        except Exception as e:
            logger.error(f"Failed to create enhancement report: {e}")
            return EnhancementReport()
    
    def get_error_summary(self, contexts: List[EnhancementContext]) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        try:
            error_summary = {
                'total_contexts': len(contexts),
                'contexts_with_errors': 0,
                'total_errors': 0,
                'error_by_severity': {severity.value: 0 for severity in ErrorSeverity},
                'error_by_component': {},
                'error_by_type': {},
                'recovery_suggestions': {},
                'most_common_errors': []
            }
            
            all_errors = []
            
            for context in contexts:
                context_has_errors = False
                
                # Component errors
                for result in context.component_results.values():
                    if result.errors:
                        context_has_errors = True
                        all_errors.extend(result.errors)
                
                # Global errors
                if context.global_errors:
                    context_has_errors = True
                    all_errors.extend(context.global_errors)
                
                if context_has_errors:
                    error_summary['contexts_with_errors'] += 1
            
            error_summary['total_errors'] = len(all_errors)
            
            # Analyze errors
            for error in all_errors:
                # By severity
                error_summary['error_by_severity'][error.severity.value] += 1
                
                # By component
                if error.component not in error_summary['error_by_component']:
                    error_summary['error_by_component'][error.component] = 0
                error_summary['error_by_component'][error.component] += 1
                
                # By type
                if error.error_type not in error_summary['error_by_type']:
                    error_summary['error_by_type'][error.error_type] = 0
                error_summary['error_by_type'][error.error_type] += 1
                
                # Recovery suggestions
                if error.recovery_suggestion:
                    if error.error_type not in error_summary['recovery_suggestions']:
                        error_summary['recovery_suggestions'][error.error_type] = error.recovery_suggestion
            
            # Most common errors
            from collections import Counter
            error_messages = [error.message for error in all_errors]
            error_summary['most_common_errors'] = Counter(error_messages).most_common(5)
            
            return error_summary
            
        except Exception as e:
            logger.error(f"Failed to create error summary: {e}")
            return {'error': str(e)}
    
    def get_degradation_stats(self) -> Dict[str, Any]:
        """Get graceful degradation framework statistics"""
        return {
            'config': self.degradation_config,
            'component_timeouts': self.component_timeouts,
            'fallback_strategies': list(self.fallback_strategies.keys()),
            'recovery_suggestions': list(self.recovery_suggestions.keys()),
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'continue_on_error': self.continue_on_error,
            'enable_fallbacks': self.enable_fallbacks
        }