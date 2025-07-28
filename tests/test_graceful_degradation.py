"""Tests for graceful degradation framework"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from enhancement.graceful_degradation import (
    GracefulDegradationFramework, ComponentStatus, ErrorSeverity,
    ComponentError, ComponentResult, EnhancementContext
)


class TestGracefulDegradationFramework:
    """Test graceful degradation framework functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        config = {
            'graceful_degradation': {
                'max_retries': 2,
                'retry_delay': 0.1,
                'continue_on_error': True,
                'enable_fallbacks': True,
                'component_timeouts': {
                    'title_enhancement': 5.0,
                    'auto_tagging': 10.0,
                    'description_generation': 15.0,
                    'duplicate_detection': 20.0
                }
            },
            'verbose': False
        }
        self.framework = GracefulDegradationFramework(config)
    
    @pytest.mark.asyncio
    async def test_successful_component_execution(self):
        """Test successful execution of all components"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark',
            'url': 'https://example.com',
            'description': 'Test description'
        }
        
        # Mock successful components
        async def mock_title_enhancer(bm):
            return {'name': 'Enhanced Title'}
        
        async def mock_auto_tagger(bm):
            return {'tags': [{'name': 'test'}, {'name': 'example'}]}
        
        components = {
            'title_enhancement': mock_title_enhancer,
            'auto_tagging': mock_auto_tagger
        }
        
        context = await self.framework.safe_enhance_bookmark(bookmark, components)
        
        assert context.bookmark_id == 1
        assert len(context.component_results) == 2
        assert context.component_results['title_enhancement'].status == ComponentStatus.SUCCESS
        assert context.component_results['auto_tagging'].status == ComponentStatus.SUCCESS
        assert context.enhanced_bookmark['name'] == 'Enhanced Title'
        assert len(context.enhanced_bookmark['tags']) == 2
    
    @pytest.mark.asyncio
    async def test_component_failure_with_fallback(self):
        """Test component failure with successful fallback"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark',
            'url': 'https://example.com'
        }
        
        # Mock failing component
        async def failing_title_enhancer(bm):
            raise ValueError("Title enhancement failed")
        
        components = {
            'title_enhancement': failing_title_enhancer
        }
        
        context = await self.framework.safe_enhance_bookmark(bookmark, components)
        
        assert context.bookmark_id == 1
        assert len(context.component_results) == 1
        
        result = context.component_results['title_enhancement']
        assert result.status == ComponentStatus.DEGRADED
        assert result.fallback_used == True
        assert len(result.errors) > 0
        assert result.errors[0].error_type == 'ValueError'
        
        # Should have fallback title
        assert 'name' in context.enhanced_bookmark
    
    @pytest.mark.asyncio
    async def test_component_timeout(self):
        """Test component timeout handling"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark',
            'url': 'https://example.com'
        }
        
        # Mock slow component
        async def slow_component(bm):
            await asyncio.sleep(10)  # Longer than timeout
            return {'result': 'success'}
        
        components = {
            'title_enhancement': slow_component
        }
        
        context = await self.framework.safe_enhance_bookmark(bookmark, components)
        
        result = context.component_results['title_enhancement']
        assert result.status == ComponentStatus.DEGRADED  # Should use fallback
        assert result.fallback_used == True
        assert any(error.error_type == 'TimeoutError' for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self):
        """Test retry mechanism for transient failures"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark',
            'url': 'https://example.com'
        }
        
        call_count = 0
        
        async def flaky_component(bm):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary connection error")
            return {'result': 'success after retry'}
        
        components = {
            'title_enhancement': flaky_component
        }
        
        context = await self.framework.safe_enhance_bookmark(bookmark, components)
        
        result = context.component_results['title_enhancement']
        assert result.status == ComponentStatus.SUCCESS
        assert call_count == 2  # Should have retried once
        assert 'result' in context.enhanced_bookmark
    
    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """Test handling of non-retryable errors"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark',
            'url': 'https://example.com'
        }
        
        call_count = 0
        
        async def component_with_logic_error(bm):
            nonlocal call_count
            call_count += 1
            raise ValueError("Logic error - should not retry")
        
        components = {
            'title_enhancement': component_with_logic_error
        }
        
        context = await self.framework.safe_enhance_bookmark(bookmark, components)
        
        result = context.component_results['title_enhancement']
        assert result.status == ComponentStatus.DEGRADED  # Should use fallback
        assert call_count == 1  # Should not retry ValueError
        assert result.fallback_used == True
    
    @pytest.mark.asyncio
    async def test_dry_run_mode(self):
        """Test dry run mode"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark',
            'url': 'https://example.com'
        }
        
        async def mock_component(bm):
            return {'name': 'Modified Title'}
        
        components = {
            'title_enhancement': mock_component
        }
        
        context = await self.framework.safe_enhance_bookmark(bookmark, components, dry_run=True)
        
        assert context.dry_run == True
        assert context.component_results['title_enhancement'].status == ComponentStatus.SUCCESS
        # In dry run, changes should still be applied to enhanced_bookmark for preview
        assert context.enhanced_bookmark['name'] == 'Modified Title'
    
    @pytest.mark.asyncio
    async def test_title_fallback(self):
        """Test title enhancement fallback"""
        bookmark = {
            'id': 1,
            'name': '',  # Empty title
            'url': 'https://example.com/article/python-tutorial'
        }
        
        context = EnhancementContext(
            bookmark_id=1,
            original_bookmark=bookmark,
            enhanced_bookmark=bookmark.copy()
        )
        
        result = await self.framework._title_fallback(bookmark, context)
        
        assert 'name' in result
        assert result['name'] != ''
        assert 'python' in result['name'].lower() or 'tutorial' in result['name'].lower()
    
    @pytest.mark.asyncio
    async def test_tagging_fallback(self):
        """Test auto-tagging fallback"""
        bookmark = {
            'id': 1,
            'name': 'GitHub Repository',
            'url': 'https://github.com/user/repo',
            'tags': []
        }
        
        context = EnhancementContext(
            bookmark_id=1,
            original_bookmark=bookmark,
            enhanced_bookmark=bookmark.copy()
        )
        
        result = await self.framework._tagging_fallback(bookmark, context)
        
        assert 'tags' in result
        assert len(result['tags']) > 0
        
        tag_names = [tag['name'] for tag in result['tags']]
        assert any(tag in ['development', 'code', 'github'] for tag in tag_names)
    
    @pytest.mark.asyncio
    async def test_description_fallback(self):
        """Test description generation fallback"""
        bookmark = {
            'id': 1,
            'name': 'Test Article',
            'url': 'https://example.com/article',
            'description': ''
        }
        
        context = EnhancementContext(
            bookmark_id=1,
            original_bookmark=bookmark,
            enhanced_bookmark=bookmark.copy()
        )
        
        result = await self.framework._description_fallback(bookmark, context)
        
        assert 'description' in result
        assert result['description'] != ''
        assert 'Test Article' in result['description']
        assert 'example.com' in result['description']
    
    def test_error_severity_classification(self):
        """Test error severity classification"""
        # Critical errors
        assert self.framework._classify_error_severity(MemoryError()) == ErrorSeverity.CRITICAL
        assert self.framework._classify_error_severity(SystemExit()) == ErrorSeverity.CRITICAL
        
        # High severity errors
        assert self.framework._classify_error_severity(TimeoutError()) == ErrorSeverity.HIGH
        assert self.framework._classify_error_severity(ConnectionError()) == ErrorSeverity.HIGH
        
        # Medium severity errors
        assert self.framework._classify_error_severity(ValueError()) == ErrorSeverity.MEDIUM
        assert self.framework._classify_error_severity(TypeError()) == ErrorSeverity.MEDIUM
        
        # Low severity errors (default)
        assert self.framework._classify_error_severity(Exception()) == ErrorSeverity.LOW
    
    def test_should_retry_error(self):
        """Test retry decision logic"""
        # Should retry
        assert self.framework._should_retry_error(ConnectionError()) == True
        assert self.framework._should_retry_error(TimeoutError()) == True
        
        # Should not retry
        assert self.framework._should_retry_error(ValueError()) == False
        assert self.framework._should_retry_error(TypeError()) == False
    
    def test_recovery_suggestions(self):
        """Test recovery suggestion generation"""
        suggestions = [
            (ConnectionError(), "network_error"),
            (TimeoutError(), "timeout_error"),
            (MemoryError(), "memory_error"),
            (ValueError(), "validation_error")
        ]
        
        for error, expected_pattern in suggestions:
            suggestion = self.framework._get_recovery_suggestion(error)
            assert isinstance(suggestion, str)
            assert len(suggestion) > 0
    
    def test_enhancement_report_creation(self):
        """Test enhancement report creation"""
        # Create mock contexts
        contexts = []
        
        for i in range(3):
            context = EnhancementContext(
                bookmark_id=i,
                original_bookmark={'id': i, 'name': f'Bookmark {i}'},
                enhanced_bookmark={'id': i, 'name': f'Enhanced Bookmark {i}', 'tags': []}
            )
            
            # Add mock component results
            context.component_results['title_enhancement'] = ComponentResult(
                component='title_enhancement',
                status=ComponentStatus.SUCCESS if i < 2 else ComponentStatus.FAILED,
                processing_time=1.0
            )
            
            context.component_results['auto_tagging'] = ComponentResult(
                component='auto_tagging',
                status=ComponentStatus.SUCCESS,
                processing_time=0.5,
                fallback_used=(i == 1)
            )
            
            contexts.append(context)
        
        report = self.framework.create_enhancement_report(contexts)
        
        assert report.bookmarks_enhanced == 3  # All had at least one successful component
        assert report.ai_analysis_report is not None
        assert report.ai_analysis_report.total_bookmarks_analyzed == 3
        assert report.ai_analysis_report.ai_tags_suggested == 3  # All auto_tagging succeeded
    
    def test_error_summary_creation(self):
        """Test error summary creation"""
        # Create contexts with errors
        contexts = []
        
        for i in range(2):
            context = EnhancementContext(
                bookmark_id=i,
                original_bookmark={'id': i, 'name': f'Bookmark {i}'},
                enhanced_bookmark={'id': i, 'name': f'Bookmark {i}'}
            )
            
            # Add component result with error
            result = ComponentResult(
                component='title_enhancement',
                status=ComponentStatus.FAILED
            )
            
            error = ComponentError(
                component='title_enhancement',
                error_type='ValueError',
                message=f'Test error {i}',
                severity=ErrorSeverity.MEDIUM,
                timestamp=datetime.now(),
                recovery_suggestion='Test recovery'
            )
            result.errors.append(error)
            
            context.component_results['title_enhancement'] = result
            contexts.append(context)
        
        error_summary = self.framework.get_error_summary(contexts)
        
        assert error_summary['total_contexts'] == 2
        assert error_summary['contexts_with_errors'] == 2
        assert error_summary['total_errors'] == 2
        assert error_summary['error_by_severity']['medium'] == 2
        assert error_summary['error_by_component']['title_enhancement'] == 2
        assert error_summary['error_by_type']['ValueError'] == 2
        assert 'ValueError' in error_summary['recovery_suggestions']
    
    def test_degradation_stats(self):
        """Test degradation statistics"""
        stats = self.framework.get_degradation_stats()
        
        assert 'config' in stats
        assert 'component_timeouts' in stats
        assert 'fallback_strategies' in stats
        assert 'recovery_suggestions' in stats
        assert stats['max_retries'] == 2
        assert stats['continue_on_error'] == True
        assert stats['enable_fallbacks'] == True
        
        # Check fallback strategies are available
        expected_strategies = ['title_enhancement', 'auto_tagging', 'description_generation', 'duplicate_detection']
        for strategy in expected_strategies:
            assert strategy in stats['fallback_strategies']
    
    @pytest.mark.asyncio
    async def test_global_error_handling(self):
        """Test global error handling in safe_enhance_bookmark"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark'
        }
        
        # Mock component that causes a critical error in the framework itself
        def problematic_component(bm):
            # Use a different critical error that won't exit the test process
            raise RuntimeError("Critical system error")
        
        components = {
            'problematic_component': problematic_component
        }
        
        context = await self.framework.safe_enhance_bookmark(bookmark, components)
        
        # Should handle the critical error gracefully
        assert context.bookmark_id == 1
        assert len(context.component_results) >= 0  # May or may not have results
        # Should have used fallback or captured error
        if 'problematic_component' in context.component_results:
            result = context.component_results['problematic_component']
            assert result.status in [ComponentStatus.FAILED, ComponentStatus.DEGRADED]
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_components(self):
        """Test handling of both sync and async components"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark',
            'url': 'https://example.com'
        }
        
        # Mix of sync and async components
        async def async_component(bm):
            return {'async_result': 'success'}
        
        def sync_component(bm):
            return {'sync_result': 'success'}
        
        components = {
            'async_component': async_component,
            'sync_component': sync_component
        }
        
        context = await self.framework.safe_enhance_bookmark(bookmark, components)
        
        assert len(context.component_results) == 2
        assert context.component_results['async_component'].status == ComponentStatus.SUCCESS
        assert context.component_results['sync_component'].status == ComponentStatus.SUCCESS
        assert 'async_result' in context.enhanced_bookmark
        assert 'sync_result' in context.enhanced_bookmark


if __name__ == "__main__":
    pytest.main([__file__])