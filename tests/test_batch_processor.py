"""Tests for batch processing system with recovery"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from enhancement.batch_processor import (
    BatchProcessor, BatchStatus, BatchCheckpoint, BatchMetrics, BatchResult
)
from enhancement.graceful_degradation import EnhancementContext


class TestBatchProcessor:
    """Test batch processing functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directory for checkpoints
        self.temp_dir = Path(tempfile.mkdtemp())
        
        config = {
            'batch_processing': {
                'default_batch_size': 5,
                'max_batch_size': 10,
                'min_batch_size': 2,
                'adaptive_batch_sizing': True,
                'max_memory_usage_mb': 512,
                'enable_checkpoints': True,
                'checkpoint_interval': 3,
                'checkpoint_dir': str(self.temp_dir),
                'max_concurrent_batches': 2,
                'batch_timeout': 30.0
            },
            'graceful_degradation': {
                'max_retries': 1,
                'retry_delay': 0.1,
                'continue_on_error': True,
                'enable_fallbacks': True
            },
            'verbose': False
        }
        
        self.processor = BatchProcessor(config)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @pytest.mark.asyncio
    async def test_successful_batch_processing(self):
        """Test successful batch processing of bookmarks"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(10)
        ]
        
        # Mock successful enhancement components
        async def mock_title_enhancer(bookmark):
            return {'name': f"Enhanced {bookmark['name']}"}
        
        async def mock_auto_tagger(bookmark):
            return {'tags': [{'name': 'test'}, {'name': 'example'}]}
        
        components = {
            'title_enhancement': mock_title_enhancer,
            'auto_tagging': mock_auto_tagger
        }
        
        result = await self.processor.process_bookmarks_batch(bookmarks, components)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.metrics.total_items == 10
        assert result.metrics.processed_items == 10
        assert result.metrics.successful_items == 10
        assert result.metrics.failed_items == 0
        assert len(result.successful_contexts) == 10
        assert len(result.failed_contexts) == 0
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_failures(self):
        """Test batch processing with some failures"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(8)
        ]
        
        # Mock component that fails for certain bookmarks
        async def flaky_component(bookmark):
            if bookmark['id'] % 3 == 0:  # Fail every 3rd bookmark
                raise ValueError(f"Failed for bookmark {bookmark['id']}")
            return {'result': 'success'}
        
        components = {
            'flaky_component': flaky_component
        }
        
        result = await self.processor.process_bookmarks_batch(bookmarks, components)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.metrics.total_items == 8
        assert result.metrics.processed_items == 8
        assert result.metrics.failed_items > 0  # Some should fail
        assert result.metrics.successful_items > 0  # Some should succeed
        assert result.metrics.error_rate > 0
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation_and_recovery(self):
        """Test checkpoint creation and recovery functionality"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(12)  # More than checkpoint interval (3)
        ]
        
        async def mock_component(bookmark):
            return {'result': f"processed_{bookmark['id']}"}
        
        components = {
            'test_component': mock_component
        }
        
        result = await self.processor.process_bookmarks_batch(bookmarks, components)
        
        assert result.status == BatchStatus.COMPLETED
        assert len(result.checkpoints_created) > 0  # Should create checkpoints
        
        # Test checkpoint listing
        checkpoints = self.processor.list_checkpoints(result.batch_id)
        assert len(checkpoints) > 0
        
        # Test checkpoint loading
        checkpoint_id = checkpoints[0]['checkpoint_id']
        checkpoint = await self.processor._load_checkpoint(checkpoint_id)
        assert checkpoint is not None
        assert checkpoint.batch_id == result.batch_id
        assert checkpoint.processed_count > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing based on performance"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(20)
        ]
        
        # Mock component with variable performance
        call_count = 0
        
        async def variable_performance_component(bookmark):
            nonlocal call_count
            call_count += 1
            
            # Simulate slower processing for later items
            if call_count > 10:
                await asyncio.sleep(0.1)
            
            return {'result': 'success'}
        
        components = {
            'variable_component': variable_performance_component
        }
        
        result = await self.processor.process_bookmarks_batch(bookmarks, components)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.metrics.total_items == 20
        assert result.metrics.processed_items == 20
        # Batch size may have been adjusted during processing
    
    @pytest.mark.asyncio
    async def test_memory_management(self):
        """Test memory management and batch size adjustment"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(15)
        ]
        
        async def mock_component(bookmark):
            return {'result': 'success'}
        
        components = {
            'test_component': mock_component
        }
        
        # Mock high memory usage
        with patch.object(self.processor, '_check_memory_usage', return_value=True):
            result = await self.processor.process_bookmarks_batch(bookmarks, components)
        
        assert result.status == BatchStatus.COMPLETED
        assert len(result.warnings) > 0  # Should warn about high memory usage
        # Batch size should have been reduced
    
    @pytest.mark.asyncio
    async def test_batch_cancellation(self):
        """Test batch cancellation functionality"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(100)  # Large batch
        ]
        
        async def slow_component(bookmark):
            await asyncio.sleep(0.1)  # Slow processing
            return {'result': 'success'}
        
        components = {
            'slow_component': slow_component
        }
        
        # Start processing in background
        task = asyncio.create_task(
            self.processor.process_bookmarks_batch(bookmarks, components, batch_id='test_batch')
        )
        
        # Wait a bit then cancel
        await asyncio.sleep(0.2)
        cancelled = self.processor.cancel_batch('test_batch')
        assert cancelled == True
        
        # Wait for task to complete
        result = await task
        
        # Should have been cancelled
        assert result.status == BatchStatus.CANCELLED
        assert result.metrics.processed_items < result.metrics.total_items
    
    @pytest.mark.asyncio
    async def test_batch_status_monitoring(self):
        """Test batch status monitoring"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(10)
        ]
        
        async def mock_component(bookmark):
            await asyncio.sleep(0.05)  # Small delay
            return {'result': 'success'}
        
        components = {
            'test_component': mock_component
        }
        
        # Start processing in background
        task = asyncio.create_task(
            self.processor.process_bookmarks_batch(bookmarks, components, batch_id='status_test')
        )
        
        # Check status while running
        await asyncio.sleep(0.1)
        status = self.processor.get_batch_status('status_test')
        
        if status:  # May complete too quickly in tests
            assert status['batch_id'] == 'status_test'
            assert status['status'] in ['running', 'completed']
            assert 'metrics' in status
            assert status['metrics']['total_items'] == 10
        
        # Wait for completion
        result = await task
        assert result.status == BatchStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self):
        """Test concurrent batch processing with semaphore"""
        bookmarks1 = [
            {'id': i, 'name': f'Batch1 Bookmark {i}', 'url': f'https://example.com/1/{i}'}
            for i in range(5)
        ]
        
        bookmarks2 = [
            {'id': i + 100, 'name': f'Batch2 Bookmark {i}', 'url': f'https://example.com/2/{i}'}
            for i in range(5)
        ]
        
        async def mock_component(bookmark):
            await asyncio.sleep(0.05)
            return {'result': 'success'}
        
        components = {
            'test_component': mock_component
        }
        
        # Start two batches concurrently
        task1 = asyncio.create_task(
            self.processor.process_bookmarks_batch(bookmarks1, components, batch_id='batch1')
        )
        task2 = asyncio.create_task(
            self.processor.process_bookmarks_batch(bookmarks2, components, batch_id='batch2')
        )
        
        # Wait for both to complete
        result1, result2 = await asyncio.gather(task1, task2)
        
        assert result1.status == BatchStatus.COMPLETED
        assert result2.status == BatchStatus.COMPLETED
        assert result1.batch_id == 'batch1'
        assert result2.batch_id == 'batch2'
    
    def test_checkpoint_cleanup(self):
        """Test checkpoint cleanup functionality"""
        # Create some mock checkpoint files
        for i in range(5):
            checkpoint_file = self.temp_dir / f"test_checkpoint_{i}.pkl"
            checkpoint_file.write_text("mock checkpoint data")
        
        # Should have 5 files
        assert len(list(self.temp_dir.glob("*.pkl"))) == 5
        
        # Cleanup with max_age_hours=0 should delete all
        deleted_count = self.processor.cleanup_old_checkpoints(max_age_hours=0)
        assert deleted_count == 5
        assert len(list(self.temp_dir.glob("*.pkl"))) == 0
    
    def test_batch_statistics(self):
        """Test batch statistics retrieval"""
        stats = self.processor.get_batch_statistics()
        
        assert 'config' in stats
        assert 'active_batches' in stats
        assert 'max_concurrent_batches' in stats
        assert 'default_batch_size' in stats
        assert 'adaptive_batch_sizing' in stats
        assert 'checkpoints_enabled' in stats
        assert 'memory_limits' in stats
        
        assert stats['default_batch_size'] == 5
        assert stats['max_concurrent_batches'] == 2
        assert stats['adaptive_batch_sizing'] == True
        assert stats['checkpoints_enabled'] == True
    
    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self):
        """Test resuming processing from a checkpoint"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(8)
        ]
        
        async def mock_component(bookmark):
            return {'result': f"processed_{bookmark['id']}"}
        
        components = {
            'test_component': mock_component
        }
        
        # First, run a batch that creates checkpoints
        result1 = await self.processor.process_bookmarks_batch(bookmarks, components, batch_id='resume_test')
        
        assert result1.status == BatchStatus.COMPLETED
        assert len(result1.checkpoints_created) > 0
        
        # Get the first checkpoint
        checkpoints = self.processor.list_checkpoints('resume_test')
        assert len(checkpoints) > 0
        
        checkpoint_id = checkpoints[0]['checkpoint_id']
        
        # Resume from checkpoint (simulate partial processing)
        result2 = await self.processor.process_bookmarks_batch(
            bookmarks, components, batch_id='resume_test_2', resume_from_checkpoint=checkpoint_id
        )
        
        assert result2.status == BatchStatus.COMPLETED
        assert result2.recovery_attempts > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_batch(self):
        """Test comprehensive error handling in batch processing"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(6)
        ]
        
        # Component that raises different types of errors
        async def error_prone_component(bookmark):
            bookmark_id = bookmark['id']
            
            if bookmark_id == 1:
                raise ValueError("Validation error")
            elif bookmark_id == 2:
                raise ConnectionError("Network error")
            elif bookmark_id == 3:
                raise TimeoutError("Timeout error")
            else:
                return {'result': 'success'}
        
        components = {
            'error_component': error_prone_component
        }
        
        result = await self.processor.process_bookmarks_batch(bookmarks, components)
        
        assert result.status == BatchStatus.COMPLETED
        assert result.metrics.total_items == 6
        assert result.metrics.processed_items == 6
        assert result.metrics.failed_items > 0  # Some should fail
        assert result.metrics.successful_items > 0  # Some should succeed with fallbacks
        
        # Check that errors were captured
        error_contexts = [ctx for ctx in result.failed_contexts + result.successful_contexts
                         if len(ctx.global_errors) > 0 or 
                         any(len(r.errors) > 0 for r in ctx.component_results.values())]
        assert len(error_contexts) > 0
    
    @pytest.mark.asyncio
    async def test_metrics_calculation(self):
        """Test batch metrics calculation"""
        bookmarks = [
            {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
            for i in range(10)
        ]
        
        async def mock_component(bookmark):
            # Simulate some processing time
            await asyncio.sleep(0.01)
            return {'result': 'success'}
        
        components = {
            'test_component': mock_component
        }
        
        result = await self.processor.process_bookmarks_batch(bookmarks, components)
        
        assert result.status == BatchStatus.COMPLETED
        
        metrics = result.metrics
        assert metrics.total_items == 10
        assert metrics.processed_items == 10
        assert metrics.processing_rate > 0  # Should have calculated rate
        assert metrics.memory_usage_mb > 0  # Should have memory usage
        assert 0 <= metrics.error_rate <= 1  # Error rate should be valid percentage
        assert 0 <= metrics.fallback_usage_rate <= 1  # Fallback rate should be valid percentage


if __name__ == "__main__":
    pytest.main([__file__])