"""Batch processing system with recovery capabilities for bookmark enhancement"""

import asyncio
import json
import time
import psutil
from typing import Dict, List, Any, Optional, Callable, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import pickle

from enhancement.graceful_degradation import GracefulDegradationFramework, EnhancementContext
from core.progress_monitor import ProgressMonitor
from core.backup_system import BackupSystem
from utils.logging_utils import get_logger, ComponentLogger

logger = get_logger(__name__)


class BatchStatus(Enum):
    """Status of batch processing"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


@dataclass
class BatchCheckpoint:
    """Checkpoint data for batch recovery"""
    batch_id: str
    checkpoint_id: str
    timestamp: datetime
    processed_count: int
    total_count: int
    current_batch_index: int
    successful_contexts: List[Dict[str, Any]]
    failed_contexts: List[Dict[str, Any]]
    batch_config: Dict[str, Any]
    memory_usage: float
    processing_time: float


@dataclass
class BatchMetrics:
    """Metrics for batch processing"""
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    skipped_items: int
    current_batch_size: int
    batches_completed: int
    total_batches: int
    processing_rate: float  # items per second
    estimated_time_remaining: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_rate: float
    fallback_usage_rate: float


@dataclass
class BatchResult:
    """Result of batch processing operation"""
    batch_id: str
    status: BatchStatus
    metrics: BatchMetrics
    successful_contexts: List[EnhancementContext]
    failed_contexts: List[EnhancementContext]
    checkpoints_created: List[str]
    processing_time: float
    recovery_attempts: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class BatchProcessor:
    """Advanced batch processing system with recovery capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize batch processor"""
        self.config = config
        self.batch_config = config.get('batch_processing', {})
        
        # Component logger for detailed tracking
        self.component_logger = ComponentLogger('enhancement.batch_processor', 
                                               verbose=config.get('verbose', False))
        
        # Initialize core components
        self.degradation_framework = GracefulDegradationFramework(config)
        self.progress_monitor = ProgressMonitor(config)
        self.backup_system = BackupSystem(config)
        
        # Batch processing configuration
        self.default_batch_size = self.batch_config.get('default_batch_size', 50)
        self.max_batch_size = self.batch_config.get('max_batch_size', 200)
        self.min_batch_size = self.batch_config.get('min_batch_size', 10)
        self.adaptive_batch_sizing = self.batch_config.get('adaptive_batch_sizing', True)
        
        # Memory management
        self.max_memory_usage_mb = self.batch_config.get('max_memory_usage_mb', 1024)
        self.memory_check_interval = self.batch_config.get('memory_check_interval', 10)
        
        # Recovery configuration
        self.enable_checkpoints = self.batch_config.get('enable_checkpoints', True)
        self.checkpoint_interval = self.batch_config.get('checkpoint_interval', 100)
        self.max_recovery_attempts = self.batch_config.get('max_recovery_attempts', 3)
        self.checkpoint_dir = Path(self.batch_config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance optimization
        self.max_concurrent_batches = self.batch_config.get('max_concurrent_batches', 3)
        self.batch_timeout = self.batch_config.get('batch_timeout', 300.0)  # 5 minutes
        
        # Active batch tracking
        self.active_batches: Dict[str, BatchResult] = {}
        self.batch_semaphore = asyncio.Semaphore(self.max_concurrent_batches)
        
        logger.info("Batch processor initialized with recovery capabilities")
    
    async def process_bookmarks_batch(self, 
                                     bookmarks: List[Dict[str, Any]],
                                     enhancement_components: Dict[str, Callable],
                                     batch_id: Optional[str] = None,
                                     resume_from_checkpoint: Optional[str] = None) -> BatchResult:
        """Process bookmarks in batches with comprehensive recovery"""
        
        if batch_id is None:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.component_logger.debug_operation(
            "batch_processing_start",
            {
                'batch_id': batch_id,
                'total_bookmarks': len(bookmarks),
                'components': list(enhancement_components.keys()),
                'resume_from_checkpoint': resume_from_checkpoint is not None
            }
        )
        
        start_time = time.time()
        
        # Initialize batch result
        batch_result = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.PENDING,
            metrics=BatchMetrics(
                total_items=len(bookmarks),
                processed_items=0,
                successful_items=0,
                failed_items=0,
                skipped_items=0,
                current_batch_size=self.default_batch_size,
                batches_completed=0,
                total_batches=0,
                processing_rate=0.0,
                estimated_time_remaining=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                error_rate=0.0,
                fallback_usage_rate=0.0
            ),
            successful_contexts=[],
            failed_contexts=[],
            checkpoints_created=[],
            processing_time=0.0
        )
        
        self.active_batches[batch_id] = batch_result
        
        try:
            async with self.batch_semaphore:
                # Resume from checkpoint if specified
                if resume_from_checkpoint:
                    checkpoint = await self._load_checkpoint(resume_from_checkpoint)
                    if checkpoint:
                        batch_result = await self._resume_from_checkpoint(
                            checkpoint, bookmarks, enhancement_components
                        )
                    else:
                        batch_result.warnings.append(f"Could not load checkpoint: {resume_from_checkpoint}")
                
                if batch_result.status == BatchStatus.PENDING:
                    batch_result.status = BatchStatus.RUNNING
                    batch_result = await self._process_batches(
                        bookmarks, enhancement_components, batch_result, start_time
                    )
                
                # Final processing
                batch_result.processing_time = time.time() - start_time
                
                if batch_result.status == BatchStatus.RUNNING:
                    batch_result.status = BatchStatus.COMPLETED
                
                self.component_logger.debug_performance(
                    "batch_processing_complete",
                    batch_result.processing_time,
                    {
                        'total_processed': batch_result.metrics.processed_items,
                        'success_rate': batch_result.metrics.successful_items / max(1, batch_result.metrics.processed_items),
                        'error_rate': batch_result.metrics.error_rate,
                        'checkpoints_created': len(batch_result.checkpoints_created),
                        'recovery_attempts': batch_result.recovery_attempts
                    }
                )
                
                return batch_result
                
        except Exception as e:
            batch_result.status = BatchStatus.FAILED
            batch_result.errors.append(f"Batch processing failed: {e}")
            batch_result.processing_time = time.time() - start_time
            
            logger.error(f"Batch processing failed for {batch_id}: {e}", exc_info=True)
            return batch_result
            
        finally:
            # Cleanup
            if batch_id in self.active_batches:
                del self.active_batches[batch_id]
    
    async def _process_batches(self, 
                              bookmarks: List[Dict[str, Any]],
                              enhancement_components: Dict[str, Callable],
                              batch_result: BatchResult,
                              start_time: float) -> BatchResult:
        """Process bookmarks in adaptive batches"""
        
        total_bookmarks = len(bookmarks)
        current_batch_size = batch_result.metrics.current_batch_size
        processed_count = batch_result.metrics.processed_items
        
        # Calculate total batches
        remaining_items = total_bookmarks - processed_count
        batch_result.metrics.total_batches = (remaining_items + current_batch_size - 1) // current_batch_size
        
        batch_index = 0
        last_checkpoint_count = processed_count
        
        # Process in batches
        current_position = processed_count
        
        while current_position < total_bookmarks:
            batch_end = min(current_position + current_batch_size, total_bookmarks)
            batch_bookmarks = bookmarks[current_position:batch_end]
            
            self.component_logger.debug_operation(
                f"processing_batch_{batch_index}",
                {
                    'batch_start': current_position,
                    'batch_end': batch_end,
                    'batch_size': len(batch_bookmarks),
                    'current_batch_size': current_batch_size
                }
            )
            
            # Process current batch
            batch_contexts = await self._process_single_batch(
                batch_bookmarks, enhancement_components, batch_result.batch_id
            )
            
            # Update results
            for context in batch_contexts:
                if any(r.status.value in ['success', 'partial_success', 'degraded'] 
                      for r in context.component_results.values()):
                    batch_result.successful_contexts.append(context)
                else:
                    batch_result.failed_contexts.append(context)
            
            # Update metrics based on actual counts
            batch_result.metrics.successful_items = len(batch_result.successful_contexts)
            batch_result.metrics.failed_items = len(batch_result.failed_contexts)
            batch_result.metrics.processed_items = batch_result.metrics.successful_items + batch_result.metrics.failed_items
            
            batch_result.metrics.batches_completed += 1
            
            # Update metrics
            batch_result.processing_time = time.time() - start_time
            await self._update_batch_metrics(batch_result)
            
            # Create checkpoint if needed
            if (self.enable_checkpoints and 
                batch_result.metrics.processed_items - last_checkpoint_count >= self.checkpoint_interval):
                
                checkpoint_id = await self._create_checkpoint(batch_result, batch_index)
                if checkpoint_id:
                    batch_result.checkpoints_created.append(checkpoint_id)
                    last_checkpoint_count = batch_result.metrics.processed_items
            
            # Adaptive batch sizing
            if self.adaptive_batch_sizing:
                current_batch_size = await self._adjust_batch_size(
                    current_batch_size, batch_result.metrics
                )
                batch_result.metrics.current_batch_size = current_batch_size
            
            # Memory management
            if await self._check_memory_usage():
                batch_result.warnings.append("High memory usage detected, reducing batch size")
                current_batch_size = max(self.min_batch_size, current_batch_size // 2)
                batch_result.metrics.current_batch_size = current_batch_size
            
            batch_index += 1
            
            # Move to next batch
            current_position = batch_end
            
            # Allow for cancellation
            if batch_result.status == BatchStatus.CANCELLED:
                break
        
        return batch_result
    
    async def _process_single_batch(self, 
                                   batch_bookmarks: List[Dict[str, Any]],
                                   enhancement_components: Dict[str, Callable],
                                   batch_id: str) -> List[EnhancementContext]:
        """Process a single batch of bookmarks"""
        
        try:
            # Process bookmarks concurrently within the batch
            tasks = []
            for bookmark in batch_bookmarks:
                task = self.degradation_framework.safe_enhance_bookmark(
                    bookmark, enhancement_components
                )
                tasks.append(task)
            
            # Wait for all bookmarks in batch to complete
            contexts = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            valid_contexts = []
            for i, context in enumerate(contexts):
                if isinstance(context, Exception):
                    # Create error context
                    error_context = EnhancementContext(
                        bookmark_id=batch_bookmarks[i].get('id', 0),
                        original_bookmark=batch_bookmarks[i],
                        enhanced_bookmark=batch_bookmarks[i].copy()
                    )
                    error_context.global_errors.append(
                        self.degradation_framework.ComponentError(
                            component="batch_processing",
                            error_type=type(context).__name__,
                            message=str(context),
                            severity=self.degradation_framework.ErrorSeverity.HIGH,
                            timestamp=datetime.now()
                        )
                    )
                    valid_contexts.append(error_context)
                else:
                    valid_contexts.append(context)
            
            return valid_contexts
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Return error contexts for all bookmarks in batch
            error_contexts = []
            for bookmark in batch_bookmarks:
                error_context = EnhancementContext(
                    bookmark_id=bookmark.get('id', 0),
                    original_bookmark=bookmark,
                    enhanced_bookmark=bookmark.copy()
                )
                error_context.global_errors.append(
                    self.degradation_framework.ComponentError(
                        component="batch_processing",
                        error_type=type(e).__name__,
                        message=str(e),
                        severity=self.degradation_framework.ErrorSeverity.CRITICAL,
                        timestamp=datetime.now()
                    )
                )
                error_contexts.append(error_context)
            
            return error_contexts
    
    async def _update_batch_metrics(self, batch_result: BatchResult) -> None:
        """Update batch processing metrics"""
        try:
            metrics = batch_result.metrics
            
            # Calculate processing rate
            if batch_result.processing_time > 0.001:  # Avoid division by very small numbers
                metrics.processing_rate = metrics.processed_items / batch_result.processing_time
            
            # Calculate error rate
            if metrics.processed_items > 0:
                metrics.error_rate = metrics.failed_items / metrics.processed_items
            
            # Calculate fallback usage rate
            fallback_count = sum(
                1 for context in batch_result.successful_contexts + batch_result.failed_contexts
                for result in context.component_results.values()
                if result.fallback_used
            )
            if metrics.processed_items > 0:
                metrics.fallback_usage_rate = fallback_count / metrics.processed_items
            
            # Estimate remaining time
            if metrics.processing_rate > 0:
                remaining_items = metrics.total_items - metrics.processed_items
                metrics.estimated_time_remaining = remaining_items / metrics.processing_rate
            
            # System metrics
            metrics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
            metrics.cpu_usage_percent = psutil.cpu_percent()
            
        except Exception as e:
            logger.warning(f"Failed to update batch metrics: {e}")
    
    async def _adjust_batch_size(self, current_size: int, metrics: BatchMetrics) -> int:
        """Adjust batch size based on performance metrics"""
        try:
            # Increase batch size if performance is good
            if (metrics.error_rate < 0.1 and 
                metrics.memory_usage_mb < self.max_memory_usage_mb * 0.7 and
                metrics.cpu_usage_percent < 80):
                
                new_size = min(self.max_batch_size, int(current_size * 1.2))
                
            # Decrease batch size if there are issues
            elif (metrics.error_rate > 0.3 or 
                  metrics.memory_usage_mb > self.max_memory_usage_mb * 0.9 or
                  metrics.cpu_usage_percent > 90):
                
                new_size = max(self.min_batch_size, int(current_size * 0.8))
                
            else:
                new_size = current_size
            
            if new_size != current_size:
                self.component_logger.debug_operation(
                    "batch_size_adjustment",
                    {
                        'old_size': current_size,
                        'new_size': new_size,
                        'error_rate': metrics.error_rate,
                        'memory_usage_mb': metrics.memory_usage_mb,
                        'cpu_usage_percent': metrics.cpu_usage_percent
                    }
                )
            
            return new_size
            
        except Exception as e:
            logger.warning(f"Failed to adjust batch size: {e}")
            return current_size
    
    async def _check_memory_usage(self) -> bool:
        """Check if memory usage is too high"""
        try:
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            return memory_mb > self.max_memory_usage_mb
        except Exception:
            return False
    
    async def _create_checkpoint(self, batch_result: BatchResult, batch_index: int) -> Optional[str]:
        """Create a checkpoint for recovery"""
        try:
            checkpoint_id = f"{batch_result.batch_id}_checkpoint_{batch_index}_{int(time.time())}"
            
            # Serialize contexts for checkpoint
            successful_data = []
            for context in batch_result.successful_contexts:
                successful_data.append({
                    'bookmark_id': context.bookmark_id,
                    'original_bookmark': context.original_bookmark,
                    'enhanced_bookmark': context.enhanced_bookmark,
                    'processing_time': (datetime.now() - context.start_time).total_seconds()
                })
            
            failed_data = []
            for context in batch_result.failed_contexts:
                failed_data.append({
                    'bookmark_id': context.bookmark_id,
                    'original_bookmark': context.original_bookmark,
                    'error_count': len(context.global_errors) + sum(len(r.errors) for r in context.component_results.values())
                })
            
            checkpoint = BatchCheckpoint(
                batch_id=batch_result.batch_id,
                checkpoint_id=checkpoint_id,
                timestamp=datetime.now(),
                processed_count=batch_result.metrics.processed_items,
                total_count=batch_result.metrics.total_items,
                current_batch_index=batch_index,
                successful_contexts=successful_data,
                failed_contexts=failed_data,
                batch_config={
                    'current_batch_size': batch_result.metrics.current_batch_size,
                    'batches_completed': batch_result.metrics.batches_completed
                },
                memory_usage=batch_result.metrics.memory_usage_mb,
                processing_time=batch_result.processing_time
            )
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            self.component_logger.debug_operation(
                "checkpoint_created",
                {
                    'checkpoint_id': checkpoint_id,
                    'processed_count': checkpoint.processed_count,
                    'total_count': checkpoint.total_count,
                    'checkpoint_path': str(checkpoint_path)
                }
            )
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return None
    
    async def _load_checkpoint(self, checkpoint_id: str) -> Optional[BatchCheckpoint]:
        """Load a checkpoint for recovery"""
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint file not found: {checkpoint_path}")
                return None
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.component_logger.debug_operation(
                "checkpoint_loaded",
                {
                    'checkpoint_id': checkpoint_id,
                    'processed_count': checkpoint.processed_count,
                    'total_count': checkpoint.total_count,
                    'timestamp': checkpoint.timestamp.isoformat()
                }
            )
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    async def _resume_from_checkpoint(self, 
                                     checkpoint: BatchCheckpoint,
                                     bookmarks: List[Dict[str, Any]],
                                     enhancement_components: Dict[str, Callable]) -> BatchResult:
        """Resume processing from a checkpoint"""
        
        self.component_logger.debug_operation(
            "resuming_from_checkpoint",
            {
                'checkpoint_id': checkpoint.checkpoint_id,
                'processed_count': checkpoint.processed_count,
                'total_count': checkpoint.total_count
            }
        )
        
        # Reconstruct batch result from checkpoint
        batch_result = BatchResult(
            batch_id=checkpoint.batch_id,
            status=BatchStatus.RECOVERING,
            metrics=BatchMetrics(
                total_items=checkpoint.total_count,
                processed_items=checkpoint.processed_count,
                successful_items=len(checkpoint.successful_contexts),
                failed_items=len(checkpoint.failed_contexts),
                skipped_items=0,
                current_batch_size=checkpoint.batch_config.get('current_batch_size', self.default_batch_size),
                batches_completed=checkpoint.batch_config.get('batches_completed', 0),
                total_batches=0,
                processing_rate=0.0,
                estimated_time_remaining=0.0,
                memory_usage_mb=checkpoint.memory_usage,
                cpu_usage_percent=0.0,
                error_rate=0.0,
                fallback_usage_rate=0.0
            ),
            successful_contexts=[],
            failed_contexts=[],
            checkpoints_created=[checkpoint.checkpoint_id],
            processing_time=checkpoint.processing_time,
            recovery_attempts=1
        )
        
        # Reconstruct contexts (simplified for checkpoint)
        for ctx_data in checkpoint.successful_contexts:
            context = EnhancementContext(
                bookmark_id=ctx_data['bookmark_id'],
                original_bookmark=ctx_data['original_bookmark'],
                enhanced_bookmark=ctx_data['enhanced_bookmark']
            )
            batch_result.successful_contexts.append(context)
        
        for ctx_data in checkpoint.failed_contexts:
            context = EnhancementContext(
                bookmark_id=ctx_data['bookmark_id'],
                original_bookmark=ctx_data['original_bookmark'],
                enhanced_bookmark=ctx_data['original_bookmark'].copy()
            )
            batch_result.failed_contexts.append(context)
        
        batch_result.status = BatchStatus.RUNNING
        return batch_result
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a batch"""
        if batch_id not in self.active_batches:
            return None
        
        batch_result = self.active_batches[batch_id]
        return {
            'batch_id': batch_id,
            'status': batch_result.status.value,
            'metrics': {
                'total_items': batch_result.metrics.total_items,
                'processed_items': batch_result.metrics.processed_items,
                'successful_items': batch_result.metrics.successful_items,
                'failed_items': batch_result.metrics.failed_items,
                'processing_rate': batch_result.metrics.processing_rate,
                'estimated_time_remaining': batch_result.metrics.estimated_time_remaining,
                'memory_usage_mb': batch_result.metrics.memory_usage_mb,
                'error_rate': batch_result.metrics.error_rate
            },
            'checkpoints_created': len(batch_result.checkpoints_created),
            'processing_time': batch_result.processing_time
        }
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch"""
        if batch_id in self.active_batches:
            self.active_batches[batch_id].status = BatchStatus.CANCELLED
            logger.info(f"Batch {batch_id} cancelled")
            return True
        return False
    
    def list_checkpoints(self, batch_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available checkpoints"""
        try:
            checkpoints = []
            
            for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
                try:
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint = pickle.load(f)
                    
                    if batch_id is None or checkpoint.batch_id == batch_id:
                        checkpoints.append({
                            'checkpoint_id': checkpoint.checkpoint_id,
                            'batch_id': checkpoint.batch_id,
                            'timestamp': checkpoint.timestamp.isoformat(),
                            'processed_count': checkpoint.processed_count,
                            'total_count': checkpoint.total_count,
                            'success_rate': len(checkpoint.successful_contexts) / max(1, checkpoint.processed_count),
                            'file_size_mb': checkpoint_file.stat().st_size / 1024 / 1024
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
            
            return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """Clean up old checkpoint files"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            deleted_count = 0
            
            for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
                try:
                    file_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        checkpoint_file.unlink()
                        deleted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to delete checkpoint {checkpoint_file}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old checkpoint files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return 0
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        return {
            'config': self.batch_config,
            'active_batches': len(self.active_batches),
            'max_concurrent_batches': self.max_concurrent_batches,
            'default_batch_size': self.default_batch_size,
            'adaptive_batch_sizing': self.adaptive_batch_sizing,
            'checkpoints_enabled': self.enable_checkpoints,
            'checkpoint_interval': self.checkpoint_interval,
            'available_checkpoints': len(list(self.checkpoint_dir.glob("*.pkl"))),
            'memory_limits': {
                'max_memory_usage_mb': self.max_memory_usage_mb,
                'current_memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }
        }