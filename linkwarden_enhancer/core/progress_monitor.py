"""Progress Monitor - Real-time progress tracking with safety thresholds"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class OperationStatus(Enum):
    """Operation status enumeration"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressInfo:
    """Progress information for an operation"""
    operation_name: str
    current_item: int
    total_items: int
    percentage: float
    eta_seconds: Optional[float]
    items_per_second: float
    elapsed_time: float
    status: OperationStatus
    current_task: str
    warnings: List[str]
    errors: List[str]


@dataclass
class SafetyThreshold:
    """Safety threshold configuration"""
    name: str
    threshold_value: float
    threshold_type: str  # 'percentage', 'count', 'ratio'
    description: str
    severity: str  # 'warning', 'error', 'critical'
    auto_continue: bool = False


class ProgressMonitor:
    """Real-time progress monitoring with safety thresholds"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize progress monitor with configuration"""
        self.config = config
        self.progress_config = config.get('progress', {})
        self.safety_config = config.get('safety', {})
        
        # Progress tracking
        self.operations = {}
        self.current_operation = None
        
        # Safety thresholds
        self.safety_thresholds = self._load_safety_thresholds()
        
        # User interaction callbacks
        self.confirmation_callback: Optional[Callable[[str, Dict[str, Any]], bool]] = None
        self.progress_callback: Optional[Callable[[ProgressInfo], None]] = None
        
        # Statistics
        self.operation_history = []
        
        logger.info("Progress monitor initialized")
    
    def start_operation(self, 
                       operation_name: str, 
                       total_items: int,
                       description: str = "") -> str:
        """Start tracking a new operation"""
        
        try:
            operation_id = f"{operation_name}_{int(time.time())}"
            
            progress_info = ProgressInfo(
                operation_name=operation_name,
                current_item=0,
                total_items=total_items,
                percentage=0.0,
                eta_seconds=None,
                items_per_second=0.0,
                elapsed_time=0.0,
                status=OperationStatus.RUNNING,
                current_task=description or "Starting...",
                warnings=[],
                errors=[]
            )
            
            self.operations[operation_id] = {
                'progress': progress_info,
                'start_time': time.time(),
                'last_update': time.time(),
                'description': description,
                'item_times': []  # For ETA calculation
            }
            
            self.current_operation = operation_id
            
            logger.info(f"Started operation: {operation_name} ({total_items} items)")
            self._notify_progress_update(progress_info)
            
            return operation_id
            
        except Exception as e:
            logger.error(f"Failed to start operation: {e}")
            return ""
    
    def update_progress(self, 
                       operation_id: str, 
                       current_item: int,
                       current_task: str = "",
                       additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """Update progress for an operation"""
        
        try:
            if operation_id not in self.operations:
                logger.error(f"Unknown operation: {operation_id}")
                return False
            
            operation = self.operations[operation_id]
            progress = operation['progress']
            current_time = time.time()
            
            # Update basic progress
            progress.current_item = current_item
            progress.percentage = (current_item / progress.total_items) * 100 if progress.total_items > 0 else 0
            progress.elapsed_time = current_time - operation['start_time']
            
            if current_task:
                progress.current_task = current_task
            
            # Calculate items per second
            time_since_last = current_time - operation['last_update']
            if time_since_last > 0:
                items_since_last = current_item - (operation.get('last_item', 0))
                progress.items_per_second = items_since_last / time_since_last
            
            # Calculate ETA
            if progress.items_per_second > 0:
                remaining_items = progress.total_items - current_item
                progress.eta_seconds = remaining_items / progress.items_per_second
            
            # Update operation data
            operation['last_update'] = current_time
            operation['last_item'] = current_item
            
            # Track item processing times for better ETA
            operation['item_times'].append(current_time)
            if len(operation['item_times']) > 100:  # Keep last 100 times
                operation['item_times'] = operation['item_times'][-100:]
            
            # Check safety thresholds
            safety_violations = self._check_safety_thresholds(operation_id, additional_data)
            if safety_violations:
                if not self._handle_safety_violations(operation_id, safety_violations):
                    return False  # Operation should be cancelled
            
            # Notify progress update
            self._notify_progress_update(progress)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update progress: {e}")
            return False
    
    def add_warning(self, operation_id: str, warning: str) -> None:
        """Add a warning to the operation"""
        
        if operation_id in self.operations:
            self.operations[operation_id]['progress'].warnings.append(warning)
            logger.warning(f"Operation {operation_id}: {warning}")
    
    def add_error(self, operation_id: str, error: str) -> None:
        """Add an error to the operation"""
        
        if operation_id in self.operations:
            self.operations[operation_id]['progress'].errors.append(error)
            logger.error(f"Operation {operation_id}: {error}")
    
    def complete_operation(self, operation_id: str, success: bool = True) -> bool:
        """Mark operation as completed"""
        
        try:
            if operation_id not in self.operations:
                logger.error(f"Unknown operation: {operation_id}")
                return False
            
            operation = self.operations[operation_id]
            progress = operation['progress']
            
            progress.status = OperationStatus.COMPLETED if success else OperationStatus.FAILED
            progress.percentage = 100.0 if success else progress.percentage
            progress.current_task = "Completed" if success else "Failed"
            progress.elapsed_time = time.time() - operation['start_time']
            
            # Add to history
            self.operation_history.append({
                'operation_id': operation_id,
                'operation_name': progress.operation_name,
                'completed_at': datetime.now(),
                'success': success,
                'total_items': progress.total_items,
                'elapsed_time': progress.elapsed_time,
                'warnings_count': len(progress.warnings),
                'errors_count': len(progress.errors)
            })
            
            # Keep history limited
            if len(self.operation_history) > 50:
                self.operation_history = self.operation_history[-50:]
            
            logger.info(f"Operation completed: {progress.operation_name} ({'success' if success else 'failed'})")
            self._notify_progress_update(progress)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete operation: {e}")
            return False
    
    def cancel_operation(self, operation_id: str) -> bool:
        """Cancel a running operation"""
        
        try:
            if operation_id not in self.operations:
                logger.error(f"Unknown operation: {operation_id}")
                return False
            
            operation = self.operations[operation_id]
            progress = operation['progress']
            
            progress.status = OperationStatus.CANCELLED
            progress.current_task = "Cancelled"
            
            logger.info(f"Operation cancelled: {progress.operation_name}")
            self._notify_progress_update(progress)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel operation: {e}")
            return False
    
    def get_progress(self, operation_id: str) -> Optional[ProgressInfo]:
        """Get current progress for an operation"""
        
        if operation_id in self.operations:
            return self.operations[operation_id]['progress']
        return None
    
    def get_current_progress(self) -> Optional[ProgressInfo]:
        """Get progress for the current operation"""
        
        if self.current_operation:
            return self.get_progress(self.current_operation)
        return None
    
    def list_operations(self) -> List[Dict[str, Any]]:
        """List all tracked operations"""
        
        operations = []
        for op_id, op_data in self.operations.items():
            progress = op_data['progress']
            operations.append({
                'operation_id': op_id,
                'operation_name': progress.operation_name,
                'status': progress.status.value,
                'percentage': progress.percentage,
                'current_item': progress.current_item,
                'total_items': progress.total_items,
                'elapsed_time': progress.elapsed_time,
                'eta_seconds': progress.eta_seconds,
                'warnings_count': len(progress.warnings),
                'errors_count': len(progress.errors)
            })
        
        return operations
    
    def set_confirmation_callback(self, callback: Callable[[str, Dict[str, Any]], bool]) -> None:
        """Set callback for user confirmation prompts"""
        self.confirmation_callback = callback
    
    def set_progress_callback(self, callback: Callable[[ProgressInfo], None]) -> None:
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def _load_safety_thresholds(self) -> List[SafetyThreshold]:
        """Load safety threshold configurations"""
        
        default_thresholds = [
            SafetyThreshold(
                name="deletion_percentage",
                threshold_value=self.safety_config.get('max_deletion_percentage', 10.0),
                threshold_type="percentage",
                description="Maximum percentage of items that can be deleted",
                severity="critical",
                auto_continue=False
            ),
            SafetyThreshold(
                name="error_rate",
                threshold_value=self.safety_config.get('max_error_rate', 5.0),
                threshold_type="percentage",
                description="Maximum error rate during processing",
                severity="warning",
                auto_continue=False
            ),
            SafetyThreshold(
                name="processing_time",
                threshold_value=self.safety_config.get('max_processing_minutes', 60.0),
                threshold_type="count",
                description="Maximum processing time in minutes",
                severity="warning",
                auto_continue=True
            )
        ]
        
        # Load custom thresholds from config
        custom_thresholds = self.safety_config.get('custom_thresholds', [])
        for threshold_config in custom_thresholds:
            try:
                threshold = SafetyThreshold(**threshold_config)
                default_thresholds.append(threshold)
            except Exception as e:
                logger.warning(f"Invalid safety threshold config: {e}")
        
        return default_thresholds
    
    def _check_safety_thresholds(self, 
                                operation_id: str, 
                                additional_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check if any safety thresholds are violated"""
        
        violations = []
        operation = self.operations[operation_id]
        progress = operation['progress']
        
        try:
            for threshold in self.safety_thresholds:
                violation = None
                
                if threshold.name == "deletion_percentage" and additional_data:
                    deletions = additional_data.get('deletions', 0)
                    total = additional_data.get('total_items', progress.total_items)
                    if total > 0:
                        deletion_percentage = (deletions / total) * 100
                        if deletion_percentage > threshold.threshold_value:
                            violation = {
                                'threshold': threshold,
                                'current_value': deletion_percentage,
                                'message': f"Deletion rate ({deletion_percentage:.1f}%) exceeds threshold ({threshold.threshold_value}%)"
                            }
                
                elif threshold.name == "error_rate":
                    total_processed = progress.current_item
                    if total_processed > 0:
                        error_rate = (len(progress.errors) / total_processed) * 100
                        if error_rate > threshold.threshold_value:
                            violation = {
                                'threshold': threshold,
                                'current_value': error_rate,
                                'message': f"Error rate ({error_rate:.1f}%) exceeds threshold ({threshold.threshold_value}%)"
                            }
                
                elif threshold.name == "processing_time":
                    elapsed_minutes = progress.elapsed_time / 60
                    if elapsed_minutes > threshold.threshold_value:
                        violation = {
                            'threshold': threshold,
                            'current_value': elapsed_minutes,
                            'message': f"Processing time ({elapsed_minutes:.1f} min) exceeds threshold ({threshold.threshold_value} min)"
                        }
                
                if violation:
                    violations.append(violation)
            
        except Exception as e:
            logger.error(f"Error checking safety thresholds: {e}")
        
        return violations
    
    def _handle_safety_violations(self, operation_id: str, violations: List[Dict[str, Any]]) -> bool:
        """Handle safety threshold violations"""
        
        try:
            operation = self.operations[operation_id]
            progress = operation['progress']
            
            critical_violations = [v for v in violations if v['threshold'].severity == 'critical']
            warning_violations = [v for v in violations if v['threshold'].severity == 'warning']
            
            # Log all violations
            for violation in violations:
                threshold = violation['threshold']
                message = violation['message']
                
                if threshold.severity == 'critical':
                    logger.critical(f"CRITICAL SAFETY VIOLATION: {message}")
                    progress.errors.append(f"SAFETY: {message}")
                elif threshold.severity == 'warning':
                    logger.warning(f"Safety warning: {message}")
                    progress.warnings.append(f"SAFETY: {message}")
            
            # Handle critical violations
            if critical_violations:
                # Pause operation
                progress.status = OperationStatus.PAUSED
                progress.current_task = "PAUSED - Safety violation detected"
                
                # Request user confirmation if callback available
                if self.confirmation_callback:
                    violation_summary = {
                        'operation_name': progress.operation_name,
                        'violations': violations,
                        'current_progress': progress.percentage,
                        'items_processed': progress.current_item
                    }
                    
                    continue_operation = self.confirmation_callback(
                        "Critical safety threshold violated. Continue operation?",
                        violation_summary
                    )
                    
                    if continue_operation:
                        progress.status = OperationStatus.RUNNING
                        progress.current_task = "Continuing after safety confirmation"
                        logger.info("User confirmed to continue despite safety violations")
                        return True
                    else:
                        progress.status = OperationStatus.CANCELLED
                        progress.current_task = "Cancelled due to safety violation"
                        logger.info("Operation cancelled due to safety violations")
                        return False
                else:
                    # No callback available, cancel operation
                    progress.status = OperationStatus.CANCELLED
                    progress.current_task = "Cancelled due to safety violation"
                    logger.error("Operation cancelled due to critical safety violation (no user confirmation available)")
                    return False
            
            # Handle warning violations with auto-continue
            auto_continue_violations = [v for v in warning_violations if v['threshold'].auto_continue]
            if auto_continue_violations:
                logger.info(f"Continuing operation despite {len(auto_continue_violations)} auto-continue violations")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling safety violations: {e}")
            return False
    
    def _notify_progress_update(self, progress: ProgressInfo) -> None:
        """Notify progress callback if available"""
        
        try:
            if self.progress_callback:
                self.progress_callback(progress)
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get operation statistics"""
        
        try:
            active_operations = len([op for op in self.operations.values() 
                                   if op['progress'].status == OperationStatus.RUNNING])
            
            completed_operations = len([op for op in self.operation_history if op['success']])
            failed_operations = len([op for op in self.operation_history if not op['success']])
            
            total_processing_time = sum(op['elapsed_time'] for op in self.operation_history)
            avg_processing_time = total_processing_time / len(self.operation_history) if self.operation_history else 0
            
            return {
                'active_operations': active_operations,
                'total_operations': len(self.operations),
                'completed_operations': completed_operations,
                'failed_operations': failed_operations,
                'success_rate': (completed_operations / len(self.operation_history) * 100) if self.operation_history else 0,
                'average_processing_time': avg_processing_time,
                'total_processing_time': total_processing_time,
                'safety_thresholds_count': len(self.safety_thresholds),
                'history_entries': len(self.operation_history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get operation statistics: {e}")
            return {'error': str(e)}