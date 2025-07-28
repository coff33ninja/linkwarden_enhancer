"""Metrics collector for performance tracking and system monitoring"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class OperationMetrics:
    """Metrics for a specific operation"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = True
    items_processed: int = 0
    errors_count: int = 0
    warnings_count: int = 0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    cpu_usage: List[float] = field(default_factory=list)
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Collect and track performance metrics for all system components"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize metrics collector"""
        self.config = config
        self.metrics_config = config.get('metrics', {})
        
        # Metrics storage
        self.performance_metrics = deque(maxlen=10000)  # Keep last 10k metrics
        self.operation_metrics = {}  # Active operations
        self.completed_operations = deque(maxlen=1000)  # Keep last 1k completed operations
        
        # Aggregated metrics
        self.component_metrics = defaultdict(lambda: defaultdict(list))
        self.system_metrics = defaultdict(list)
        
        # Monitoring settings
        self.collect_system_metrics = self.metrics_config.get('collect_system_metrics', True)
        self.metric_collection_interval = self.metrics_config.get('collection_interval', 60)  # seconds
        self.enable_detailed_profiling = self.metrics_config.get('detailed_profiling', False)
        
        # Background monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Performance thresholds
        self.performance_thresholds = {
            'memory_usage_mb': self.metrics_config.get('memory_threshold_mb', 1000),
            'cpu_usage_percent': self.metrics_config.get('cpu_threshold_percent', 80),
            'operation_duration_seconds': self.metrics_config.get('duration_threshold_seconds', 300),
            'error_rate_percent': self.metrics_config.get('error_rate_threshold_percent', 5)
        }
        
        # Start background monitoring if enabled
        if self.collect_system_metrics:
            self.start_monitoring()
        
        logger.info("Metrics collector initialized")
    
    def start_monitoring(self) -> None:
        """Start background system metrics monitoring"""
        
        if self.monitoring_active:
            return
        
        try:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Background metrics monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.monitoring_active = False
    
    def stop_monitoring(self) -> None:
        """Stop background system metrics monitoring"""
        
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Background metrics monitoring stopped")
    
    def record_metric(self, metric_name: str, value: float, unit: str = "",
                     context: Optional[Dict[str, Any]] = None,
                     tags: Optional[Dict[str, str]] = None) -> None:
        """Record a performance metric"""
        
        try:
            metric = PerformanceMetric(
                metric_name=metric_name,
                value=value,
                unit=unit,
                timestamp=datetime.now(),
                context=context or {},
                tags=tags or {}
            )
            
            self.performance_metrics.append(metric)
            
            # Add to component metrics for aggregation
            component = tags.get('component', 'system') if tags else 'system'
            self.component_metrics[component][metric_name].append(value)
            
            logger.debug(f"Recorded metric: {metric_name} = {value} {unit}")
            
        except Exception as e:
            logger.error(f"Failed to record metric: {e}")
    
    def start_operation(self, operation_name: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking metrics for an operation"""
        
        try:
            operation_id = f"{operation_name}_{int(time.time())}_{len(self.operation_metrics)}"
            
            operation_metrics = OperationMetrics(
                operation_name=operation_name,
                start_time=datetime.now()
            )
            
            # Record initial system state
            if self.collect_system_metrics:
                operation_metrics.memory_usage['start'] = self._get_memory_usage()
                operation_metrics.cpu_usage.append(self._get_cpu_usage())
            
            self.operation_metrics[operation_id] = operation_metrics
            
            logger.debug(f"Started operation tracking: {operation_name} ({operation_id})")
            return operation_id
            
        except Exception as e:
            logger.error(f"Failed to start operation tracking: {e}")
            return ""
    
    def end_operation(self, operation_id: str, success: bool = True,
                     items_processed: int = 0, errors_count: int = 0,
                     warnings_count: int = 0, custom_metrics: Optional[Dict[str, float]] = None) -> None:
        """End tracking metrics for an operation"""
        
        try:
            if operation_id not in self.operation_metrics:
                logger.warning(f"Unknown operation ID: {operation_id}")
                return
            
            operation = self.operation_metrics[operation_id]
            operation.end_time = datetime.now()
            operation.duration = (operation.end_time - operation.start_time).total_seconds()
            operation.success = success
            operation.items_processed = items_processed
            operation.errors_count = errors_count
            operation.warnings_count = warnings_count
            operation.custom_metrics = custom_metrics or {}
            
            # Record final system state
            if self.collect_system_metrics:
                operation.memory_usage['end'] = self._get_memory_usage()
                operation.cpu_usage.append(self._get_cpu_usage())
            
            # Move to completed operations
            self.completed_operations.append(operation)
            del self.operation_metrics[operation_id]
            
            # Record operation metrics
            self._record_operation_metrics(operation)
            
            # Check performance thresholds
            self._check_performance_thresholds(operation)
            
            logger.debug(f"Ended operation tracking: {operation.operation_name} ({operation.duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Failed to end operation tracking: {e}")
    
    @contextmanager
    def track_operation(self, operation_name: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for tracking operation metrics"""
        
        operation_id = self.start_operation(operation_name, context)
        success = True
        items_processed = 0
        errors_count = 0
        warnings_count = 0
        custom_metrics = {}
        
        try:
            yield {
                'operation_id': operation_id,
                'set_items_processed': lambda count: setattr(self, '_temp_items', count),
                'add_error': lambda: setattr(self, '_temp_errors', getattr(self, '_temp_errors', 0) + 1),
                'add_warning': lambda: setattr(self, '_temp_warnings', getattr(self, '_temp_warnings', 0) + 1),
                'add_metric': lambda name, value: custom_metrics.update({name: value})
            }
            
        except Exception as e:
            success = False
            errors_count = getattr(self, '_temp_errors', 0) + 1
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
            
        finally:
            items_processed = getattr(self, '_temp_items', 0)
            errors_count = getattr(self, '_temp_errors', 0)
            warnings_count = getattr(self, '_temp_warnings', 0)
            
            # Clean up temporary attributes
            for attr in ['_temp_items', '_temp_errors', '_temp_warnings']:
                if hasattr(self, attr):
                    delattr(self, attr)
            
            self.end_operation(operation_id, success, items_processed, errors_count, warnings_count, custom_metrics)
    
    def record_component_performance(self, component_name: str, operation: str,
                                   duration: float, success: bool = True,
                                   additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """Record performance metrics for a specific component"""
        
        try:
            tags = {'component': component_name, 'operation': operation}
            
            # Record duration
            self.record_metric(f"{component_name}_duration", duration, "seconds", tags=tags)
            
            # Record success/failure
            self.record_metric(f"{component_name}_success", 1.0 if success else 0.0, "boolean", tags=tags)
            
            # Record additional metrics
            if additional_metrics:
                for metric_name, value in additional_metrics.items():
                    self.record_metric(f"{component_name}_{metric_name}", value, "", tags=tags)
            
            logger.debug(f"Recorded component performance: {component_name}.{operation} ({duration:.3f}s)")
            
        except Exception as e:
            logger.error(f"Failed to record component performance: {e}")
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time window"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Filter recent metrics
            recent_metrics = [
                metric for metric in self.performance_metrics
                if metric.timestamp >= cutoff_time
            ]
            
            # Filter recent operations
            recent_operations = [
                op for op in self.completed_operations
                if op.start_time >= cutoff_time
            ]
            
            summary = {
                'time_window_hours': time_window_hours,
                'metrics_collected': len(recent_metrics),
                'operations_completed': len(recent_operations),
                'system_performance': self._calculate_system_performance(recent_metrics),
                'operation_performance': self._calculate_operation_performance(recent_operations),
                'component_performance': self._calculate_component_performance(recent_metrics),
                'performance_trends': self._calculate_performance_trends(recent_metrics),
                'threshold_violations': self._get_threshold_violations(recent_operations)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {'error': str(e)}
    
    def get_operation_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed statistics for operations"""
        
        try:
            # Filter operations by name if specified
            operations = self.completed_operations
            if operation_name:
                operations = [op for op in operations if op.operation_name == operation_name]
            
            if not operations:
                return {'message': 'No operations found'}
            
            # Calculate statistics
            durations = [op.duration for op in operations if op.duration]
            success_count = sum(1 for op in operations if op.success)
            total_items = sum(op.items_processed for op in operations)
            total_errors = sum(op.errors_count for op in operations)
            
            statistics = {
                'total_operations': len(operations),
                'successful_operations': success_count,
                'success_rate': success_count / len(operations) if operations else 0,
                'total_items_processed': total_items,
                'total_errors': total_errors,
                'error_rate': total_errors / total_items if total_items > 0 else 0,
                'duration_statistics': {
                    'min': min(durations) if durations else 0,
                    'max': max(durations) if durations else 0,
                    'average': sum(durations) / len(durations) if durations else 0,
                    'total': sum(durations) if durations else 0
                },
                'throughput': {
                    'items_per_second': total_items / sum(durations) if durations and sum(durations) > 0 else 0,
                    'operations_per_hour': len(operations) / (sum(durations) / 3600) if durations and sum(durations) > 0 else 0
                }
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Failed to get operation statistics: {e}")
            return {'error': str(e)}
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""
        
        try:
            health = {
                'timestamp': datetime.now().isoformat(),
                'memory': {
                    'current_usage_mb': self._get_memory_usage(),
                    'threshold_mb': self.performance_thresholds['memory_usage_mb'],
                    'status': 'healthy'
                },
                'cpu': {
                    'current_usage_percent': self._get_cpu_usage(),
                    'threshold_percent': self.performance_thresholds['cpu_usage_percent'],
                    'status': 'healthy'
                },
                'active_operations': len(self.operation_metrics),
                'metrics_collected_today': len([
                    m for m in self.performance_metrics
                    if m.timestamp.date() == datetime.now().date()
                ]),
                'overall_status': 'healthy'
            }
            
            # Check thresholds
            if health['memory']['current_usage_mb'] > health['memory']['threshold_mb']:
                health['memory']['status'] = 'warning'
                health['overall_status'] = 'warning'
            
            if health['cpu']['current_usage_percent'] > health['cpu']['threshold_percent']:
                health['cpu']['status'] = 'warning'
                health['overall_status'] = 'warning'
            
            return health
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {'error': str(e), 'overall_status': 'error'}
    
    def export_metrics(self, format_type: str = 'json', time_window_hours: int = 24) -> Optional[str]:
        """Export metrics data in specified format"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Prepare export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'time_window_hours': time_window_hours,
                'performance_metrics': [
                    {
                        'metric_name': m.metric_name,
                        'value': m.value,
                        'unit': m.unit,
                        'timestamp': m.timestamp.isoformat(),
                        'context': m.context,
                        'tags': m.tags
                    }
                    for m in self.performance_metrics
                    if m.timestamp >= cutoff_time
                ],
                'completed_operations': [
                    {
                        'operation_name': op.operation_name,
                        'start_time': op.start_time.isoformat(),
                        'end_time': op.end_time.isoformat() if op.end_time else None,
                        'duration': op.duration,
                        'success': op.success,
                        'items_processed': op.items_processed,
                        'errors_count': op.errors_count,
                        'warnings_count': op.warnings_count,
                        'memory_usage': op.memory_usage,
                        'cpu_usage': op.cpu_usage,
                        'custom_metrics': op.custom_metrics
                    }
                    for op in self.completed_operations
                    if op.start_time >= cutoff_time
                ]
            }
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"metrics_export_{timestamp}.{format_type}"
            filepath = self.config.get('directories', {}).get('data_dir', 'data')
            full_path = f"{filepath}/{filename}"
            
            if format_type == 'json':
                import json
                with open(full_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                logger.error(f"Unsupported export format: {format_type}")
                return None
            
            logger.info(f"Metrics exported to {full_path}")
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return None    

    def _monitoring_loop(self) -> None:
        """Background monitoring loop for system metrics"""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.record_metric('system_memory_usage', self._get_memory_usage(), 'MB', tags={'component': 'system'})
                self.record_metric('system_cpu_usage', self._get_cpu_usage(), '%', tags={'component': 'system'})
                
                # Collect disk usage if available
                try:
                    disk_usage = psutil.disk_usage('/')
                    self.record_metric('system_disk_usage', disk_usage.percent, '%', tags={'component': 'system'})
                except:
                    pass  # Ignore disk usage errors
                
                # Sleep until next collection
                time.sleep(self.metric_collection_interval)
                
            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
                time.sleep(self.metric_collection_interval)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _record_operation_metrics(self, operation: OperationMetrics) -> None:
        """Record metrics from completed operation"""
        
        try:
            tags = {'component': 'operations', 'operation': operation.operation_name}
            
            # Record duration
            if operation.duration:
                self.record_metric('operation_duration', operation.duration, 'seconds', tags=tags)
            
            # Record success/failure
            self.record_metric('operation_success', 1.0 if operation.success else 0.0, 'boolean', tags=tags)
            
            # Record items processed
            if operation.items_processed > 0:
                self.record_metric('operation_items_processed', operation.items_processed, 'count', tags=tags)
                
                # Calculate throughput
                if operation.duration and operation.duration > 0:
                    throughput = operation.items_processed / operation.duration
                    self.record_metric('operation_throughput', throughput, 'items/second', tags=tags)
            
            # Record errors and warnings
            if operation.errors_count > 0:
                self.record_metric('operation_errors', operation.errors_count, 'count', tags=tags)
            
            if operation.warnings_count > 0:
                self.record_metric('operation_warnings', operation.warnings_count, 'count', tags=tags)
            
            # Record memory usage change
            if 'start' in operation.memory_usage and 'end' in operation.memory_usage:
                memory_delta = operation.memory_usage['end'] - operation.memory_usage['start']
                self.record_metric('operation_memory_delta', memory_delta, 'MB', tags=tags)
            
            # Record custom metrics
            for metric_name, value in operation.custom_metrics.items():
                self.record_metric(f'operation_{metric_name}', value, '', tags=tags)
            
        except Exception as e:
            logger.warning(f"Failed to record operation metrics: {e}")
    
    def _check_performance_thresholds(self, operation: OperationMetrics) -> None:
        """Check if operation exceeded performance thresholds"""
        
        try:
            violations = []
            
            # Check duration threshold
            if (operation.duration and 
                operation.duration > self.performance_thresholds['operation_duration_seconds']):
                violations.append(f"Duration exceeded threshold: {operation.duration:.2f}s")
            
            # Check error rate threshold
            if operation.items_processed > 0:
                error_rate = (operation.errors_count / operation.items_processed) * 100
                if error_rate > self.performance_thresholds['error_rate_percent']:
                    violations.append(f"Error rate exceeded threshold: {error_rate:.1f}%")
            
            # Check memory usage
            if 'end' in operation.memory_usage:
                if operation.memory_usage['end'] > self.performance_thresholds['memory_usage_mb']:
                    violations.append(f"Memory usage exceeded threshold: {operation.memory_usage['end']:.1f}MB")
            
            # Log violations
            if violations:
                logger.warning(f"Performance threshold violations for {operation.operation_name}: {'; '.join(violations)}")
                
                # Record threshold violation metric
                self.record_metric('threshold_violations', len(violations), 'count', 
                                 tags={'component': 'performance', 'operation': operation.operation_name})
            
        except Exception as e:
            logger.warning(f"Failed to check performance thresholds: {e}")
    
    def _calculate_system_performance(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate system performance metrics"""
        
        try:
            system_metrics = [m for m in metrics if m.tags.get('component') == 'system']
            
            if not system_metrics:
                return {}
            
            # Group by metric name
            metrics_by_name = defaultdict(list)
            for metric in system_metrics:
                metrics_by_name[metric.metric_name].append(metric.value)
            
            performance = {}
            for metric_name, values in metrics_by_name.items():
                performance[metric_name] = {
                    'min': min(values),
                    'max': max(values),
                    'average': sum(values) / len(values),
                    'current': values[-1] if values else 0
                }
            
            return performance
            
        except Exception as e:
            logger.warning(f"Failed to calculate system performance: {e}")
            return {}
    
    def _calculate_operation_performance(self, operations: List[OperationMetrics]) -> Dict[str, Any]:
        """Calculate operation performance metrics"""
        
        try:
            if not operations:
                return {}
            
            # Group by operation name
            ops_by_name = defaultdict(list)
            for op in operations:
                ops_by_name[op.operation_name].append(op)
            
            performance = {}
            for op_name, ops in ops_by_name.items():
                durations = [op.duration for op in ops if op.duration]
                success_count = sum(1 for op in ops if op.success)
                total_items = sum(op.items_processed for op in ops)
                total_errors = sum(op.errors_count for op in ops)
                
                performance[op_name] = {
                    'total_executions': len(ops),
                    'success_rate': success_count / len(ops) if ops else 0,
                    'average_duration': sum(durations) / len(durations) if durations else 0,
                    'total_items_processed': total_items,
                    'error_rate': total_errors / total_items if total_items > 0 else 0,
                    'throughput': total_items / sum(durations) if durations and sum(durations) > 0 else 0
                }
            
            return performance
            
        except Exception as e:
            logger.warning(f"Failed to calculate operation performance: {e}")
            return {}
    
    def _calculate_component_performance(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate component-specific performance metrics"""
        
        try:
            # Group by component
            metrics_by_component = defaultdict(lambda: defaultdict(list))
            
            for metric in metrics:
                component = metric.tags.get('component', 'unknown')
                metrics_by_component[component][metric.metric_name].append(metric.value)
            
            performance = {}
            for component, component_metrics in metrics_by_component.items():
                performance[component] = {}
                
                for metric_name, values in component_metrics.items():
                    if values:
                        performance[component][metric_name] = {
                            'min': min(values),
                            'max': max(values),
                            'average': sum(values) / len(values),
                            'count': len(values)
                        }
            
            return performance
            
        except Exception as e:
            logger.warning(f"Failed to calculate component performance: {e}")
            return {}
    
    def _calculate_performance_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        
        try:
            if len(metrics) < 2:
                return {}
            
            # Sort metrics by timestamp
            sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
            
            # Group by metric name
            metrics_by_name = defaultdict(list)
            for metric in sorted_metrics:
                metrics_by_name[metric.metric_name].append((metric.timestamp, metric.value))
            
            trends = {}
            for metric_name, time_values in metrics_by_name.items():
                if len(time_values) >= 2:
                    # Calculate simple trend (first vs last)
                    first_value = time_values[0][1]
                    last_value = time_values[-1][1]
                    
                    if first_value != 0:
                        trend_percent = ((last_value - first_value) / first_value) * 100
                    else:
                        trend_percent = 0
                    
                    trends[metric_name] = {
                        'trend_percent': trend_percent,
                        'direction': 'increasing' if trend_percent > 5 else 'decreasing' if trend_percent < -5 else 'stable',
                        'first_value': first_value,
                        'last_value': last_value,
                        'data_points': len(time_values)
                    }
            
            return trends
            
        except Exception as e:
            logger.warning(f"Failed to calculate performance trends: {e}")
            return {}
    
    def _get_threshold_violations(self, operations: List[OperationMetrics]) -> List[Dict[str, Any]]:
        """Get list of performance threshold violations"""
        
        violations = []
        
        try:
            for operation in operations:
                operation_violations = []
                
                # Check duration threshold
                if (operation.duration and 
                    operation.duration > self.performance_thresholds['operation_duration_seconds']):
                    operation_violations.append({
                        'type': 'duration',
                        'threshold': self.performance_thresholds['operation_duration_seconds'],
                        'actual': operation.duration,
                        'severity': 'warning'
                    })
                
                # Check error rate threshold
                if operation.items_processed > 0:
                    error_rate = (operation.errors_count / operation.items_processed) * 100
                    if error_rate > self.performance_thresholds['error_rate_percent']:
                        operation_violations.append({
                            'type': 'error_rate',
                            'threshold': self.performance_thresholds['error_rate_percent'],
                            'actual': error_rate,
                            'severity': 'critical'
                        })
                
                if operation_violations:
                    violations.append({
                        'operation_name': operation.operation_name,
                        'timestamp': operation.start_time.isoformat(),
                        'violations': operation_violations
                    })
            
        except Exception as e:
            logger.warning(f"Failed to get threshold violations: {e}")
        
        return violations
    
    def get_metrics_by_component(self, component_name: str, hours: int = 24) -> List[PerformanceMetric]:
        """Get metrics for a specific component"""
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            component_metrics = [
                metric for metric in self.performance_metrics
                if (metric.tags.get('component') == component_name and 
                    metric.timestamp >= cutoff_time)
            ]
            
            return component_metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics for component {component_name}: {e}")
            return []
    
    def get_top_performing_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing operations by various metrics"""
        
        try:
            if not self.completed_operations:
                return []
            
            # Calculate performance scores
            operation_scores = []
            
            for operation in self.completed_operations:
                if not operation.duration:
                    continue
                
                # Calculate performance score (higher is better)
                throughput = operation.items_processed / operation.duration if operation.duration > 0 else 0
                success_rate = 1.0 if operation.success else 0.0
                error_rate = operation.errors_count / max(operation.items_processed, 1)
                
                # Composite score
                score = (throughput * success_rate) / max(error_rate + 1, 1)
                
                operation_scores.append({
                    'operation_name': operation.operation_name,
                    'score': score,
                    'duration': operation.duration,
                    'throughput': throughput,
                    'success_rate': success_rate,
                    'error_rate': error_rate,
                    'items_processed': operation.items_processed,
                    'timestamp': operation.start_time.isoformat()
                })
            
            # Sort by score and return top performers
            operation_scores.sort(key=lambda x: x['score'], reverse=True)
            return operation_scores[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get top performing operations: {e}")
            return []
    
    def cleanup_old_metrics(self, days_to_keep: int = 7) -> int:
        """Clean up old metrics data"""
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean performance metrics
            original_count = len(self.performance_metrics)
            self.performance_metrics = deque(
                (metric for metric in self.performance_metrics if metric.timestamp >= cutoff_time),
                maxlen=self.performance_metrics.maxlen
            )
            
            # Clean completed operations
            original_ops_count = len(self.completed_operations)
            self.completed_operations = deque(
                (op for op in self.completed_operations if op.start_time >= cutoff_time),
                maxlen=self.completed_operations.maxlen
            )
            
            metrics_removed = original_count - len(self.performance_metrics)
            operations_removed = original_ops_count - len(self.completed_operations)
            
            logger.info(f"Cleaned up {metrics_removed} old metrics and {operations_removed} old operations")
            return metrics_removed + operations_removed
            
        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")
            return 0
    
    def get_metrics_statistics(self) -> Dict[str, Any]:
        """Get comprehensive metrics system statistics"""
        
        try:
            return {
                'collection_status': {
                    'monitoring_active': self.monitoring_active,
                    'system_metrics_enabled': self.collect_system_metrics,
                    'collection_interval_seconds': self.metric_collection_interval,
                    'detailed_profiling_enabled': self.enable_detailed_profiling
                },
                'data_statistics': {
                    'total_metrics_collected': len(self.performance_metrics),
                    'active_operations': len(self.operation_metrics),
                    'completed_operations': len(self.completed_operations),
                    'unique_components': len(set(m.tags.get('component', 'unknown') for m in self.performance_metrics)),
                    'unique_metric_names': len(set(m.metric_name for m in self.performance_metrics))
                },
                'performance_thresholds': self.performance_thresholds,
                'recent_activity': {
                    'metrics_last_hour': len([
                        m for m in self.performance_metrics
                        if (datetime.now() - m.timestamp).total_seconds() < 3600
                    ]),
                    'operations_last_hour': len([
                        op for op in self.completed_operations
                        if (datetime.now() - op.start_time).total_seconds() < 3600
                    ])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics statistics: {e}")
            return {'error': str(e)}
    
    def __del__(self):
        """Destructor to cleanup monitoring thread"""
        try:
            self.stop_monitoring()
        except:
            pass  # Ignore errors during cleanup