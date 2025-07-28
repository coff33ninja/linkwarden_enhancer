"""Logging utilities for Linkwarden Enhancer"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logging(level: str = 'INFO', 
                 log_file: Optional[str] = None,
                 max_file_size_mb: int = 10,
                 backup_count: int = 5) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        max_file_size_mb: Maximum log file size in MB
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels for noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    
    return root_logger


def setup_verbose_logging(enable_debug: bool = True, 
                         component_filters: Optional[Dict[str, str]] = None,
                         log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up verbose logging for debugging all system components
    
    Args:
        enable_debug: Enable DEBUG level logging
        component_filters: Component-specific log levels
        log_file: Optional debug log file path
        
    Returns:
        Configured logger for verbose output
    """
    
    # Default component filters for verbose mode
    if component_filters is None:
        component_filters = {
            'linkwarden_enhancer.core': 'DEBUG',
            'linkwarden_enhancer.ai': 'DEBUG',
            'linkwarden_enhancer.intelligence': 'DEBUG',
            'linkwarden_enhancer.enhancement': 'DEBUG',
            'linkwarden_enhancer.importers': 'DEBUG',
            'linkwarden_enhancer.reporting': 'DEBUG',
            'linkwarden_enhancer.cli': 'DEBUG',
            'urllib3': 'WARNING',
            'requests': 'WARNING',
            'selenium': 'WARNING',
            'transformers': 'INFO'
        }
    
    # Create verbose formatter with more detail
    verbose_formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d [%(levelname)8s] %(name)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger for verbose mode
    root_logger = logging.getLogger()
    
    if enable_debug:
        root_logger.setLevel(logging.DEBUG)
    
    # Update console handler with verbose formatter
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setFormatter(verbose_formatter)
            if enable_debug:
                handler.setLevel(logging.DEBUG)
    
    # Add debug file handler if specified
    if log_file:
        debug_log_path = Path(log_file)
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        debug_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB for debug logs
            backupCount=3,
            encoding='utf-8'
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(verbose_formatter)
        root_logger.addHandler(debug_handler)
    
    # Apply component-specific filters
    for component, level in component_filters.items():
        component_logger = logging.getLogger(component)
        component_logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    
    # Log verbose mode activation
    logger = logging.getLogger(__name__)
    logger.info("Verbose logging enabled with detailed component debugging")
    
    return root_logger


class ComponentLogger:
    """Enhanced logger for specific components with structured logging"""
    
    def __init__(self, component_name: str, verbose: bool = False):
        """Initialize component logger"""
        self.component_name = component_name
        self.verbose = verbose
        self.logger = logging.getLogger(component_name)
        
        # Component-specific context
        self.context = {
            'component': component_name,
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
    
    def debug_operation(self, operation: str, details: Dict[str, Any] = None) -> None:
        """Log detailed operation information"""
        
        if details is None:
            details = {}
        
        log_data = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'details': details,
            **self.context
        }
        
        if self.verbose:
            self.logger.debug(f"OPERATION: {operation}")
            for key, value in details.items():
                self.logger.debug(f"  {key}: {value}")
        else:
            self.logger.debug(f"{operation}: {json.dumps(log_data, default=str)}")
    
    def debug_data_flow(self, stage: str, data_summary: Dict[str, Any]) -> None:
        """Log data flow information"""
        
        log_data = {
            'stage': stage,
            'data_summary': data_summary,
            'timestamp': datetime.now().isoformat(),
            **self.context
        }
        
        if self.verbose:
            self.logger.debug(f"DATA FLOW - {stage}:")
            for key, value in data_summary.items():
                self.logger.debug(f"  {key}: {value}")
        else:
            self.logger.debug(f"Data flow [{stage}]: {json.dumps(log_data, default=str)}")
    
    def debug_performance(self, operation: str, duration: float, 
                         metrics: Dict[str, Any] = None) -> None:
        """Log performance metrics"""
        
        if metrics is None:
            metrics = {}
        
        perf_data = {
            'operation': operation,
            'duration_seconds': duration,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            **self.context
        }
        
        if self.verbose:
            self.logger.debug(f"PERFORMANCE - {operation}: {duration:.3f}s")
            for key, value in metrics.items():
                self.logger.debug(f"  {key}: {value}")
        else:
            self.logger.debug(f"Performance [{operation}]: {json.dumps(perf_data, default=str)}")
    
    def debug_learning(self, learning_event: str, learning_data: Dict[str, Any]) -> None:
        """Log learning and intelligence events"""
        
        learn_data = {
            'learning_event': learning_event,
            'learning_data': learning_data,
            'timestamp': datetime.now().isoformat(),
            **self.context
        }
        
        if self.verbose:
            self.logger.debug(f"LEARNING - {learning_event}:")
            for key, value in learning_data.items():
                self.logger.debug(f"  {key}: {value}")
        else:
            self.logger.debug(f"Learning [{learning_event}]: {json.dumps(learn_data, default=str)}")
    
    def info(self, message: str, extra_data: Dict[str, Any] = None) -> None:
        """Log info message with optional extra data"""
        if extra_data:
            self.logger.info(f"{message} | {json.dumps(extra_data, default=str)}")
        else:
            self.logger.info(message)
    
    def warning(self, message: str, extra_data: Dict[str, Any] = None) -> None:
        """Log warning message with optional extra data"""
        if extra_data:
            self.logger.warning(f"{message} | {json.dumps(extra_data, default=str)}")
        else:
            self.logger.warning(message)
    
    def error(self, message: str, extra_data: Dict[str, Any] = None, exc_info: bool = False) -> None:
        """Log error message with optional extra data"""
        if extra_data:
            self.logger.error(f"{message} | {json.dumps(extra_data, default=str)}", exc_info=exc_info)
        else:
            self.logger.error(message, exc_info=exc_info)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)


def get_component_logger(component_name: str, verbose: bool = False) -> ComponentLogger:
    """Get an enhanced component logger"""
    return ComponentLogger(component_name, verbose)