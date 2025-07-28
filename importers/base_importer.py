"""Base importer interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from data_models import ImportResult
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseImporter(ABC):
    """Abstract base class for all importers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize importer with configuration"""
        self.config = config
        self.errors = []
        self.warnings = []
        
    @abstractmethod
    def import_data(self, **kwargs) -> ImportResult:
        """Import data from source"""
        pass
    
    def validate_config(self) -> bool:
        """Validate importer configuration"""
        return True
    
    def get_import_stats(self) -> Dict[str, Any]:
        """Get statistics about the import process"""
        return {
            'errors': len(self.errors),
            'warnings': len(self.warnings),
            'source': self.__class__.__name__
        }
    
    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)
        logger.error(f"{self.__class__.__name__}: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message"""
        self.warnings.append(warning)
        logger.warning(f"{self.__class__.__name__}: {warning}")
    
    def clear_messages(self) -> None:
        """Clear error and warning messages"""
        self.errors.clear()
        self.warnings.clear()