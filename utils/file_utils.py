"""File operation utilities"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import hashlib
from datetime import datetime

from .logging_utils import get_logger

logger = get_logger(__name__)


class FileUtils:
    """Utilities for file operations"""
    
    @staticmethod
    def ensure_directory(directory_path: str) -> bool:
        """Ensure directory exists, create if it doesn't"""
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            return False
    
    @staticmethod
    def get_file_hash(file_path: str) -> Optional[str]:
        """Get SHA256 hash of file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return None
    
    @staticmethod
    def backup_file(source_path: str, backup_dir: str, prefix: str = "backup") -> Optional[str]:
        """Create a backup of a file with timestamp"""
        try:
            source_file = Path(source_path)
            if not source_file.exists():
                logger.error(f"Source file does not exist: {source_path}")
                return None
            
            # Ensure backup directory exists
            FileUtils.ensure_directory(backup_dir)
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{prefix}_{timestamp}_{source_file.name}"
            backup_path = Path(backup_dir) / backup_filename
            
            # Copy file
            shutil.copy2(source_path, backup_path)
            
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to backup file {source_path}: {e}")
            return None
    
    @staticmethod
    def cleanup_old_backups(backup_dir: str, pattern: str = "backup_*", keep_count: int = 5) -> int:
        """Clean up old backup files, keeping only the most recent ones"""
        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                return 0
            
            # Find all backup files matching pattern
            backup_files = list(backup_path.glob(pattern))
            
            if len(backup_files) <= keep_count:
                return 0
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old backups
            removed_count = 0
            for old_backup in backup_files[keep_count:]:
                try:
                    old_backup.unlink()
                    removed_count += 1
                    logger.info(f"Removed old backup: {old_backup}")
                except Exception as e:
                    logger.warning(f"Failed to remove old backup {old_backup}: {e}")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups in {backup_dir}: {e}")
            return 0
    
    @staticmethod
    def get_file_size(file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return Path(file_path).stat().st_size
        except Exception as e:
            logger.error(f"Failed to get file size for {file_path}: {e}")
            return 0
    
    @staticmethod
    def is_file_readable(file_path: str) -> bool:
        """Check if file is readable"""
        try:
            with open(file_path, 'r') as f:
                f.read(1)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_available_disk_space(directory: str) -> int:
        """Get available disk space in bytes"""
        try:
            statvfs = os.statvfs(directory)
            return statvfs.f_frsize * statvfs.f_bavail
        except AttributeError:
            # Windows
            import shutil
            return shutil.disk_usage(directory).free
        except Exception as e:
            logger.error(f"Failed to get disk space for {directory}: {e}")
            return 0
    
    @staticmethod
    def safe_filename(filename: str) -> str:
        """Make filename safe for filesystem"""
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        safe_name = filename
        
        for char in unsafe_chars:
            safe_name = safe_name.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        safe_name = safe_name.strip('. ')
        
        # Limit length
        if len(safe_name) > 255:
            safe_name = safe_name[:255]
        
        return safe_name or "unnamed_file"
    
    @staticmethod
    def find_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
        """Find files matching pattern"""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return []
            
            if recursive:
                files = dir_path.rglob(pattern)
            else:
                files = dir_path.glob(pattern)
            
            return [str(f) for f in files if f.is_file()]
            
        except Exception as e:
            logger.error(f"Failed to find files in {directory}: {e}")
            return []