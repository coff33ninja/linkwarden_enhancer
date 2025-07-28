"""Backup System - Multi-tier backup with retention management"""

import json
import gzip
import shutil
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..utils.logging_utils import get_logger
from ..utils.file_utils import FileUtils

logger = get_logger(__name__)


@dataclass
class BackupInfo:
    """Information about a backup file"""
    path: str
    timestamp: datetime
    operation_name: str
    file_size: int
    checksum: str
    compressed: bool
    metadata: Dict[str, Any]


class BackupSystem:
    """Multi-tier backup system with retention management"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize backup system with configuration"""
        self.config = config
        self.backup_config = config.get('backup', {})
        
        # Backup directories
        self.backup_dir = Path(config.get('directories', {}).get('backup_dir', 'backups'))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup settings
        self.max_backups = self.backup_config.get('max_backups', 10)
        self.compress_backups = self.backup_config.get('compress_backups', True)
        self.retention_days = self.backup_config.get('retention_days', 30)
        self.incremental_enabled = self.backup_config.get('incremental_enabled', True)
        
        # Backup registry
        self.backup_registry = self._load_backup_registry()
        
        logger.info(f"Backup system initialized: {self.backup_dir}, max_backups={self.max_backups}")
    
    def create_backup(self, 
                     source_file: str, 
                     operation_name: str = "manual",
                     metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a timestamped backup of the source file"""
        
        try:
            source_path = Path(source_file)
            if not source_path.exists():
                logger.error(f"Source file does not exist: {source_file}")
                return None
            
            # Generate backup filename
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.stem}_{operation_name}_{timestamp_str}"
            
            if self.compress_backups:
                backup_path = self.backup_dir / f"{backup_name}.json.gz"
            else:
                backup_path = self.backup_dir / f"{backup_name}.json"
            
            # Create backup
            if self.compress_backups:
                self._create_compressed_backup(source_path, backup_path)
            else:
                shutil.copy2(source_path, backup_path)
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_path)
            
            # Create backup info
            backup_info = BackupInfo(
                path=str(backup_path),
                timestamp=timestamp,
                operation_name=operation_name,
                file_size=backup_path.stat().st_size,
                checksum=checksum,
                compressed=self.compress_backups,
                metadata=metadata or {}
            )
            
            # Register backup
            self._register_backup(backup_info)
            
            # Apply retention policy
            self._apply_retention_policy()
            
            logger.info(f"Backup created: {backup_path} ({backup_info.file_size} bytes)")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def create_incremental_backup(self, 
                                 source_file: str, 
                                 operation_name: str = "incremental",
                                 metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create incremental backup if changes detected"""
        
        try:
            if not self.incremental_enabled:
                return self.create_backup(source_file, operation_name, metadata)
            
            source_path = Path(source_file)
            current_checksum = self._calculate_checksum(source_path)
            
            # Check if we need to create backup
            latest_backup = self.get_latest_backup(operation_name)
            if latest_backup and latest_backup.checksum == current_checksum:
                logger.info(f"No changes detected, skipping incremental backup for {source_file}")
                return latest_backup.path
            
            # Create backup with incremental metadata
            if metadata is None:
                metadata = {}
            metadata['incremental'] = True
            metadata['previous_checksum'] = latest_backup.checksum if latest_backup else None
            
            return self.create_backup(source_file, operation_name, metadata)
            
        except Exception as e:
            logger.error(f"Failed to create incremental backup: {e}")
            return None
    
    def restore_backup(self, backup_path: str, target_file: str) -> bool:
        """Restore from backup file"""
        
        try:
            backup_path_obj = Path(backup_path)
            target_path = Path(target_file)
            
            if not backup_path_obj.exists():
                logger.error(f"Backup file does not exist: {backup_path}")
                return False
            
            # Create target directory if needed
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Restore based on compression
            if backup_path.endswith('.gz'):
                self._restore_compressed_backup(backup_path_obj, target_path)
            else:
                shutil.copy2(backup_path_obj, target_path)
            
            # Verify restoration
            if self.verify_backup_integrity(backup_path):
                logger.info(f"Backup restored successfully: {backup_path} -> {target_file}")
                return True
            else:
                logger.error(f"Backup restoration verification failed: {backup_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def verify_backup_integrity(self, backup_path: str) -> bool:
        """Verify backup file integrity using checksums"""
        
        try:
            backup_info = self.get_backup_info(backup_path)
            if not backup_info:
                logger.warning(f"No backup info found for {backup_path}")
                return True  # Assume valid if no info available
            
            current_checksum = self._calculate_checksum(Path(backup_path))
            
            if current_checksum == backup_info.checksum:
                logger.debug(f"Backup integrity verified: {backup_path}")
                return True
            else:
                logger.error(f"Backup integrity check failed: {backup_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify backup integrity: {e}")
            return False
    
    def list_backups(self, operation_name: Optional[str] = None) -> List[BackupInfo]:
        """List all backups, optionally filtered by operation name"""
        
        try:
            backups = list(self.backup_registry.values())
            
            if operation_name:
                backups = [b for b in backups if b.operation_name == operation_name]
            
            # Sort by timestamp (newest first)
            backups.sort(key=lambda b: b.timestamp, reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []
    
    def get_latest_backup(self, operation_name: Optional[str] = None) -> Optional[BackupInfo]:
        """Get the most recent backup"""
        
        backups = self.list_backups(operation_name)
        return backups[0] if backups else None
    
    def get_backup_info(self, backup_path: str) -> Optional[BackupInfo]:
        """Get information about a specific backup"""
        
        return self.backup_registry.get(backup_path)
    
    def delete_backup(self, backup_path: str) -> bool:
        """Delete a specific backup"""
        
        try:
            backup_path_obj = Path(backup_path)
            
            if backup_path_obj.exists():
                backup_path_obj.unlink()
            
            # Remove from registry
            if backup_path in self.backup_registry:
                del self.backup_registry[backup_path]
                self._save_backup_registry()
            
            logger.info(f"Backup deleted: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup: {e}")
            return False
    
    def cleanup_old_backups(self) -> int:
        """Clean up old backups based on retention policy"""
        
        try:
            deleted_count = 0
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            for backup_path, backup_info in list(self.backup_registry.items()):
                if backup_info.timestamp < cutoff_date:
                    if self.delete_backup(backup_path):
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
            return 0
    
    def get_backup_statistics(self) -> Dict[str, Any]:
        """Get backup system statistics"""
        
        try:
            backups = list(self.backup_registry.values())
            
            if not backups:
                return {
                    'total_backups': 0,
                    'total_size': 0,
                    'oldest_backup': None,
                    'newest_backup': None,
                    'operations': {}
                }
            
            total_size = sum(b.file_size for b in backups)
            operations = {}
            
            for backup in backups:
                op_name = backup.operation_name
                if op_name not in operations:
                    operations[op_name] = {'count': 0, 'total_size': 0}
                operations[op_name]['count'] += 1
                operations[op_name]['total_size'] += backup.file_size
            
            oldest = min(backups, key=lambda b: b.timestamp)
            newest = max(backups, key=lambda b: b.timestamp)
            
            return {
                'total_backups': len(backups),
                'total_size': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'oldest_backup': oldest.timestamp.isoformat(),
                'newest_backup': newest.timestamp.isoformat(),
                'operations': operations,
                'compression_enabled': self.compress_backups,
                'retention_days': self.retention_days,
                'max_backups': self.max_backups
            }
            
        except Exception as e:
            logger.error(f"Failed to get backup statistics: {e}")
            return {'error': str(e)}
    
    def _create_compressed_backup(self, source_path: Path, backup_path: Path) -> None:
        """Create compressed backup using gzip"""
        
        with open(source_path, 'rb') as f_in:
            with gzip.open(backup_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _restore_compressed_backup(self, backup_path: Path, target_path: Path) -> None:
        """Restore from compressed backup"""
        
        with gzip.open(backup_path, 'rb') as f_in:
            with open(target_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        
        sha256_hash = hashlib.sha256()
        
        if str(file_path).endswith('.gz'):
            # For compressed files, calculate checksum of compressed content
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        else:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def _register_backup(self, backup_info: BackupInfo) -> None:
        """Register backup in the registry"""
        
        self.backup_registry[backup_info.path] = backup_info
        self._save_backup_registry()
    
    def _apply_retention_policy(self) -> None:
        """Apply backup retention policy"""
        
        try:
            # Clean up by count
            backups = sorted(self.backup_registry.values(), key=lambda b: b.timestamp, reverse=True)
            
            if len(backups) > self.max_backups:
                excess_backups = backups[self.max_backups:]
                for backup in excess_backups:
                    self.delete_backup(backup.path)
            
            # Clean up by age
            self.cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"Failed to apply retention policy: {e}")
    
    def _load_backup_registry(self) -> Dict[str, BackupInfo]:
        """Load backup registry from file"""
        
        registry_file = self.backup_dir / 'backup_registry.json'
        
        try:
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                registry = {}
                for path, info_data in data.items():
                    # Convert timestamp string back to datetime
                    timestamp = datetime.fromisoformat(info_data['timestamp'])
                    
                    backup_info = BackupInfo(
                        path=info_data['path'],
                        timestamp=timestamp,
                        operation_name=info_data['operation_name'],
                        file_size=info_data['file_size'],
                        checksum=info_data['checksum'],
                        compressed=info_data.get('compressed', False),
                        metadata=info_data.get('metadata', {})
                    )
                    registry[path] = backup_info
                
                logger.debug(f"Loaded {len(registry)} backup entries from registry")
                return registry
            
        except Exception as e:
            logger.warning(f"Failed to load backup registry: {e}")
        
        return {}
    
    def _save_backup_registry(self) -> None:
        """Save backup registry to file"""
        
        registry_file = self.backup_dir / 'backup_registry.json'
        
        try:
            # Convert BackupInfo objects to serializable format
            data = {}
            for path, backup_info in self.backup_registry.items():
                data[path] = {
                    'path': backup_info.path,
                    'timestamp': backup_info.timestamp.isoformat(),
                    'operation_name': backup_info.operation_name,
                    'file_size': backup_info.file_size,
                    'checksum': backup_info.checksum,
                    'compressed': backup_info.compressed,
                    'metadata': backup_info.metadata
                }
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(data)} backup entries to registry")
            
        except Exception as e:
            logger.error(f"Failed to save backup registry: {e}")