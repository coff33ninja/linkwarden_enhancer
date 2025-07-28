"""Intelligence export/import system for backup and sharing capabilities"""

import json
import gzip
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from .dictionary_manager import SmartDictionaryManager
from .continuous_learner import ContinuousLearner
from .adaptive_intelligence import AdaptiveIntelligence
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class IntelligenceSnapshot:
    """Represents a snapshot of intelligence data"""
    snapshot_id: str
    created_at: datetime
    version: str
    description: str
    data_types: List[str]
    file_size: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntelligenceExport:
    """Complete intelligence export package"""
    export_id: str
    created_at: datetime
    version: str
    description: str
    dictionary_data: Optional[Dict[str, Any]] = None
    learning_data: Optional[Dict[str, Any]] = None
    adaptation_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntelligenceManager:
    """Manager for intelligence export/import operations"""
    
    def __init__(self, config: Dict[str, Any], data_dir: str = "data"):
        """Initialize intelligence manager"""
        self.config = config
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Export/import settings
        self.compress_exports = config.get('intelligence', {}).get('compress_exports', True)
        self.max_export_history = config.get('intelligence', {}).get('max_export_history', 50)
        self.auto_backup_enabled = config.get('intelligence', {}).get('auto_backup_enabled', True)
        self.backup_interval_hours = config.get('intelligence', {}).get('backup_interval_hours', 24)
        
        # Export history
        self.export_history = []
        self.import_history = []
        
        # Load existing history
        self._load_export_history()
        
        logger.info("Intelligence manager initialized")
    
    def create_full_export(self, description: str = "", include_user_data: bool = True) -> Optional[str]:
        """Create a complete export of all intelligence data"""
        
        try:
            export_id = f"full_export_{int(datetime.now().timestamp())}"
            
            logger.info(f"Creating full intelligence export: {export_id}")
            
            # Initialize components
            dictionary_manager = SmartDictionaryManager(self.config)
            continuous_learner = ContinuousLearner(str(self.data_dir))
            adaptive_intelligence = AdaptiveIntelligence(str(self.data_dir)) if include_user_data else None
            
            # Export dictionary data
            dictionary_data = dictionary_manager.export_intelligence_data()
            
            # Export learning data
            learning_data = continuous_learner.export_learned_patterns()
            
            # Export adaptation data
            adaptation_data = adaptive_intelligence.export_user_data() if adaptive_intelligence else None
            
            # Create export package
            export_package = IntelligenceExport(
                export_id=export_id,
                created_at=datetime.now(),
                version="1.0",
                description=description or f"Full intelligence export created at {datetime.now().isoformat()}",
                dictionary_data=dictionary_data,
                learning_data=learning_data,
                adaptation_data=adaptation_data,
                metadata={
                    'export_type': 'full',
                    'include_user_data': include_user_data,
                    'components_exported': ['dictionary', 'learning'] + (['adaptation'] if include_user_data else [])
                }
            )
            
            # Save export
            export_file = self._save_export(export_package)
            
            if export_file:
                # Add to history
                snapshot = IntelligenceSnapshot(
                    snapshot_id=export_id,
                    created_at=export_package.created_at,
                    version=export_package.version,
                    description=export_package.description,
                    data_types=export_package.metadata['components_exported'],
                    file_size=Path(export_file).stat().st_size,
                    checksum=self._calculate_file_checksum(export_file),
                    metadata=export_package.metadata
                )
                
                self.export_history.append(snapshot)
                self._save_export_history()
                
                logger.info(f"Full export completed: {export_file}")
                return export_file
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create full export: {e}")
            return None
    
    def create_selective_export(self, components: List[str], description: str = "") -> Optional[str]:
        """Create export of selected intelligence components"""
        
        try:
            export_id = f"selective_export_{int(datetime.now().timestamp())}"
            
            logger.info(f"Creating selective intelligence export: {export_id}, components: {components}")
            
            export_package = IntelligenceExport(
                export_id=export_id,
                created_at=datetime.now(),
                version="1.0",
                description=description or f"Selective export of {', '.join(components)}",
                metadata={
                    'export_type': 'selective',
                    'components_exported': components
                }
            )
            
            # Export selected components
            if 'dictionary' in components:
                dictionary_manager = SmartDictionaryManager(self.config)
                export_package.dictionary_data = dictionary_manager.export_intelligence_data()
            
            if 'learning' in components:
                continuous_learner = ContinuousLearner(str(self.data_dir))
                export_package.learning_data = continuous_learner.export_learned_patterns()
            
            if 'adaptation' in components:
                adaptive_intelligence = AdaptiveIntelligence(str(self.data_dir))
                export_package.adaptation_data = adaptive_intelligence.export_user_data()
            
            # Save export
            export_file = self._save_export(export_package)
            
            if export_file:
                # Add to history
                snapshot = IntelligenceSnapshot(
                    snapshot_id=export_id,
                    created_at=export_package.created_at,
                    version=export_package.version,
                    description=export_package.description,
                    data_types=components,
                    file_size=Path(export_file).stat().st_size,
                    checksum=self._calculate_file_checksum(export_file),
                    metadata=export_package.metadata
                )
                
                self.export_history.append(snapshot)
                self._save_export_history()
                
                logger.info(f"Selective export completed: {export_file}")
                return export_file
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create selective export: {e}")
            return None
    
    def import_intelligence_data(self, import_file: str, components: Optional[List[str]] = None,
                               merge_strategy: str = "merge") -> Dict[str, Any]:
        """Import intelligence data from export file"""
        
        try:
            import_id = f"import_{int(datetime.now().timestamp())}"
            
            logger.info(f"Starting intelligence import: {import_id} from {import_file}")
            
            # Load export package
            export_package = self._load_export(import_file)
            if not export_package:
                return {'success': False, 'error': 'Failed to load export file'}
            
            import_results = {
                'import_id': import_id,
                'success': True,
                'components_imported': [],
                'components_failed': [],
                'merge_strategy': merge_strategy,
                'warnings': [],
                'errors': []
            }
            
            # Import dictionary data
            if (not components or 'dictionary' in components) and export_package.dictionary_data:
                try:
                    dictionary_manager = SmartDictionaryManager(self.config)
                    success = dictionary_manager.import_intelligence_data(export_package.dictionary_data)
                    
                    if success:
                        import_results['components_imported'].append('dictionary')
                        logger.info("Dictionary data imported successfully")
                    else:
                        import_results['components_failed'].append('dictionary')
                        import_results['errors'].append("Failed to import dictionary data")
                        
                except Exception as e:
                    import_results['components_failed'].append('dictionary')
                    import_results['errors'].append(f"Dictionary import error: {e}")
            
            # Import learning data
            if (not components or 'learning' in components) and export_package.learning_data:
                try:
                    continuous_learner = ContinuousLearner(str(self.data_dir))
                    success = continuous_learner.import_learned_patterns(export_package.learning_data)
                    
                    if success:
                        import_results['components_imported'].append('learning')
                        logger.info("Learning data imported successfully")
                    else:
                        import_results['components_failed'].append('learning')
                        import_results['errors'].append("Failed to import learning data")
                        
                except Exception as e:
                    import_results['components_failed'].append('learning')
                    import_results['errors'].append(f"Learning import error: {e}")
            
            # Import adaptation data
            if (not components or 'adaptation' in components) and export_package.adaptation_data:
                try:
                    adaptive_intelligence = AdaptiveIntelligence(str(self.data_dir))
                    success = adaptive_intelligence.import_user_data(export_package.adaptation_data)
                    
                    if success:
                        import_results['components_imported'].append('adaptation')
                        logger.info("Adaptation data imported successfully")
                    else:
                        import_results['components_failed'].append('adaptation')
                        import_results['errors'].append("Failed to import adaptation data")
                        
                except Exception as e:
                    import_results['components_failed'].append('adaptation')
                    import_results['errors'].append(f"Adaptation import error: {e}")
            
            # Update success status
            import_results['success'] = len(import_results['components_failed']) == 0
            
            # Add to import history
            import_record = {
                'import_id': import_id,
                'timestamp': datetime.now().isoformat(),
                'source_file': import_file,
                'export_id': export_package.export_id,
                'components_imported': import_results['components_imported'],
                'success': import_results['success']
            }
            
            self.import_history.append(import_record)
            self._save_import_history()
            
            logger.info(f"Intelligence import completed: {import_results}")
            return import_results
            
        except Exception as e:
            logger.error(f"Failed to import intelligence data: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_intelligence_backup(self) -> Optional[str]:
        """Create automatic backup of intelligence data"""
        
        try:
            # Check if backup is needed
            if not self.auto_backup_enabled:
                return None
            
            # Check last backup time
            if self.export_history:
                last_backup = max(
                    (snapshot for snapshot in self.export_history 
                     if snapshot.metadata.get('export_type') == 'backup'),
                    key=lambda s: s.created_at,
                    default=None
                )
                
                if last_backup:
                    hours_since_backup = (datetime.now() - last_backup.created_at).total_seconds() / 3600
                    if hours_since_backup < self.backup_interval_hours:
                        logger.debug(f"Backup not needed, last backup was {hours_since_backup:.1f} hours ago")
                        return None
            
            # Create backup
            backup_description = f"Automatic backup created at {datetime.now().isoformat()}"
            backup_file = self.create_full_export(backup_description, include_user_data=True)
            
            if backup_file:
                # Mark as backup in metadata
                if self.export_history:
                    self.export_history[-1].metadata['export_type'] = 'backup'
                    self._save_export_history()
                
                logger.info(f"Intelligence backup created: {backup_file}")
                return backup_file
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create intelligence backup: {e}")
            return None
    
    def restore_from_backup(self, backup_file: Optional[str] = None) -> Dict[str, Any]:
        """Restore intelligence data from backup"""
        
        try:
            # Find backup file if not specified
            if not backup_file:
                backup_snapshots = [
                    snapshot for snapshot in self.export_history
                    if snapshot.metadata.get('export_type') == 'backup'
                ]
                
                if not backup_snapshots:
                    return {'success': False, 'error': 'No backup files found'}
                
                # Use most recent backup
                latest_backup = max(backup_snapshots, key=lambda s: s.created_at)
                backup_file = str(self.data_dir / f"{latest_backup.snapshot_id}.json")
                
                if self.compress_exports:
                    backup_file += ".gz"
            
            # Verify backup file exists
            if not Path(backup_file).exists():
                return {'success': False, 'error': f'Backup file not found: {backup_file}'}
            
            # Import from backup
            restore_result = self.import_intelligence_data(backup_file)
            restore_result['restore_type'] = 'backup'
            restore_result['backup_file'] = backup_file
            
            logger.info(f"Intelligence restore completed from backup: {backup_file}")
            return restore_result
            
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return {'success': False, 'error': str(e)}
    
    def migrate_intelligence_data(self, source_version: str, target_version: str) -> Dict[str, Any]:
        """Migrate intelligence data between versions"""
        
        try:
            migration_id = f"migration_{int(datetime.now().timestamp())}"
            
            logger.info(f"Starting intelligence migration: {source_version} -> {target_version}")
            
            migration_results = {
                'migration_id': migration_id,
                'source_version': source_version,
                'target_version': target_version,
                'success': True,
                'components_migrated': [],
                'warnings': [],
                'errors': []
            }
            
            # Version-specific migration logic
            if source_version == "1.0" and target_version == "1.1":
                # Example migration from v1.0 to v1.1
                migration_results = self._migrate_v1_0_to_v1_1(migration_results)
            else:
                migration_results['success'] = False
                migration_results['errors'].append(f"Unsupported migration path: {source_version} -> {target_version}")
            
            logger.info(f"Intelligence migration completed: {migration_results}")
            return migration_results
            
        except Exception as e:
            logger.error(f"Failed to migrate intelligence data: {e}")
            return {'success': False, 'error': str(e)}
    
    def _migrate_v1_0_to_v1_1(self, migration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.0 to 1.1"""
        
        try:
            # Example migration logic - add new fields, convert formats, etc.
            
            # Migrate dictionary data
            try:
                dictionary_manager = SmartDictionaryManager(self.config)
                # Add migration logic here
                migration_results['components_migrated'].append('dictionary')
            except Exception as e:
                migration_results['errors'].append(f"Dictionary migration failed: {e}")
            
            # Migrate learning data
            try:
                continuous_learner = ContinuousLearner(str(self.data_dir))
                # Add migration logic here
                migration_results['components_migrated'].append('learning')
            except Exception as e:
                migration_results['errors'].append(f"Learning migration failed: {e}")
            
            # Migrate adaptation data
            try:
                adaptive_intelligence = AdaptiveIntelligence(str(self.data_dir))
                # Add migration logic here
                migration_results['components_migrated'].append('adaptation')
            except Exception as e:
                migration_results['errors'].append(f"Adaptation migration failed: {e}")
            
            migration_results['success'] = len(migration_results['errors']) == 0
            
        except Exception as e:
            migration_results['success'] = False
            migration_results['errors'].append(f"Migration failed: {e}")
        
        return migration_results
    
    def validate_export_file(self, export_file: str) -> Dict[str, Any]:
        """Validate an intelligence export file"""
        
        try:
            validation_results = {
                'file_path': export_file,
                'valid': True,
                'file_exists': False,
                'file_size': 0,
                'checksum_valid': False,
                'structure_valid': False,
                'version_supported': False,
                'components_found': [],
                'errors': [],
                'warnings': []
            }
            
            # Check file existence
            export_path = Path(export_file)
            if not export_path.exists():
                validation_results['valid'] = False
                validation_results['errors'].append(f"Export file does not exist: {export_file}")
                return validation_results
            
            validation_results['file_exists'] = True
            validation_results['file_size'] = export_path.stat().st_size
            
            # Load and validate structure
            try:
                export_package = self._load_export(export_file)
                if export_package:
                    validation_results['structure_valid'] = True
                    validation_results['version_supported'] = export_package.version in ["1.0"]
                    
                    # Check components
                    if export_package.dictionary_data:
                        validation_results['components_found'].append('dictionary')
                    if export_package.learning_data:
                        validation_results['components_found'].append('learning')
                    if export_package.adaptation_data:
                        validation_results['components_found'].append('adaptation')
                    
                    # Validate checksum if available
                    calculated_checksum = self._calculate_file_checksum(export_file)
                    
                    # Find matching snapshot in history
                    matching_snapshot = None
                    for snapshot in self.export_history:
                        if snapshot.snapshot_id == export_package.export_id:
                            matching_snapshot = snapshot
                            break
                    
                    if matching_snapshot:
                        validation_results['checksum_valid'] = calculated_checksum == matching_snapshot.checksum
                        if not validation_results['checksum_valid']:
                            validation_results['warnings'].append("File checksum does not match recorded checksum")
                    else:
                        validation_results['warnings'].append("No matching export record found in history")
                
                else:
                    validation_results['valid'] = False
                    validation_results['errors'].append("Failed to load export package")
                    
            except Exception as e:
                validation_results['valid'] = False
                validation_results['structure_valid'] = False
                validation_results['errors'].append(f"Structure validation failed: {e}")
            
            # Overall validation
            validation_results['valid'] = (
                validation_results['file_exists'] and
                validation_results['structure_valid'] and
                validation_results['version_supported'] and
                len(validation_results['components_found']) > 0 and
                len(validation_results['errors']) == 0
            )
            
            logger.debug(f"Export validation completed: {validation_results['valid']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate export file: {e}")
            return {'valid': False, 'error': str(e)}
    
    def list_available_exports(self) -> List[Dict[str, Any]]:
        """List all available intelligence exports"""
        
        try:
            exports = []
            
            for snapshot in sorted(self.export_history, key=lambda s: s.created_at, reverse=True):
                export_info = {
                    'snapshot_id': snapshot.snapshot_id,
                    'created_at': snapshot.created_at.isoformat(),
                    'version': snapshot.version,
                    'description': snapshot.description,
                    'data_types': snapshot.data_types,
                    'file_size': snapshot.file_size,
                    'file_size_mb': round(snapshot.file_size / (1024 * 1024), 2),
                    'checksum': snapshot.checksum[:16] + "...",
                    'export_type': snapshot.metadata.get('export_type', 'manual'),
                    'age_hours': (datetime.now() - snapshot.created_at).total_seconds() / 3600
                }
                exports.append(export_info)
            
            return exports
            
        except Exception as e:
            logger.error(f"Failed to list available exports: {e}")
            return []
    
    def cleanup_old_exports(self, keep_count: Optional[int] = None) -> int:
        """Clean up old export files"""
        
        try:
            if keep_count is None:
                keep_count = self.max_export_history
            
            if len(self.export_history) <= keep_count:
                return 0
            
            # Sort by creation time (oldest first)
            sorted_exports = sorted(self.export_history, key=lambda s: s.created_at)
            
            # Keep the most recent exports
            exports_to_remove = sorted_exports[:-keep_count]
            
            removed_count = 0
            for snapshot in exports_to_remove:
                try:
                    # Remove file
                    export_file = self.data_dir / f"{snapshot.snapshot_id}.json"
                    if self.compress_exports:
                        export_file = Path(str(export_file) + ".gz")
                    
                    if export_file.exists():
                        export_file.unlink()
                        removed_count += 1
                    
                    # Remove from history
                    self.export_history.remove(snapshot)
                    
                except Exception as e:
                    logger.warning(f"Failed to remove export {snapshot.snapshot_id}: {e}")
            
            # Save updated history
            self._save_export_history()
            
            logger.info(f"Cleaned up {removed_count} old export files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old exports: {e}")
            return 0
    
    def get_intelligence_statistics(self) -> Dict[str, Any]:
        """Get comprehensive intelligence system statistics"""
        
        try:
            # Get component statistics
            dictionary_manager = SmartDictionaryManager(self.config)
            continuous_learner = ContinuousLearner(str(self.data_dir))
            adaptive_intelligence = AdaptiveIntelligence(str(self.data_dir))
            
            return {
                'export_statistics': {
                    'total_exports': len(self.export_history),
                    'total_imports': len(self.import_history),
                    'latest_export': self.export_history[-1].created_at.isoformat() if self.export_history else None,
                    'latest_import': self.import_history[-1]['timestamp'] if self.import_history else None,
                    'total_export_size': sum(s.file_size for s in self.export_history),
                    'backup_count': len([s for s in self.export_history if s.metadata.get('export_type') == 'backup'])
                },
                'component_statistics': {
                    'dictionary': dictionary_manager.get_intelligence_stats(),
                    'learning': continuous_learner.get_learning_statistics(),
                    'adaptation': adaptive_intelligence.get_adaptation_statistics()
                },
                'system_configuration': {
                    'compress_exports': self.compress_exports,
                    'max_export_history': self.max_export_history,
                    'auto_backup_enabled': self.auto_backup_enabled,
                    'backup_interval_hours': self.backup_interval_hours
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get intelligence statistics: {e}")
            return {'error': str(e)}
    
    def _save_export(self, export_package: IntelligenceExport) -> Optional[str]:
        """Save export package to file"""
        
        try:
            # Convert to serializable format
            export_data = {
                'export_id': export_package.export_id,
                'created_at': export_package.created_at.isoformat(),
                'version': export_package.version,
                'description': export_package.description,
                'dictionary_data': export_package.dictionary_data,
                'learning_data': export_package.learning_data,
                'adaptation_data': export_package.adaptation_data,
                'metadata': export_package.metadata
            }
            
            # Save to file
            export_file = self.data_dir / f"{export_package.export_id}.json"
            
            if self.compress_exports:
                export_file = Path(str(export_file) + ".gz")
                with gzip.open(export_file, 'wt', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved export to {export_file}")
            return str(export_file)
            
        except Exception as e:
            logger.error(f"Failed to save export: {e}")
            return None
    
    def _load_export(self, export_file: str) -> Optional[IntelligenceExport]:
        """Load export package from file"""
        
        try:
            export_path = Path(export_file)
            
            if export_file.endswith('.gz'):
                with gzip.open(export_path, 'rt', encoding='utf-8') as f:
                    export_data = json.load(f)
            else:
                with open(export_path, 'r', encoding='utf-8') as f:
                    export_data = json.load(f)
            
            # Convert back to IntelligenceExport
            export_package = IntelligenceExport(
                export_id=export_data['export_id'],
                created_at=datetime.fromisoformat(export_data['created_at']),
                version=export_data['version'],
                description=export_data['description'],
                dictionary_data=export_data.get('dictionary_data'),
                learning_data=export_data.get('learning_data'),
                adaptation_data=export_data.get('adaptation_data'),
                metadata=export_data.get('metadata', {})
            )
            
            logger.debug(f"Loaded export from {export_file}")
            return export_package
            
        except Exception as e:
            logger.error(f"Failed to load export: {e}")
            return None
    
    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file"""
        
        try:
            sha256_hash = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to calculate checksum: {e}")
            return ""
    
    def _save_export_history(self) -> None:
        """Save export history to file"""
        
        try:
            history_data = []
            
            for snapshot in self.export_history:
                history_data.append({
                    'snapshot_id': snapshot.snapshot_id,
                    'created_at': snapshot.created_at.isoformat(),
                    'version': snapshot.version,
                    'description': snapshot.description,
                    'data_types': snapshot.data_types,
                    'file_size': snapshot.file_size,
                    'checksum': snapshot.checksum,
                    'metadata': snapshot.metadata
                })
            
            history_file = self.data_dir / 'export_history.json'
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved export history: {len(history_data)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save export history: {e}")
    
    def _load_export_history(self) -> None:
        """Load export history from file"""
        
        try:
            history_file = self.data_dir / 'export_history.json'
            
            if not history_file.exists():
                logger.info("No existing export history found")
                return
            
            with open(history_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            for entry in history_data:
                snapshot = IntelligenceSnapshot(
                    snapshot_id=entry['snapshot_id'],
                    created_at=datetime.fromisoformat(entry['created_at']),
                    version=entry['version'],
                    description=entry['description'],
                    data_types=entry['data_types'],
                    file_size=entry['file_size'],
                    checksum=entry['checksum'],
                    metadata=entry.get('metadata', {})
                )
                self.export_history.append(snapshot)
            
            logger.info(f"Loaded export history: {len(self.export_history)} entries")
            
        except Exception as e:
            logger.warning(f"Failed to load export history: {e}")
    
    def _save_import_history(self) -> None:
        """Save import history to file"""
        
        try:
            history_file = self.data_dir / 'import_history.json'
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.import_history, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved import history: {len(self.import_history)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save import history: {e}")
    
    def _load_import_history(self) -> None:
        """Load import history from file"""
        
        try:
            history_file = self.data_dir / 'import_history.json'
            
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.import_history = json.load(f)
                
                logger.info(f"Loaded import history: {len(self.import_history)} entries")
            
        except Exception as e:
            logger.warning(f"Failed to load import history: {e}")