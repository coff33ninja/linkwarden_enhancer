"""Safety Manager - Central orchestrator for all safety operations"""

import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from utils.logging_utils import get_logger
from importers.universal_importer import UniversalImporter, ImportConfig
from utils.file_utils import FileUtils
from intelligence.dictionary_manager import SmartDictionaryManager
from core.validation_engine import ValidationEngine
from core.backup_system import BackupSystem
from core.progress_monitor import ProgressMonitor
from core.integrity_checker import IntegrityChecker
from core.recovery_system import RecoverySystem

logger = get_logger(__name__)


class SafetyManager:
    """Central orchestrator for all safety operations"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize SafetyManager with configuration"""
        self.config = config
        self.safety_config = config.get("safety", {})

        # Initialize core safety components
        self.validator = ValidationEngine(config)
        self.backup_system = BackupSystem(config)
        self.progress_monitor = ProgressMonitor(config)
        self.integrity_checker = IntegrityChecker(config)
        self.recovery_system = RecoverySystem(config, self.backup_system)

        # Initialize importers and intelligence
        self.universal_importer = UniversalImporter(config)
        self.smart_dictionary = SmartDictionaryManager(config)

        logger.info("SafetyManager initialized with all core safety components")

    def execute_safe_cleanup(
        self,
        input_file: str,
        output_file: str,
        import_github: bool = False,
        import_browser: Optional[str] = None,
    ) -> "SafetyResult":
        """Execute safe cleanup with comprehensive safety checks"""

        start_time = time.time()
        logger.info(f"Starting safe cleanup: {input_file} -> {output_file}")

        # Start progress monitoring
        operation_id = self.progress_monitor.start_operation(
            "safe_cleanup",
            total_items=100,  # Will be updated as we discover actual item counts
            description="Initializing safe cleanup operation",
        )

        try:
            # Step 1: Validate input file
            self.progress_monitor.update_progress(
                operation_id, 5, "Validating input file"
            )

            if not Path(input_file).exists():
                error_msg = f"Input file does not exist: {input_file}"
                self.progress_monitor.add_error(operation_id, error_msg)
                self.progress_monitor.complete_operation(operation_id, False)

                from data_models import SafetyResult, ChangeSet

                return SafetyResult(
                    success=False,
                    changes_applied=ChangeSet(),
                    backups_created=[],
                    integrity_report=None,
                    enhancement_report=None,
                    execution_time=time.time() - start_time,
                    warnings=[],
                    errors=[error_msg],
                )

            # Load and validate input data
            self.progress_monitor.update_progress(
                operation_id, 10, "Loading and validating input data"
            )

            try:
                import json

                with open(input_file, "r", encoding="utf-8") as f:
                    input_data = json.load(f)
            except Exception as e:
                error_msg = f"Failed to load input file: {e}"
                self.progress_monitor.add_error(operation_id, error_msg)
                self.progress_monitor.complete_operation(operation_id, False)

                from data_models import SafetyResult, ChangeSet

                return SafetyResult(
                    success=False,
                    changes_applied=ChangeSet(),
                    backups_created=[],
                    integrity_report=None,
                    enhancement_report=None,
                    execution_time=time.time() - start_time,
                    warnings=[],
                    errors=[error_msg],
                )

            # Validate data schema and consistency
            validation_result = self.validator.validate_json_schema(
                input_data, "linkwarden_backup"
            )
            if not validation_result.valid:
                self.progress_monitor.add_error(
                    operation_id,
                    f"Schema validation failed: {validation_result.errors}",
                )

            consistency_result = self.validator.validate_data_consistency(input_data)
            if not consistency_result.valid:
                self.progress_monitor.add_error(
                    operation_id,
                    f"Data consistency check failed: {consistency_result.errors}",
                )

            # Update progress with actual item counts
            total_items = (
                validation_result.total_bookmarks
                + validation_result.total_collections
                + validation_result.total_tags
            )
            if total_items > 0:
                # Restart progress with accurate count
                self.progress_monitor.complete_operation(operation_id, True)
                operation_id = self.progress_monitor.start_operation(
                    "safe_cleanup",
                    total_items=total_items,
                    description=f"Processing {total_items} items",
                )

            # Step 2: Create backup
            self.progress_monitor.update_progress(operation_id, 15, "Creating backup")

            backups_created = []
            if not self.safety_config.get("dry_run_mode", False):
                backup_path = self.backup_system.create_backup(
                    input_file,
                    "pre_enhancement",
                    {"operation": "safe_cleanup", "target": output_file},
                )
                if backup_path:
                    backups_created.append(backup_path)
                    logger.info(f"Created backup: {backup_path}")
                else:
                    self.progress_monitor.add_warning(
                        operation_id, "Failed to create backup"
                    )

            # Step 3: Run integrity check on input
            self.progress_monitor.update_progress(
                operation_id, 25, "Running integrity check"
            )

            integrity_report = self.integrity_checker.check_data_integrity(input_data)
            if not integrity_report.success:
                self.progress_monitor.add_warning(
                    operation_id,
                    f"Integrity issues found: {len(integrity_report.integrity_issues)}",
                )

            # Step 4: Create import configuration
            self.progress_monitor.update_progress(
                operation_id, 35, "Configuring import sources"
            )

            import_config = ImportConfig(
                linkwarden_backup_path=input_file,
                github_token=self.config.get("github", {}).get("token"),
                github_username=self.config.get("github", {}).get("username"),
                import_github_starred=import_github,
                import_github_owned=import_github,
                browser_bookmarks_path=import_browser,
                dry_run=self.safety_config.get("dry_run_mode", False),
                verbose=True,
            )

            # Validate import configuration
            validation_errors = self.universal_importer.validate_import_config(
                import_config
            )
            if validation_errors:
                for error in validation_errors:
                    self.progress_monitor.add_error(operation_id, error)

            # Step 5: Import from all sources
            self.progress_monitor.update_progress(
                operation_id, 50, "Importing from all sources"
            )

            import_result = self.universal_importer.import_all_sources(import_config)

            # Step 6: Learn from imported data
            self.progress_monitor.update_progress(
                operation_id, 70, "Training intelligence systems"
            )

            all_bookmarks = import_result.get_all_bookmarks()
            if all_bookmarks:
                logger.info(
                    f"Training smart dictionary with {len(all_bookmarks)} bookmarks..."
                )
                learning_result = self.smart_dictionary.learn_from_bookmark_data(
                    all_bookmarks
                )
                logger.info(
                    f"Learning completed: {learning_result.get('success', False)}"
                )

            # Step 7: Write output and verify
            self.progress_monitor.update_progress(
                operation_id, 85, "Writing output file"
            )

            # TODO: Write enhanced output file here
            # For now, we'll just copy the input to output for testing
            if not self.safety_config.get("dry_run_mode", False):
                import shutil

                shutil.copy2(input_file, output_file)

            # Step 8: Final integrity check
            self.progress_monitor.update_progress(
                operation_id, 95, "Final integrity verification"
            )

            if (
                not self.safety_config.get("dry_run_mode", False)
                and Path(output_file).exists()
            ):
                try:
                    with open(output_file, "r", encoding="utf-8") as f:
                        output_data = json.load(f)

                    # Compare before and after
                    comparison_result = self.integrity_checker.compare_data_sets(
                        input_data, output_data
                    )
                    consistency_check = (
                        self.integrity_checker.validate_before_after_consistency(
                            input_data, output_data
                        )
                    )

                    if not consistency_check["success"]:
                        for issue in consistency_check["issues"]:
                            self.progress_monitor.add_error(operation_id, issue)

                except Exception as e:
                    self.progress_monitor.add_warning(
                        operation_id, f"Output verification failed: {e}"
                    )

            # Complete operation
            self.progress_monitor.update_progress(
                operation_id, 100, "Cleanup completed"
            )

            # Determine success
            progress_info = self.progress_monitor.get_progress(operation_id)
            has_critical_errors = any(
                "CRITICAL" in error or "SAFETY" in error
                for error in progress_info.errors
            )
            success = len(progress_info.errors) == 0 or not has_critical_errors

            self.progress_monitor.complete_operation(operation_id, success)

            logger.info(f"Import completed: {import_result.get_summary()}")

            # Create result
            from data_models import SafetyResult, ChangeSet, EnhancementReport

            # Create enhancement report
            enhancement_report = EnhancementReport(
                bookmarks_enhanced=import_result.total_bookmarks,
                metadata_fields_added=0,  # Will be calculated later
                scraping_failures=0,
                scrapers_used={},
                average_scraping_time=0.0,
                cache_hit_rate=0.0,
            )

            result = SafetyResult(
                success=success,
                changes_applied=ChangeSet(),  # Will be populated with actual changes
                backups_created=backups_created,
                integrity_report=integrity_report,
                enhancement_report=enhancement_report,
                execution_time=time.time() - start_time,
                warnings=import_result.warnings + progress_info.warnings,
                errors=import_result.errors + progress_info.errors,
            )

            logger.info(f"Safe cleanup completed in {result.execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Safe cleanup failed: {e}")
            self.progress_monitor.add_error(operation_id, str(e))
            self.progress_monitor.complete_operation(operation_id, False)

            from data_models import SafetyResult, ChangeSet

            return SafetyResult(
                success=False,
                changes_applied=ChangeSet(),
                backups_created=[],
                integrity_report=None,
                enhancement_report=None,
                execution_time=time.time() - start_time,
                warnings=[],
                errors=[str(e)],
            )

    def import_from_github(self) -> "SafetyResult":
        """Import from GitHub with safety checks"""

        start_time = time.time()
        logger.info("Starting GitHub import")

        try:
            # Create GitHub-only import configuration
            import_config = ImportConfig(
                github_token=self.config.get("github", {}).get("token"),
                github_username=self.config.get("github", {}).get("username"),
                import_github_starred=self.config.get("github", {}).get(
                    "import_starred", True
                ),
                import_github_owned=self.config.get("github", {}).get(
                    "import_owned_repos", True
                ),
                dry_run=self.safety_config.get("dry_run_mode", False),
                verbose=True,
            )

            # Validate configuration
            validation_errors = self.universal_importer.validate_import_config(
                import_config
            )
            if validation_errors:
                logger.error(
                    f"GitHub import configuration validation failed: {validation_errors}"
                )
                from data_models import SafetyResult, ChangeSet

                return SafetyResult(
                    success=False,
                    changes_applied=ChangeSet(),
                    backups_created=[],
                    integrity_report=None,
                    enhancement_report=None,
                    execution_time=time.time() - start_time,
                    warnings=[],
                    errors=validation_errors,
                )

            # Import from GitHub
            import_result = self.universal_importer.import_all_sources(import_config)

            # Learn from GitHub data
            all_bookmarks = import_result.get_all_bookmarks()
            if all_bookmarks:
                logger.info(
                    f"Training smart dictionary with {len(all_bookmarks)} GitHub bookmarks..."
                )
                learning_result = self.smart_dictionary.learn_from_bookmark_data(
                    all_bookmarks
                )
                logger.info(
                    f"Learning completed: {learning_result.get('success', False)}"
                )

            logger.info(f"GitHub import completed: {import_result.get_summary()}")

            # Create result
            from data_models import SafetyResult, ChangeSet, EnhancementReport

            enhancement_report = EnhancementReport(
                bookmarks_enhanced=import_result.total_bookmarks,
                metadata_fields_added=0,
                scraping_failures=0,
                scrapers_used={},
                average_scraping_time=0.0,
                cache_hit_rate=0.0,
            )

            result = SafetyResult(
                success=len(import_result.errors) == 0,
                changes_applied=ChangeSet(),
                backups_created=[],
                integrity_report=None,
                enhancement_report=enhancement_report,
                execution_time=time.time() - start_time,
                warnings=import_result.warnings,
                errors=import_result.errors,
            )

            logger.info(f"GitHub import completed in {result.execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"GitHub import failed: {e}")
            from data_models import SafetyResult, ChangeSet

            return SafetyResult(
                success=False,
                changes_applied=ChangeSet(),
                backups_created=[],
                integrity_report=None,
                enhancement_report=None,
                execution_time=time.time() - start_time,
                warnings=[],
                errors=[str(e)],
            )

    def rollback_to_backup(self, backup_path: str, target_file: str) -> Dict[str, Any]:
        """Rollback to a specific backup using the recovery system"""

        try:
            logger.info(f"Initiating rollback: {backup_path} -> {target_file}")

            recovery_result = self.recovery_system.rollback_to_backup(
                backup_path, target_file
            )

            return {
                "success": recovery_result.success,
                "recovery_id": recovery_result.recovery_id,
                "recovery_time": recovery_result.recovery_time,
                "verification_passed": recovery_result.verification_passed,
                "issues": recovery_result.issues_found,
                "warnings": recovery_result.warnings,
            }

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {"success": False, "error": str(e)}

    def create_recovery_plan(
        self, target_file: str, backup_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a recovery plan for manual recovery"""

        try:
            recovery_plan = self.recovery_system.create_recovery_plan(
                target_file, backup_path
            )

            return {
                "recovery_id": recovery_plan.recovery_id,
                "backup_path": recovery_plan.backup_path,
                "target_file": recovery_plan.target_file,
                "estimated_time": recovery_plan.estimated_time,
                "risk_level": recovery_plan.risk_level,
                "recovery_steps": recovery_plan.recovery_steps,
                "verification_steps": recovery_plan.verification_steps,
                "prerequisites": recovery_plan.prerequisites,
            }

        except Exception as e:
            logger.error(f"Failed to create recovery plan: {e}")
            return {"error": str(e)}

    def generate_recovery_documentation(
        self, target_file: str, backup_path: Optional[str] = None
    ) -> str:
        """Generate manual recovery documentation"""

        try:
            recovery_plan = self.recovery_system.create_recovery_plan(
                target_file, backup_path
            )
            return self.recovery_system.generate_recovery_documentation(recovery_plan)

        except Exception as e:
            logger.error(f"Failed to generate recovery documentation: {e}")
            return f"Error generating recovery documentation: {e}"

    def validate_data_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a data file using the validation engine"""

        try:
            # Load data
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Run validations
            schema_result = self.validator.validate_json_schema(
                data, "linkwarden_backup"
            )
            consistency_result = self.validator.validate_data_consistency(data)
            field_result = self.validator.validate_field_requirements(data)
            integrity_result = self.integrity_checker.check_data_integrity(data)

            # Create inventory
            inventory = self.validator.create_data_inventory(data)

            return {
                "file_path": file_path,
                "schema_valid": schema_result.valid,
                "consistency_valid": consistency_result.valid,
                "fields_valid": field_result.valid,
                "integrity_valid": integrity_result.success,
                "overall_valid": all(
                    [
                        schema_result.valid,
                        consistency_result.valid,
                        field_result.valid,
                        integrity_result.success,
                    ]
                ),
                "errors": (
                    schema_result.errors
                    + consistency_result.errors
                    + field_result.errors
                    + integrity_result.integrity_issues
                ),
                "warnings": (
                    schema_result.warnings
                    + consistency_result.warnings
                    + field_result.warnings
                ),
                "inventory": inventory,
                "validation_timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {"file_path": file_path, "error": str(e), "overall_valid": False}

    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get comprehensive safety system statistics"""

        try:
            return {
                "validation_stats": self.validator.get_validation_stats(),
                "backup_stats": self.backup_system.get_backup_statistics(),
                "progress_stats": self.progress_monitor.get_operation_statistics(),
                "integrity_stats": self.integrity_checker.get_integrity_statistics(),
                "recovery_stats": self.recovery_system.get_recovery_statistics(),
                "system_timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Failed to get safety statistics: {e}")
            return {"error": str(e)}

    def list_available_backups(
        self, operation_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available backups for recovery"""

        try:
            backups = self.backup_system.list_backups(operation_name)

            backup_list = []
            for backup in backups:
                backup_list.append(
                    {
                        "path": backup.path,
                        "timestamp": backup.timestamp.isoformat(),
                        "operation_name": backup.operation_name,
                        "file_size": backup.file_size,
                        "file_size_mb": round(backup.file_size / (1024 * 1024), 2),
                        "checksum": backup.checksum[:16] + "...",
                        "compressed": backup.compressed,
                        "age_hours": (time.time() - backup.timestamp.timestamp())
                        / 3600,
                        "metadata": backup.metadata,
                    }
                )

            return backup_list

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []

    def cleanup_old_backups(self) -> Dict[str, Any]:
        """Clean up old backups based on retention policy"""

        try:
            deleted_count = self.backup_system.cleanup_old_backups()

            return {
                "success": True,
                "deleted_count": deleted_count,
                "message": f"Cleaned up {deleted_count} old backups",
            }

        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            return {"success": False, "error": str(e)}
