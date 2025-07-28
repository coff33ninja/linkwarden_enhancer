"""Recovery System - Automated rollback and manual recovery procedures"""

import json
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from ..utils.logging_utils import get_logger
from .backup_system import BackupSystem, BackupInfo
from .integrity_checker import IntegrityChecker

logger = get_logger(__name__)


@dataclass
class RecoveryPlan:
    """Recovery plan with steps and verification"""
    recovery_id: str
    backup_path: str
    target_file: str
    recovery_steps: List[str]
    verification_steps: List[str]
    estimated_time: float
    risk_level: str  # 'low', 'medium', 'high'
    prerequisites: List[str]


@dataclass
class RecoveryResult:
    """Result of recovery operation"""
    success: bool
    recovery_id: str
    backup_used: str
    target_file: str
    recovery_time: float
    verification_passed: bool
    issues_found: List[str]
    warnings: List[str]
    recovery_log: List[str]


class RecoverySystem:
    """Automated rollback and manual recovery system"""
    
    def __init__(self, config: Dict[str, Any], backup_system: BackupSystem):
        """Initialize recovery system with configuration and backup system"""
        self.config = config
        self.recovery_config = config.get('recovery', {})
        self.backup_system = backup_system
        
        # Initialize integrity checker for verification
        self.integrity_checker = IntegrityChecker(config)
        
        # Recovery settings
        self.auto_verify_recovery = self.recovery_config.get('auto_verify_recovery', True)
        self.create_pre_recovery_backup = self.recovery_config.get('create_pre_recovery_backup', True)
        self.max_recovery_attempts = self.recovery_config.get('max_recovery_attempts', 3)
        
        # Recovery history
        self.recovery_history = []
        
        logger.info("Recovery system initialized")
    
    def rollback_to_latest_backup(self, 
                                 target_file: str, 
                                 operation_name: Optional[str] = None) -> RecoveryResult:
        """Rollback to the most recent backup"""
        
        try:
            # Find latest backup
            latest_backup = self.backup_system.get_latest_backup(operation_name)
            
            if not latest_backup:
                error_msg = f"No backup found for operation: {operation_name or 'any'}"
                logger.error(error_msg)
                return RecoveryResult(
                    success=False,
                    recovery_id="",
                    backup_used="",
                    target_file=target_file,
                    recovery_time=0.0,
                    verification_passed=False,
                    issues_found=[error_msg],
                    warnings=[],
                    recovery_log=[]
                )
            
            return self.rollback_to_backup(latest_backup.path, target_file)
            
        except Exception as e:
            logger.error(f"Failed to rollback to latest backup: {e}")
            return RecoveryResult(
                success=False,
                recovery_id="",
                backup_used="",
                target_file=target_file,
                recovery_time=0.0,
                verification_passed=False,
                issues_found=[str(e)],
                warnings=[],
                recovery_log=[]
            )
    
    def rollback_to_backup(self, backup_path: str, target_file: str) -> RecoveryResult:
        """Rollback to a specific backup"""
        
        start_time = datetime.now()
        recovery_id = f"recovery_{int(start_time.timestamp())}"
        recovery_log = []
        
        try:
            logger.info(f"Starting rollback: {backup_path} -> {target_file}")
            recovery_log.append(f"Started rollback at {start_time.isoformat()}")
            
            # Verify backup exists and is valid
            if not Path(backup_path).exists():
                error_msg = f"Backup file does not exist: {backup_path}"
                logger.error(error_msg)
                return RecoveryResult(
                    success=False,
                    recovery_id=recovery_id,
                    backup_used=backup_path,
                    target_file=target_file,
                    recovery_time=0.0,
                    verification_passed=False,
                    issues_found=[error_msg],
                    warnings=[],
                    recovery_log=recovery_log
                )
            
            # Verify backup integrity
            if not self.backup_system.verify_backup_integrity(backup_path):
                error_msg = f"Backup integrity check failed: {backup_path}"
                logger.error(error_msg)
                return RecoveryResult(
                    success=False,
                    recovery_id=recovery_id,
                    backup_used=backup_path,
                    target_file=target_file,
                    recovery_time=0.0,
                    verification_passed=False,
                    issues_found=[error_msg],
                    warnings=[],
                    recovery_log=recovery_log
                )
            
            recovery_log.append("Backup integrity verified")
            
            # Create pre-recovery backup if enabled
            pre_recovery_backup = None
            if self.create_pre_recovery_backup and Path(target_file).exists():
                pre_recovery_backup = self.backup_system.create_backup(
                    target_file, 
                    "pre_recovery",
                    {"recovery_id": recovery_id, "original_backup": backup_path}
                )
                if pre_recovery_backup:
                    recovery_log.append(f"Created pre-recovery backup: {pre_recovery_backup}")
                else:
                    recovery_log.append("Warning: Failed to create pre-recovery backup")
            
            # Perform the rollback
            success = self.backup_system.restore_backup(backup_path, target_file)
            
            if not success:
                error_msg = "Backup restoration failed"
                logger.error(error_msg)
                return RecoveryResult(
                    success=False,
                    recovery_id=recovery_id,
                    backup_used=backup_path,
                    target_file=target_file,
                    recovery_time=(datetime.now() - start_time).total_seconds(),
                    verification_passed=False,
                    issues_found=[error_msg],
                    warnings=[],
                    recovery_log=recovery_log
                )
            
            recovery_log.append("Backup restored successfully")
            
            # Verify recovery if enabled
            verification_passed = True
            verification_issues = []
            
            if self.auto_verify_recovery:
                verification_result = self._verify_recovery(target_file, backup_path)
                verification_passed = verification_result['success']
                verification_issues = verification_result.get('issues', [])
                recovery_log.extend(verification_result.get('log', []))
            
            recovery_time = (datetime.now() - start_time).total_seconds()
            
            # Create recovery result
            result = RecoveryResult(
                success=success and verification_passed,
                recovery_id=recovery_id,
                backup_used=backup_path,
                target_file=target_file,
                recovery_time=recovery_time,
                verification_passed=verification_passed,
                issues_found=verification_issues,
                warnings=[],
                recovery_log=recovery_log
            )
            
            # Add to history
            self._add_to_recovery_history(result)
            
            logger.info(f"Rollback completed: {'SUCCESS' if result.success else 'FAILED'} ({recovery_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            recovery_time = (datetime.now() - start_time).total_seconds()
            
            return RecoveryResult(
                success=False,
                recovery_id=recovery_id,
                backup_used=backup_path,
                target_file=target_file,
                recovery_time=recovery_time,
                verification_passed=False,
                issues_found=[str(e)],
                warnings=[],
                recovery_log=recovery_log
            )
    
    def create_recovery_plan(self, 
                           target_file: str, 
                           backup_path: Optional[str] = None) -> RecoveryPlan:
        """Create a detailed recovery plan"""
        
        try:
            recovery_id = f"plan_{int(datetime.now().timestamp())}"
            
            # Determine backup to use
            if not backup_path:
                latest_backup = self.backup_system.get_latest_backup()
                if not latest_backup:
                    raise ValueError("No backup available for recovery")
                backup_path = latest_backup.path
            
            # Get backup info
            backup_info = self.backup_system.get_backup_info(backup_path)
            
            # Create recovery steps
            recovery_steps = [
                "1. Verify backup file exists and is accessible",
                "2. Check backup integrity using checksums",
                "3. Create pre-recovery backup of current state (if file exists)",
                "4. Restore backup file to target location",
                "5. Verify restored file integrity",
                "6. Validate data consistency in restored file"
            ]
            
            # Create verification steps
            verification_steps = [
                "1. Check file size matches backup",
                "2. Verify JSON structure is valid",
                "3. Validate data relationships and integrity",
                "4. Compare item counts with backup metadata",
                "5. Check for any data corruption"
            ]
            
            # Estimate recovery time based on file size
            estimated_time = 30.0  # Base time in seconds
            if backup_info:
                # Add time based on file size (rough estimate)
                size_mb = backup_info.file_size / (1024 * 1024)
                estimated_time += size_mb * 2  # 2 seconds per MB
            
            # Determine risk level
            risk_level = "low"
            if Path(target_file).exists():
                risk_level = "medium"  # Overwriting existing file
            
            # Prerequisites
            prerequisites = [
                "Ensure no other processes are using the target file",
                "Verify sufficient disk space for recovery operation",
                "Confirm backup file is not corrupted"
            ]
            
            if backup_info and backup_info.compressed:
                prerequisites.append("Ensure gzip decompression is available")
            
            plan = RecoveryPlan(
                recovery_id=recovery_id,
                backup_path=backup_path,
                target_file=target_file,
                recovery_steps=recovery_steps,
                verification_steps=verification_steps,
                estimated_time=estimated_time,
                risk_level=risk_level,
                prerequisites=prerequisites
            )
            
            logger.info(f"Recovery plan created: {recovery_id}")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create recovery plan: {e}")
            raise
    
    def generate_recovery_script(self, recovery_plan: RecoveryPlan) -> str:
        """Generate a manual recovery script"""
        
        try:
            script_lines = [
                "#!/bin/bash",
                "# Linkwarden Enhancer Recovery Script",
                f"# Generated: {datetime.now().isoformat()}",
                f"# Recovery ID: {recovery_plan.recovery_id}",
                "",
                "set -e  # Exit on any error",
                "",
                "# Configuration",
                f'BACKUP_FILE="{recovery_plan.backup_path}"',
                f'TARGET_FILE="{recovery_plan.target_file}"',
                f'RECOVERY_ID="{recovery_plan.recovery_id}"',
                "",
                "echo 'Starting manual recovery...'",
                "",
                "# Step 1: Verify backup exists",
                "if [ ! -f \"$BACKUP_FILE\" ]; then",
                "    echo 'ERROR: Backup file does not exist'",
                "    exit 1",
                "fi",
                "",
                "# Step 2: Create pre-recovery backup",
                "if [ -f \"$TARGET_FILE\" ]; then",
                "    TIMESTAMP=$(date +%Y%m%d_%H%M%S)",
                "    PRE_RECOVERY_BACKUP=\"${TARGET_FILE}.pre_recovery_${TIMESTAMP}\"",
                "    echo \"Creating pre-recovery backup: $PRE_RECOVERY_BACKUP\"",
                "    cp \"$TARGET_FILE\" \"$PRE_RECOVERY_BACKUP\"",
                "fi",
                "",
                "# Step 3: Restore from backup",
                "echo 'Restoring from backup...'",
            ]
            
            # Add decompression if needed
            backup_info = self.backup_system.get_backup_info(recovery_plan.backup_path)
            if backup_info and backup_info.compressed:
                script_lines.extend([
                    "# Decompress backup",
                    "gunzip -c \"$BACKUP_FILE\" > \"$TARGET_FILE\"",
                ])
            else:
                script_lines.extend([
                    "# Copy backup",
                    "cp \"$BACKUP_FILE\" \"$TARGET_FILE\"",
                ])
            
            script_lines.extend([
                "",
                "# Step 4: Verify recovery",
                "if [ -f \"$TARGET_FILE\" ]; then",
                "    echo 'Recovery completed successfully'",
                "    echo \"Restored file: $TARGET_FILE\"",
                "    echo \"File size: $(stat -c%s \"$TARGET_FILE\") bytes\"",
                "else",
                "    echo 'ERROR: Recovery failed - target file not found'",
                "    exit 1",
                "fi",
                "",
                "echo 'Manual recovery completed!'",
                ""
            ])
            
            script_content = "\n".join(script_lines)
            
            logger.info(f"Recovery script generated for {recovery_plan.recovery_id}")
            return script_content
            
        except Exception as e:
            logger.error(f"Failed to generate recovery script: {e}")
            return f"# Error generating recovery script: {e}"
    
    def generate_recovery_documentation(self, recovery_plan: RecoveryPlan) -> str:
        """Generate manual recovery documentation"""
        
        try:
            backup_info = self.backup_system.get_backup_info(recovery_plan.backup_path)
            
            doc_lines = [
                "# Manual Recovery Documentation",
                f"**Recovery ID:** {recovery_plan.recovery_id}",
                f"**Generated:** {datetime.now().isoformat()}",
                "",
                "## Overview",
                f"This document provides step-by-step instructions for manually recovering",
                f"your Linkwarden data from backup.",
                "",
                "## Recovery Details",
                f"- **Backup File:** `{recovery_plan.backup_path}`",
                f"- **Target File:** `{recovery_plan.target_file}`",
                f"- **Estimated Time:** {recovery_plan.estimated_time:.1f} seconds",
                f"- **Risk Level:** {recovery_plan.risk_level.upper()}",
                "",
            ]
            
            if backup_info:
                doc_lines.extend([
                    "## Backup Information",
                    f"- **Created:** {backup_info.timestamp.isoformat()}",
                    f"- **Operation:** {backup_info.operation_name}",
                    f"- **File Size:** {backup_info.file_size:,} bytes",
                    f"- **Compressed:** {'Yes' if backup_info.compressed else 'No'}",
                    f"- **Checksum:** {backup_info.checksum[:16]}...",
                    "",
                ])
            
            doc_lines.extend([
                "## Prerequisites",
                "",
            ])
            
            for i, prereq in enumerate(recovery_plan.prerequisites, 1):
                doc_lines.append(f"{i}. {prereq}")
            
            doc_lines.extend([
                "",
                "## Recovery Steps",
                "",
            ])
            
            for step in recovery_plan.recovery_steps:
                doc_lines.append(f"### {step}")
                doc_lines.append("")
                
                # Add detailed instructions for each step
                if "Verify backup file" in step:
                    doc_lines.extend([
                        "```bash",
                        f"ls -la '{recovery_plan.backup_path}'",
                        "```",
                        "",
                    ])
                elif "Create pre-recovery backup" in step:
                    doc_lines.extend([
                        "```bash",
                        f"cp '{recovery_plan.target_file}' '{recovery_plan.target_file}.backup_$(date +%Y%m%d_%H%M%S)'",
                        "```",
                        "",
                    ])
                elif "Restore backup file" in step:
                    if backup_info and backup_info.compressed:
                        doc_lines.extend([
                            "```bash",
                            f"gunzip -c '{recovery_plan.backup_path}' > '{recovery_plan.target_file}'",
                            "```",
                            "",
                        ])
                    else:
                        doc_lines.extend([
                            "```bash",
                            f"cp '{recovery_plan.backup_path}' '{recovery_plan.target_file}'",
                            "```",
                            "",
                        ])
            
            doc_lines.extend([
                "## Verification Steps",
                "",
            ])
            
            for step in recovery_plan.verification_steps:
                doc_lines.append(f"- {step}")
            
            doc_lines.extend([
                "",
                "## Troubleshooting",
                "",
                "### Common Issues",
                "",
                "1. **Permission Denied**",
                "   - Ensure you have write permissions to the target directory",
                "   - Try running with appropriate privileges",
                "",
                "2. **File Not Found**",
                "   - Verify the backup file path is correct",
                "   - Check if the backup file was moved or deleted",
                "",
                "3. **Insufficient Disk Space**",
                "   - Check available disk space with `df -h`",
                "   - Free up space if needed",
                "",
                "4. **Corrupted Backup**",
                "   - Try using an earlier backup",
                "   - Check backup integrity if tools are available",
                "",
                "### Getting Help",
                "",
                "If you encounter issues during recovery:",
                "1. Check the application logs for error details",
                "2. Verify all prerequisites are met",
                "3. Try using the automated recovery system if available",
                "",
                "---",
                f"*Generated by Linkwarden Enhancer Recovery System*",
                ""
            ])
            
            documentation = "\n".join(doc_lines)
            
            logger.info(f"Recovery documentation generated for {recovery_plan.recovery_id}")
            return documentation
            
        except Exception as e:
            logger.error(f"Failed to generate recovery documentation: {e}")
            return f"# Error generating recovery documentation: {e}"
    
    def list_recovery_options(self, operation_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available recovery options"""
        
        try:
            backups = self.backup_system.list_backups(operation_name)
            
            recovery_options = []
            for backup in backups:
                option = {
                    'backup_path': backup.path,
                    'timestamp': backup.timestamp.isoformat(),
                    'operation_name': backup.operation_name,
                    'file_size': backup.file_size,
                    'compressed': backup.compressed,
                    'age_hours': (datetime.now() - backup.timestamp).total_seconds() / 3600,
                    'recommended': False
                }
                
                # Mark most recent backup as recommended
                if backup == backups[0]:
                    option['recommended'] = True
                
                recovery_options.append(option)
            
            logger.info(f"Found {len(recovery_options)} recovery options")
            return recovery_options
            
        except Exception as e:
            logger.error(f"Failed to list recovery options: {e}")
            return []
    
    def _verify_recovery(self, target_file: str, backup_path: str) -> Dict[str, Any]:
        """Verify that recovery was successful"""
        
        verification_log = []
        issues = []
        
        try:
            verification_log.append("Starting recovery verification")
            
            # Check if target file exists
            if not Path(target_file).exists():
                issues.append("Target file does not exist after recovery")
                return {'success': False, 'issues': issues, 'log': verification_log}
            
            verification_log.append("Target file exists")
            
            # Load and validate JSON structure
            try:
                with open(target_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                verification_log.append("JSON structure is valid")
            except json.JSONDecodeError as e:
                issues.append(f"Invalid JSON structure: {e}")
                return {'success': False, 'issues': issues, 'log': verification_log}
            
            # Run integrity check
            integrity_result = self.integrity_checker.check_data_integrity(data)
            
            if not integrity_result.success:
                issues.extend(integrity_result.integrity_issues[:5])  # Show first 5 issues
                verification_log.append(f"Integrity check found {len(integrity_result.integrity_issues)} issues")
            else:
                verification_log.append("Data integrity check passed")
            
            # Check basic data counts
            bookmarks_count = len(data.get('bookmarks', []))
            collections_count = len(data.get('collections', []))
            tags_count = len(data.get('tags', []))
            
            verification_log.append(f"Data counts: {bookmarks_count} bookmarks, {collections_count} collections, {tags_count} tags")
            
            success = len(issues) == 0
            verification_log.append(f"Recovery verification: {'PASSED' if success else 'FAILED'}")
            
            return {
                'success': success,
                'issues': issues,
                'log': verification_log,
                'data_counts': {
                    'bookmarks': bookmarks_count,
                    'collections': collections_count,
                    'tags': tags_count
                }
            }
            
        except Exception as e:
            issues.append(f"Verification error: {e}")
            verification_log.append(f"Verification failed with error: {e}")
            return {'success': False, 'issues': issues, 'log': verification_log}
    
    def _add_to_recovery_history(self, result: RecoveryResult) -> None:
        """Add recovery result to history"""
        
        history_entry = {
            'recovery_id': result.recovery_id,
            'timestamp': datetime.now().isoformat(),
            'success': result.success,
            'backup_used': result.backup_used,
            'target_file': result.target_file,
            'recovery_time': result.recovery_time,
            'verification_passed': result.verification_passed,
            'issues_count': len(result.issues_found),
            'warnings_count': len(result.warnings)
        }
        
        self.recovery_history.append(history_entry)
        
        # Keep history limited
        if len(self.recovery_history) > 20:
            self.recovery_history = self.recovery_history[-20:]
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery system statistics"""
        
        try:
            total_recoveries = len(self.recovery_history)
            successful_recoveries = sum(1 for entry in self.recovery_history if entry['success'])
            
            avg_recovery_time = 0.0
            if self.recovery_history:
                avg_recovery_time = sum(entry['recovery_time'] for entry in self.recovery_history) / len(self.recovery_history)
            
            return {
                'total_recoveries': total_recoveries,
                'successful_recoveries': successful_recoveries,
                'success_rate': (successful_recoveries / total_recoveries * 100) if total_recoveries > 0 else 0,
                'average_recovery_time': avg_recovery_time,
                'settings': {
                    'auto_verify_recovery': self.auto_verify_recovery,
                    'create_pre_recovery_backup': self.create_pre_recovery_backup,
                    'max_recovery_attempts': self.max_recovery_attempts
                },
                'recent_recoveries': self.recovery_history[-5:] if self.recovery_history else []
            }
            
        except Exception as e:
            logger.error(f"Failed to get recovery statistics: {e}")
            return {'error': str(e)}