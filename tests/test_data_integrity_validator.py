"""Tests for data integrity validation system"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

from enhancement.data_integrity_validator import (
    DataIntegrityValidator, ValidationLevel, IntegrityIssueType, 
    ValidationReport, RollbackPlan
)
from enhancement.graceful_degradation import EnhancementContext, ComponentError, ErrorSeverity


class TestDataIntegrityValidator:
    """Test data integrity validation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary directory for rollback plans
        self.temp_dir = Path(tempfile.mkdtemp())
        
        config = {
            'data_integrity': {
                'default_level': 'standard',
                'enable_rollback': True,
                'max_data_loss_threshold': 0.05,
                'rollback_dir': str(self.temp_dir),
                'required_fields': {
                    'bookmark': ['id', 'name', 'url'],
                    'collection': ['id', 'name'],
                    'tag': ['id', 'name']
                }
            },
            'validation': {},
            'verbose': False
        }
        
        self.validator = DataIntegrityValidator(config)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_successful_validation(self):
        """Test successful validation with no issues"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'},
                {'id': 2, 'name': 'Bookmark 2', 'url': 'https://example.com/2'}
            ],
            'collections': [
                {'id': 1, 'name': 'Collection 1'}
            ],
            'tags': [
                {'id': 1, 'name': 'tag1'},
                {'id': 2, 'name': 'tag2'}
            ]
        }
        
        enhanced_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Enhanced Bookmark 1', 'url': 'https://example.com/1', 'description': 'Added description'},
                {'id': 2, 'name': 'Enhanced Bookmark 2', 'url': 'https://example.com/2', 'tags': [{'id': 1, 'name': 'tag1'}]}
            ],
            'collections': [
                {'id': 1, 'name': 'Collection 1'}
            ],
            'tags': [
                {'id': 1, 'name': 'tag1'},
                {'id': 2, 'name': 'tag2'}
            ]
        }
        
        contexts = []
        
        report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.STANDARD
        )
        
        assert report.passed == True
        assert report.rollback_recommended == False
        assert len(report.integrity_issues) == 0
        assert report.total_items_before == 5  # 2 bookmarks + 1 collection + 2 tags
        assert report.total_items_after == 5
        assert report.processing_time > 0
    
    def test_data_loss_detection(self):
        """Test detection of significant data loss"""
        original_data = {
            'bookmarks': [
                {'id': i, 'name': f'Bookmark {i}', 'url': f'https://example.com/{i}'}
                for i in range(1, 11)  # 10 bookmarks
            ],
            'collections': [],
            'tags': []
        }
        
        enhanced_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ],  # Only 1 bookmark left (90% loss)
            'collections': [],
            'tags': []
        }
        
        contexts = []
        
        report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.BASIC
        )
        
        assert report.passed == False
        assert report.rollback_recommended == True
        assert len(report.integrity_issues) > 0
        
        # Should detect data loss
        data_loss_issues = [issue for issue in report.integrity_issues 
                           if issue.issue_type == IntegrityIssueType.DATA_LOSS]
        assert len(data_loss_issues) > 0
        assert data_loss_issues[0].severity == "critical"
    
    def test_missing_required_fields(self):
        """Test detection of missing required fields"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ],
            'collections': [],
            'tags': []
        }
        
        enhanced_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1'}  # Missing URL
            ],
            'collections': [],
            'tags': []
        }
        
        contexts = []
        
        report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.STANDARD
        )
        
        assert report.passed == False
        
        # Should detect missing required field
        missing_field_issues = [issue for issue in report.integrity_issues 
                               if issue.issue_type == IntegrityIssueType.MISSING_REQUIRED_FIELD]
        assert len(missing_field_issues) > 0
        assert 'url' in missing_field_issues[0].description.lower()
    
    def test_duplicate_id_detection(self):
        """Test detection of duplicate IDs"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'},
                {'id': 2, 'name': 'Bookmark 2', 'url': 'https://example.com/2'}
            ],
            'collections': [],
            'tags': []
        }
        
        enhanced_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'},
                {'id': 1, 'name': 'Duplicate Bookmark', 'url': 'https://example.com/duplicate'}  # Duplicate ID
            ],
            'collections': [],
            'tags': []
        }
        
        contexts = []
        
        report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.STRICT
        )
        
        assert report.passed == False
        
        # Should detect duplicate ID
        duplicate_issues = [issue for issue in report.integrity_issues 
                           if issue.issue_type == IntegrityIssueType.DUPLICATE_ID]
        assert len(duplicate_issues) > 0
        assert duplicate_issues[0].severity == "critical"
    
    def test_enhancement_context_validation(self):
        """Test validation of enhancement contexts"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ],
            'collections': [],
            'tags': []
        }
        
        enhanced_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Enhanced Bookmark 1', 'url': 'https://example.com/1'}
            ],
            'collections': [],
            'tags': []
        }
        
        # Create contexts with many critical errors
        contexts = []
        for i in range(5):
            context = EnhancementContext(
                bookmark_id=i,
                original_bookmark={'id': i, 'name': f'Bookmark {i}'},
                enhanced_bookmark={'id': i, 'name': f'Bookmark {i}'}
            )
            
            # Add critical error
            context.global_errors.append(ComponentError(
                component="test",
                error_type="CriticalError",
                message=f"Critical error {i}",
                severity=ErrorSeverity.CRITICAL,
                timestamp=context.start_time
            ))
            
            contexts.append(context)
        
        report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.STANDARD
        )
        
        # Should detect high error rate
        error_rate_issues = [issue for issue in report.integrity_issues 
                            if "critical errors" in issue.description.lower()]
        assert len(error_rate_issues) > 0
    
    def test_validation_levels(self):
        """Test different validation levels"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1', 'created_at': '2024-01-01'}
            ],
            'collections': [],
            'tags': []
        }
        
        enhanced_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Enhanced Bookmark 1', 'url': 'https://example.com/1'}  # Missing created_at
            ],
            'collections': [],
            'tags': []
        }
        
        contexts = []
        
        # Basic validation should pass (only checks critical issues)
        basic_report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.BASIC
        )
        
        # Comprehensive validation should detect metadata loss
        comprehensive_report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.COMPREHENSIVE
        )
        
        # Comprehensive should find more issues than basic
        assert len(comprehensive_report.integrity_issues) >= len(basic_report.integrity_issues)
    
    def test_change_statistics_calculation(self):
        """Test calculation of change statistics"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'},
                {'id': 2, 'name': 'Bookmark 2', 'url': 'https://example.com/2'},
                {'id': 3, 'name': 'Bookmark 3', 'url': 'https://example.com/3'}
            ],
            'collections': [],
            'tags': []
        }
        
        enhanced_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Enhanced Bookmark 1', 'url': 'https://example.com/1', 'description': 'Added'},  # Modified
                {'id': 2, 'name': 'Bookmark 2', 'url': 'https://example.com/2'},  # Unchanged
                {'id': 4, 'name': 'New Bookmark', 'url': 'https://example.com/4'}  # Added (id 3 removed)
            ],
            'collections': [],
            'tags': []
        }
        
        contexts = []
        
        report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.STANDARD
        )
        
        assert report.items_modified >= 1  # At least bookmark 1 was modified
        assert report.items_added >= 1     # Bookmark 4 was added
        assert report.items_removed >= 1   # Bookmark 3 was removed
    
    def test_rollback_plan_creation(self):
        """Test creation of rollback plans"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ],
            'collections': [],
            'tags': []
        }
        
        # Create a failed validation report
        from datetime import datetime
        report = ValidationReport(
            validation_id="test_validation",
            timestamp=datetime.now(),
            validation_level=ValidationLevel.STANDARD,
            original_data_hash="hash1",
            enhanced_data_hash="hash2",
            total_items_before=1,
            total_items_after=1,
            items_modified=0,
            items_added=0,
            items_removed=0,
            passed=False,
            rollback_recommended=True
        )
        
        # Add some integrity issues
        from enhancement.data_integrity_validator import IntegrityIssue
        report.integrity_issues.append(IntegrityIssue(
            issue_type=IntegrityIssueType.DATA_LOSS,
            severity="critical",
            description="Critical data loss detected",
            affected_items=[1]
        ))
        
        rollback_plan = self.validator.create_rollback_plan(original_data, report)
        
        assert rollback_plan.rollback_id.startswith("rollback_")
        assert rollback_plan.original_data == original_data
        assert len(rollback_plan.affected_items) > 0
        assert len(rollback_plan.rollback_steps) > 0
        assert rollback_plan.estimated_time > 0
        assert any(level in rollback_plan.risk_assessment for level in ["LOW", "MEDIUM", "HIGH"])
    
    def test_rollback_execution(self):
        """Test rollback plan execution"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ],
            'collections': [],
            'tags': []
        }
        
        rollback_plan = RollbackPlan(
            rollback_id="test_rollback",
            original_data=original_data,
            backup_path=None,
            affected_items=[1],
            rollback_steps=[
                "Step 1: Test step",
                "Step 2: Another test step"
            ],
            estimated_time=60.0,
            risk_assessment="LOW"
        )
        
        result = self.validator.execute_rollback(rollback_plan)
        
        assert result['rollback_id'] == "test_rollback"
        assert result['success'] == True
        assert len(result['steps_completed']) == 2
        assert len(result['errors']) == 0
        assert result['execution_time'] >= 0  # Should be non-negative
    
    def test_data_hash_calculation(self):
        """Test data hash calculation for integrity checking"""
        data1 = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ]
        }
        
        data2 = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ]
        }
        
        data3 = {
            'bookmarks': [
                {'id': 1, 'name': 'Different Bookmark', 'url': 'https://example.com/1'}
            ]
        }
        
        hash1 = self.validator._calculate_data_hash(data1)
        hash2 = self.validator._calculate_data_hash(data2)
        hash3 = self.validator._calculate_data_hash(data3)
        
        assert hash1 == hash2  # Same data should have same hash
        assert hash1 != hash3  # Different data should have different hash
        assert len(hash1) == 64  # SHA256 hash length
    
    def test_validation_statistics(self):
        """Test validation statistics retrieval"""
        stats = self.validator.get_validation_statistics()
        
        assert 'config' in stats
        assert 'default_validation_level' in stats
        assert 'enable_rollback' in stats
        assert 'max_data_loss_threshold' in stats
        assert 'required_fields' in stats
        assert 'rollback_plans_available' in stats
        
        assert stats['default_validation_level'] == 'standard'
        assert stats['enable_rollback'] == True
        assert stats['max_data_loss_threshold'] == 0.05
    
    def test_metadata_preservation_validation(self):
        """Test metadata preservation validation"""
        original_data = {
            'bookmarks': [
                {
                    'id': 1, 
                    'name': 'Bookmark 1', 
                    'url': 'https://example.com/1',
                    'created_at': '2024-01-01T00:00:00Z',
                    'updated_at': '2024-01-01T00:00:00Z',
                    'collection': {'id': 1, 'name': 'Collection 1'},
                    'tags': [{'id': 1, 'name': 'tag1'}]
                }
            ],
            'collections': [{'id': 1, 'name': 'Collection 1'}],
            'tags': [{'id': 1, 'name': 'tag1'}]
        }
        
        enhanced_data = {
            'bookmarks': [
                {
                    'id': 1, 
                    'name': 'Enhanced Bookmark 1', 
                    'url': 'https://example.com/1'
                    # Missing created_at, updated_at, collection, tags
                }
            ],
            'collections': [{'id': 1, 'name': 'Collection 1'}],
            'tags': [{'id': 1, 'name': 'tag1'}]
        }
        
        contexts = []
        
        report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.COMPREHENSIVE
        )
        
        # Should detect metadata loss
        metadata_issues = [issue for issue in report.integrity_issues 
                          if issue.issue_type == IntegrityIssueType.METADATA_INCONSISTENCY]
        assert len(metadata_issues) > 0
    
    def test_error_handling_in_validation(self):
        """Test error handling during validation"""
        # Test with malformed data that might cause validation errors
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ]
        }
        
        # Malformed enhanced data
        enhanced_data = {
            'bookmarks': "not a list"  # Should be a list
        }
        
        contexts = []
        
        # Should handle the error gracefully
        report = self.validator.validate_enhancement_results(
            original_data, enhanced_data, contexts, ValidationLevel.STANDARD
        )
        
        assert not report.passed
        assert len(report.integrity_issues) > 0
        
        # Should have data corruption issues
        corruption_issues = [issue for issue in report.integrity_issues 
                            if issue.issue_type == IntegrityIssueType.DATA_CORRUPTION]
        assert len(corruption_issues) > 0
    
    def test_backup_validation(self):
        """Test backup creation validation (Requirement 8.1)"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ],
            'collections': [],
            'tags': []
        }
        
        # Create a temporary backup file
        backup_file = self.temp_dir / "test_backup.json"
        with open(backup_file, 'w') as f:
            json.dump(original_data, f)
        
        # Test successful backup validation
        assert self.validator.validate_backup_creation(original_data, str(backup_file))
        
        # Test with non-existent backup file
        assert not self.validator.validate_backup_creation(original_data, str(self.temp_dir / "nonexistent.json"))
        
        # Test with empty backup file
        empty_backup = self.temp_dir / "empty_backup.json"
        empty_backup.touch()
        assert not self.validator.validate_backup_creation(original_data, str(empty_backup))
        
        # Test with invalid JSON backup file
        invalid_backup = self.temp_dir / "invalid_backup.json"
        with open(invalid_backup, 'w') as f:
            f.write("invalid json content")
        assert not self.validator.validate_backup_creation(original_data, str(invalid_backup))
    
    def test_enhancement_completion_validation(self):
        """Test enhancement completion validation (Requirement 8.3)"""
        original_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'},
                {'id': 2, 'name': 'Bookmark 2', 'url': 'https://example.com/2'},
                {'id': 3, 'name': 'Bookmark 3', 'url': 'https://example.com/3'}
            ],
            'collections': [],
            'tags': []
        }
        
        # Test successful enhancement completion
        enhanced_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Enhanced Bookmark 1', 'url': 'https://example.com/1', 'description': 'Added'},
                {'id': 2, 'name': 'Enhanced Bookmark 2', 'url': 'https://example.com/2', 'tags': [{'id': 1, 'name': 'tag1'}]},
                {'id': 3, 'name': 'Enhanced Bookmark 3', 'url': 'https://example.com/3'}
            ],
            'collections': [],
            'tags': [{'id': 1, 'name': 'tag1'}]
        }
        
        contexts = []
        
        report = self.validator.validate_enhancement_completion(original_data, enhanced_data, contexts)
        
        assert report.passed
        assert report.items_modified > 0  # Should have modifications
        
        # Test completion with no changes (should warn)
        unchanged_data = original_data.copy()
        
        report = self.validator.validate_enhancement_completion(original_data, unchanged_data, contexts)
        
        # Should have warning about no changes
        no_change_issues = [issue for issue in report.integrity_issues 
                           if "no items were modified" in issue.description.lower()]
        assert len(no_change_issues) > 0
        
        # Test completion with excessive data loss
        minimal_data = {
            'bookmarks': [
                {'id': 1, 'name': 'Bookmark 1', 'url': 'https://example.com/1'}
            ],  # Lost 2 out of 3 bookmarks (66% loss)
            'collections': [],
            'tags': []
        }
        
        report = self.validator.validate_enhancement_completion(original_data, minimal_data, contexts)
        
        # Should detect excessive data loss
        data_loss_issues = [issue for issue in report.integrity_issues 
                           if issue.issue_type == IntegrityIssueType.DATA_LOSS and "excessive" in issue.description.lower()]
        assert len(data_loss_issues) > 0


if __name__ == "__main__":
    pytest.main([__file__])