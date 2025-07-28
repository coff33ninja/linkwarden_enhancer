"""Data models for Linkwarden Enhancer"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class ChangeSet:
    """Track all changes made during processing"""
    bookmarks_added: List[dict] = field(default_factory=list)
    bookmarks_modified: List[Tuple[dict, dict]] = field(default_factory=list)  # (original, modified)
    bookmarks_deleted: List[dict] = field(default_factory=list)
    collections_added: List[dict] = field(default_factory=list)
    collections_modified: List[Tuple[dict, dict]] = field(default_factory=list)
    collections_deleted: List[dict] = field(default_factory=list)
    tags_added: List[dict] = field(default_factory=list)
    tags_modified: List[Tuple[dict, dict]] = field(default_factory=list)
    tags_deleted: List[dict] = field(default_factory=list)
    
    def get_deletion_percentage(self, total_items: int) -> float:
        """Calculate percentage of items deleted"""
        if total_items == 0:
            return 0.0
        deleted_count = len(self.bookmarks_deleted) + len(self.collections_deleted) + len(self.tags_deleted)
        return (deleted_count / total_items) * 100
    
    def get_total_changes(self) -> int:
        """Get total number of changes"""
        return (len(self.bookmarks_added) + len(self.bookmarks_modified) + len(self.bookmarks_deleted) +
                len(self.collections_added) + len(self.collections_modified) + len(self.collections_deleted) +
                len(self.tags_added) + len(self.tags_modified) + len(self.tags_deleted))


@dataclass
class AIAnalysisReport:
    """Report from AI analysis operations"""
    total_bookmarks_analyzed: int = 0
    ai_tags_suggested: int = 0
    duplicates_detected: int = 0
    clusters_created: int = 0
    topics_discovered: int = 0
    processing_time: float = 0.0
    model_accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    ollama_requests: int = 0
    ollama_response_time: float = 0.0


@dataclass
class EnhancementReport:
    """Report from enhancement operations"""
    bookmarks_enhanced: int = 0
    metadata_fields_added: int = 0
    scraping_failures: int = 0
    scrapers_used: Dict[str, int] = field(default_factory=dict)
    average_scraping_time: float = 0.0
    cache_hit_rate: float = 0.0
    ai_analysis_report: Optional[AIAnalysisReport] = None


@dataclass
class IntegrityResult:
    """Result from integrity checking"""
    bookmarks_verified: int = 0
    collections_verified: int = 0
    tags_verified: int = 0
    orphaned_references: List[str] = field(default_factory=list)
    integrity_issues: List[str] = field(default_factory=list)
    success: bool = True


@dataclass
class SafetyResult:
    """Result from safety operations"""
    success: bool
    changes_applied: ChangeSet
    backups_created: List[str]
    integrity_report: Optional[IntegrityResult]
    enhancement_report: Optional[EnhancementReport]
    execution_time: float
    warnings: List[str]
    errors: List[str]


@dataclass
class ValidationResult:
    """Result from validation operations"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_bookmarks: int = 0
    total_collections: int = 0
    total_tags: int = 0


@dataclass
class ImportResult:
    """Result from import operations"""
    bookmarks: List[dict] = field(default_factory=list)
    total_imported: int = 0
    collections_found: int = 0
    import_source: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class GitHubBookmark:
    """GitHub repository converted to bookmark format"""
    name: str
    url: str
    description: str
    tags: List[str]
    suggested_collection: str
    metadata: Dict[str, Any] = field(default_factory=dict)