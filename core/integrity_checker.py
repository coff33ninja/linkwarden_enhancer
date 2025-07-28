"""Integrity Checker - Comprehensive data integrity validation and comparison"""

import json
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

from ..utils.logging_utils import get_logger
from ..data_models import IntegrityResult

logger = get_logger(__name__)


@dataclass
class DataDiff:
    """Represents differences between two data sets"""
    added_items: List[Dict[str, Any]]
    removed_items: List[Dict[str, Any]]
    modified_items: List[Tuple[Dict[str, Any], Dict[str, Any]]]  # (before, after)
    unchanged_items: List[Dict[str, Any]]


@dataclass
class RelationshipIssue:
    """Represents a relationship integrity issue"""
    issue_type: str
    description: str
    affected_item_id: int
    affected_item_type: str  # 'bookmark', 'collection', 'tag'
    severity: str  # 'warning', 'error', 'critical'
    suggested_fix: Optional[str] = None


class IntegrityChecker:
    """Comprehensive data integrity validation and comparison system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize integrity checker with configuration"""
        self.config = config
        self.integrity_config = config.get('integrity', {})
        
        # Integrity check settings
        self.check_orphaned_refs = self.integrity_config.get('check_orphaned_refs', True)
        self.check_circular_refs = self.integrity_config.get('check_circular_refs', True)
        self.check_duplicate_urls = self.integrity_config.get('check_duplicate_urls', True)
        self.check_empty_collections = self.integrity_config.get('check_empty_collections', True)
        
        # Statistics
        self.integrity_stats = {
            'total_checks': 0,
            'issues_found': 0,
            'comparisons_made': 0
        }
        
        logger.info("Integrity checker initialized")
    
    def check_data_integrity(self, data: Dict[str, Any]) -> IntegrityResult:
        """Perform comprehensive integrity check on data"""
        
        try:
            self.integrity_stats['total_checks'] += 1
            
            bookmarks = data.get('bookmarks', [])
            collections = data.get('collections', [])
            tags = data.get('tags', [])
            
            integrity_issues = []
            orphaned_references = []
            
            logger.info(f"Starting integrity check: {len(bookmarks)} bookmarks, {len(collections)} collections, {len(tags)} tags")
            
            # Check URL preservation
            url_issues = self._check_url_preservation(bookmarks)
            integrity_issues.extend(url_issues)
            
            # Check collection relationships
            collection_issues, collection_orphans = self._check_collection_relationships(collections, bookmarks)
            integrity_issues.extend(collection_issues)
            orphaned_references.extend(collection_orphans)
            
            # Check tag relationships
            tag_issues, tag_orphans = self._check_tag_relationships(tags, bookmarks)
            integrity_issues.extend(tag_issues)
            orphaned_references.extend(tag_orphans)
            
            # Check for orphaned references
            if self.check_orphaned_refs:
                orphan_issues = self._check_orphaned_references(bookmarks, collections, tags)
                integrity_issues.extend(orphan_issues)
            
            # Check for circular references
            if self.check_circular_refs:
                circular_issues = self._check_circular_references(collections)
                integrity_issues.extend(circular_issues)
            
            # Check for duplicate URLs
            if self.check_duplicate_urls:
                duplicate_issues = self._check_duplicate_urls(bookmarks)
                integrity_issues.extend(duplicate_issues)
            
            # Check for empty collections
            if self.check_empty_collections:
                empty_collection_issues = self._check_empty_collections(collections, bookmarks)
                integrity_issues.extend(empty_collection_issues)
            
            # Update statistics
            self.integrity_stats['issues_found'] += len(integrity_issues)
            
            # Convert issues to string format
            issue_strings = [self._format_issue(issue) for issue in integrity_issues]
            
            result = IntegrityResult(
                bookmarks_verified=len(bookmarks),
                collections_verified=len(collections),
                tags_verified=len(tags),
                orphaned_references=orphaned_references,
                integrity_issues=issue_strings,
                success=len([issue for issue in integrity_issues if issue.severity == 'critical']) == 0
            )
            
            logger.info(f"Integrity check completed: {len(integrity_issues)} issues found")
            return result
            
        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return IntegrityResult(
                bookmarks_verified=0,
                collections_verified=0,
                tags_verified=0,
                orphaned_references=[],
                integrity_issues=[f"Integrity check error: {e}"],
                success=False
            )
    
    def compare_data_sets(self, 
                         before_data: Dict[str, Any], 
                         after_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two data sets and generate detailed diff report"""
        
        try:
            self.integrity_stats['comparisons_made'] += 1
            
            logger.info("Starting data set comparison")
            
            # Compare bookmarks
            bookmark_diff = self._compare_items(
                before_data.get('bookmarks', []),
                after_data.get('bookmarks', []),
                'id'
            )
            
            # Compare collections
            collection_diff = self._compare_items(
                before_data.get('collections', []),
                after_data.get('collections', []),
                'id'
            )
            
            # Compare tags
            tag_diff = self._compare_items(
                before_data.get('tags', []),
                after_data.get('tags', []),
                'id'
            )
            
            # Generate summary statistics
            summary = {
                'bookmarks': {
                    'added': len(bookmark_diff.added_items),
                    'removed': len(bookmark_diff.removed_items),
                    'modified': len(bookmark_diff.modified_items),
                    'unchanged': len(bookmark_diff.unchanged_items)
                },
                'collections': {
                    'added': len(collection_diff.added_items),
                    'removed': len(collection_diff.removed_items),
                    'modified': len(collection_diff.modified_items),
                    'unchanged': len(collection_diff.unchanged_items)
                },
                'tags': {
                    'added': len(tag_diff.added_items),
                    'removed': len(tag_diff.removed_items),
                    'modified': len(tag_diff.modified_items),
                    'unchanged': len(tag_diff.unchanged_items)
                }
            }
            
            # Calculate change percentages
            total_before = len(before_data.get('bookmarks', [])) + len(before_data.get('collections', [])) + len(before_data.get('tags', []))
            total_changes = (summary['bookmarks']['added'] + summary['bookmarks']['removed'] + summary['bookmarks']['modified'] +
                           summary['collections']['added'] + summary['collections']['removed'] + summary['collections']['modified'] +
                           summary['tags']['added'] + summary['tags']['removed'] + summary['tags']['modified'])
            
            change_percentage = (total_changes / total_before * 100) if total_before > 0 else 0
            
            comparison_result = {
                'summary': summary,
                'change_percentage': change_percentage,
                'total_changes': total_changes,
                'bookmark_diff': bookmark_diff,
                'collection_diff': collection_diff,
                'tag_diff': tag_diff,
                'comparison_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Data comparison completed: {change_percentage:.1f}% change rate")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Data comparison failed: {e}")
            return {'error': str(e)}
    
    def validate_before_after_consistency(self, 
                                        before_data: Dict[str, Any], 
                                        after_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that critical data is preserved between before and after states"""
        
        try:
            validation_results = {
                'url_preservation': True,
                'collection_structure': True,
                'tag_relationships': True,
                'issues': [],
                'warnings': []
            }
            
            # Check URL preservation
            before_urls = {bm.get('url') for bm in before_data.get('bookmarks', [])}
            after_urls = {bm.get('url') for bm in after_data.get('bookmarks', [])}
            
            missing_urls = before_urls - after_urls
            if missing_urls:
                validation_results['url_preservation'] = False
                validation_results['issues'].append(f"Missing URLs: {len(missing_urls)} bookmarks lost")
                for url in list(missing_urls)[:5]:  # Show first 5
                    validation_results['issues'].append(f"  Lost URL: {url}")
            
            # Check collection structure preservation
            before_collections = {col.get('id'): col.get('name') for col in before_data.get('collections', [])}
            after_collections = {col.get('id'): col.get('name') for col in after_data.get('collections', [])}
            
            missing_collections = set(before_collections.keys()) - set(after_collections.keys())
            if missing_collections:
                validation_results['collection_structure'] = False
                validation_results['issues'].append(f"Missing collections: {len(missing_collections)} collections lost")
            
            # Check tag preservation
            before_tags = {tag.get('id'): tag.get('name') for tag in before_data.get('tags', [])}
            after_tags = {tag.get('id'): tag.get('name') for tag in after_data.get('tags', [])}
            
            missing_tags = set(before_tags.keys()) - set(after_tags.keys())
            if missing_tags:
                validation_results['tag_relationships'] = False
                validation_results['warnings'].append(f"Missing tags: {len(missing_tags)} tags removed")
            
            # Check bookmark count changes
            before_count = len(before_data.get('bookmarks', []))
            after_count = len(after_data.get('bookmarks', []))
            
            if after_count < before_count:
                reduction_percentage = ((before_count - after_count) / before_count) * 100
                if reduction_percentage > 10:  # More than 10% reduction
                    validation_results['warnings'].append(f"Significant bookmark reduction: {reduction_percentage:.1f}%")
            
            validation_results['success'] = all([
                validation_results['url_preservation'],
                validation_results['collection_structure'],
                validation_results['tag_relationships']
            ])
            
            logger.info(f"Before/after validation: {'PASSED' if validation_results['success'] else 'FAILED'}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Before/after validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _check_url_preservation(self, bookmarks: List[Dict[str, Any]]) -> List[RelationshipIssue]:
        """Check that all bookmarks have valid URLs"""
        
        issues = []
        
        for bookmark in bookmarks:
            bookmark_id = bookmark.get('id')
            url = bookmark.get('url', '').strip()
            
            if not url:
                issues.append(RelationshipIssue(
                    issue_type="missing_url",
                    description=f"Bookmark {bookmark_id} has no URL",
                    affected_item_id=bookmark_id,
                    affected_item_type="bookmark",
                    severity="critical",
                    suggested_fix="Remove bookmark or add valid URL"
                ))
            elif not self._is_valid_url_format(url):
                issues.append(RelationshipIssue(
                    issue_type="invalid_url",
                    description=f"Bookmark {bookmark_id} has invalid URL format: {url}",
                    affected_item_id=bookmark_id,
                    affected_item_type="bookmark",
                    severity="warning",
                    suggested_fix="Correct URL format"
                ))
        
        return issues
    
    def _check_collection_relationships(self, 
                                      collections: List[Dict[str, Any]], 
                                      bookmarks: List[Dict[str, Any]]) -> Tuple[List[RelationshipIssue], List[str]]:
        """Check collection relationship integrity"""
        
        issues = []
        orphaned_refs = []
        
        collection_ids = {col['id'] for col in collections}
        
        # Check parent-child relationships
        for collection in collections:
            collection_id = collection.get('id')
            parent_id = collection.get('parent_id')
            
            if parent_id and parent_id not in collection_ids:
                issues.append(RelationshipIssue(
                    issue_type="orphaned_parent",
                    description=f"Collection {collection_id} references non-existent parent {parent_id}",
                    affected_item_id=collection_id,
                    affected_item_type="collection",
                    severity="error",
                    suggested_fix="Remove parent reference or create parent collection"
                ))
                orphaned_refs.append(f"collection:{collection_id}->parent:{parent_id}")
        
        # Check bookmark-collection relationships
        for bookmark in bookmarks:
            bookmark_id = bookmark.get('id')
            collection = bookmark.get('collection')
            
            if collection:
                collection_id = collection.get('id')
                if collection_id and collection_id not in collection_ids:
                    issues.append(RelationshipIssue(
                        issue_type="orphaned_collection",
                        description=f"Bookmark {bookmark_id} references non-existent collection {collection_id}",
                        affected_item_id=bookmark_id,
                        affected_item_type="bookmark",
                        severity="error",
                        suggested_fix="Remove collection reference or create collection"
                    ))
                    orphaned_refs.append(f"bookmark:{bookmark_id}->collection:{collection_id}")
        
        return issues, orphaned_refs
    
    def _check_tag_relationships(self, 
                               tags: List[Dict[str, Any]], 
                               bookmarks: List[Dict[str, Any]]) -> Tuple[List[RelationshipIssue], List[str]]:
        """Check tag relationship integrity"""
        
        issues = []
        orphaned_refs = []
        
        tag_ids = {tag['id'] for tag in tags}
        
        # Check bookmark-tag relationships
        for bookmark in bookmarks:
            bookmark_id = bookmark.get('id')
            bookmark_tags = bookmark.get('tags', [])
            
            for tag in bookmark_tags:
                tag_id = tag.get('id')
                if tag_id and tag_id not in tag_ids:
                    issues.append(RelationshipIssue(
                        issue_type="orphaned_tag",
                        description=f"Bookmark {bookmark_id} references non-existent tag {tag_id}",
                        affected_item_id=bookmark_id,
                        affected_item_type="bookmark",
                        severity="warning",
                        suggested_fix="Remove tag reference or create tag"
                    ))
                    orphaned_refs.append(f"bookmark:{bookmark_id}->tag:{tag_id}")
        
        return issues, orphaned_refs
    
    def _check_orphaned_references(self, 
                                 bookmarks: List[Dict[str, Any]], 
                                 collections: List[Dict[str, Any]], 
                                 tags: List[Dict[str, Any]]) -> List[RelationshipIssue]:
        """Check for orphaned references across all data types"""
        
        issues = []
        
        # Find unused collections
        collection_ids = {col['id'] for col in collections}
        used_collection_ids = set()
        
        for bookmark in bookmarks:
            collection = bookmark.get('collection')
            if collection and collection.get('id'):
                used_collection_ids.add(collection['id'])
        
        # Add parent collections as used
        for collection in collections:
            parent_id = collection.get('parent_id')
            if parent_id:
                used_collection_ids.add(parent_id)
        
        unused_collections = collection_ids - used_collection_ids
        for collection_id in unused_collections:
            issues.append(RelationshipIssue(
                issue_type="unused_collection",
                description=f"Collection {collection_id} is not used by any bookmarks",
                affected_item_id=collection_id,
                affected_item_type="collection",
                severity="warning",
                suggested_fix="Remove unused collection or assign bookmarks to it"
            ))
        
        # Find unused tags
        tag_ids = {tag['id'] for tag in tags}
        used_tag_ids = set()
        
        for bookmark in bookmarks:
            bookmark_tags = bookmark.get('tags', [])
            for tag in bookmark_tags:
                if tag.get('id'):
                    used_tag_ids.add(tag['id'])
        
        unused_tags = tag_ids - used_tag_ids
        for tag_id in unused_tags:
            issues.append(RelationshipIssue(
                issue_type="unused_tag",
                description=f"Tag {tag_id} is not used by any bookmarks",
                affected_item_id=tag_id,
                affected_item_type="tag",
                severity="warning",
                suggested_fix="Remove unused tag or assign to bookmarks"
            ))
        
        return issues
    
    def _check_circular_references(self, collections: List[Dict[str, Any]]) -> List[RelationshipIssue]:
        """Check for circular references in collection hierarchy"""
        
        issues = []
        
        # Build parent-child mapping
        parent_map = {col['id']: col.get('parent_id') for col in collections}
        
        for collection in collections:
            collection_id = collection['id']
            visited = set()
            current_id = collection_id
            
            # Follow parent chain
            while current_id and current_id in parent_map:
                if current_id in visited:
                    issues.append(RelationshipIssue(
                        issue_type="circular_reference",
                        description=f"Collection {collection_id} has circular parent reference",
                        affected_item_id=collection_id,
                        affected_item_type="collection",
                        severity="critical",
                        suggested_fix="Break circular reference by removing parent relationship"
                    ))
                    break
                
                visited.add(current_id)
                current_id = parent_map.get(current_id)
        
        return issues
    
    def _check_duplicate_urls(self, bookmarks: List[Dict[str, Any]]) -> List[RelationshipIssue]:
        """Check for duplicate URLs in bookmarks"""
        
        issues = []
        url_counts = Counter()
        url_bookmarks = defaultdict(list)
        
        for bookmark in bookmarks:
            url = bookmark.get('url', '').strip().lower()
            if url:
                url_counts[url] += 1
                url_bookmarks[url].append(bookmark.get('id'))
        
        for url, count in url_counts.items():
            if count > 1:
                bookmark_ids = url_bookmarks[url]
                issues.append(RelationshipIssue(
                    issue_type="duplicate_url",
                    description=f"URL '{url}' is duplicated in bookmarks: {bookmark_ids}",
                    affected_item_id=bookmark_ids[0],  # Use first bookmark ID
                    affected_item_type="bookmark",
                    severity="warning",
                    suggested_fix="Merge duplicate bookmarks or remove duplicates"
                ))
        
        return issues
    
    def _check_empty_collections(self, 
                               collections: List[Dict[str, Any]], 
                               bookmarks: List[Dict[str, Any]]) -> List[RelationshipIssue]:
        """Check for empty collections"""
        
        issues = []
        
        # Count bookmarks per collection
        collection_bookmark_counts = Counter()
        for bookmark in bookmarks:
            collection = bookmark.get('collection')
            if collection and collection.get('id'):
                collection_bookmark_counts[collection['id']] += 1
        
        # Check for empty collections
        for collection in collections:
            collection_id = collection['id']
            if collection_bookmark_counts[collection_id] == 0:
                # Check if it has child collections
                has_children = any(col.get('parent_id') == collection_id for col in collections)
                
                if not has_children:
                    issues.append(RelationshipIssue(
                        issue_type="empty_collection",
                        description=f"Collection {collection_id} '{collection.get('name', '')}' is empty",
                        affected_item_id=collection_id,
                        affected_item_type="collection",
                        severity="warning",
                        suggested_fix="Add bookmarks to collection or remove empty collection"
                    ))
        
        return issues
    
    def _compare_items(self, 
                      before_items: List[Dict[str, Any]], 
                      after_items: List[Dict[str, Any]], 
                      id_field: str) -> DataDiff:
        """Compare two lists of items and return differences"""
        
        # Create ID-based lookups
        before_lookup = {item[id_field]: item for item in before_items}
        after_lookup = {item[id_field]: item for item in after_items}
        
        before_ids = set(before_lookup.keys())
        after_ids = set(after_lookup.keys())
        
        # Find added, removed, and potentially modified items
        added_ids = after_ids - before_ids
        removed_ids = before_ids - after_ids
        common_ids = before_ids & after_ids
        
        added_items = [after_lookup[item_id] for item_id in added_ids]
        removed_items = [before_lookup[item_id] for item_id in removed_ids]
        
        modified_items = []
        unchanged_items = []
        
        for item_id in common_ids:
            before_item = before_lookup[item_id]
            after_item = after_lookup[item_id]
            
            if self._items_are_different(before_item, after_item):
                modified_items.append((before_item, after_item))
            else:
                unchanged_items.append(after_item)
        
        return DataDiff(
            added_items=added_items,
            removed_items=removed_items,
            modified_items=modified_items,
            unchanged_items=unchanged_items
        )
    
    def _items_are_different(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """Check if two items are different (excluding timestamps)"""
        
        # Fields to ignore in comparison
        ignore_fields = {'updated_at', 'created_at'}
        
        # Create copies without ignored fields
        item1_filtered = {k: v for k, v in item1.items() if k not in ignore_fields}
        item2_filtered = {k: v for k, v in item2.items() if k not in ignore_fields}
        
        return item1_filtered != item2_filtered
    
    def _is_valid_url_format(self, url: str) -> bool:
        """Basic URL format validation"""
        
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _format_issue(self, issue: RelationshipIssue) -> str:
        """Format relationship issue as string"""
        
        severity_prefix = {
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }.get(issue.severity, '❓')
        
        message = f"{severity_prefix} {issue.description}"
        if issue.suggested_fix:
            message += f" (Suggested fix: {issue.suggested_fix})"
        
        return message
    
    def get_integrity_statistics(self) -> Dict[str, Any]:
        """Get integrity checker statistics"""
        
        return {
            'integrity_stats': self.integrity_stats.copy(),
            'check_settings': {
                'check_orphaned_refs': self.check_orphaned_refs,
                'check_circular_refs': self.check_circular_refs,
                'check_duplicate_urls': self.check_duplicate_urls,
                'check_empty_collections': self.check_empty_collections
            },
            'config': self.integrity_config
        }