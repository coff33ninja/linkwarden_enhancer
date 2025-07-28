"""Advanced duplicate detection and resolution system"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

from ai.similarity_engine import DuplicateGroup
from enhancement.similarity_calculator import SimilarityCalculator, ComprehensiveSimilarityScore
from enhancement.url_normalizer import URLNormalizer
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResolutionStrategy(Enum):
    """Duplicate resolution strategies"""
    MERGE = "merge"
    USER_CHOICE = "user_choice"
    QUALITY_BASED = "quality_based"
    RECENCY_BASED = "recency_based"
    KEEP_FIRST = "keep_first"
    KEEP_ALL = "keep_all"


@dataclass
class DuplicateResolution:
    """Resolution for a duplicate group"""
    group_id: int
    strategy: ResolutionStrategy
    action: str  # "merge", "keep", "delete", "user_review"
    primary_bookmark_id: int
    secondary_bookmark_ids: List[int]
    merged_data: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    reason: str = ""


@dataclass
class QualityScore:
    """Quality score for a bookmark"""
    bookmark_id: int
    title_quality: float
    description_quality: float
    url_quality: float
    metadata_completeness: float
    overall_quality: float
    factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractiveChoice:
    """Interactive choice for ambiguous duplicates"""
    group_id: int
    bookmarks: List[Dict[str, Any]]
    similarity_scores: List[float]
    recommended_action: str
    options: List[str]
    user_choice: Optional[str] = None


class DuplicateDetector:
    """Advanced duplicate detection and resolution system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize duplicate detector"""
        self.config = config
        self.duplicate_config = config.get('duplicate_detection', {})
        
        # Initialize components
        self.similarity_calculator = SimilarityCalculator(config)
        self.url_normalizer = URLNormalizer()
        
        # Detection thresholds
        self.exact_duplicate_threshold = self.duplicate_config.get('exact_threshold', 0.95)
        self.near_duplicate_threshold = self.duplicate_config.get('near_threshold', 0.85)
        self.similar_threshold = self.duplicate_config.get('similar_threshold', 0.7)
        
        # Resolution settings
        self.default_strategy = ResolutionStrategy(
            self.duplicate_config.get('default_strategy', 'quality_based')
        )
        self.auto_resolve_threshold = self.duplicate_config.get('auto_resolve_threshold', 0.9)
        self.require_user_confirmation = self.duplicate_config.get('require_confirmation', True)
        
        logger.info("Duplicate detector initialized")
    
    def detect_all_duplicates(self, bookmarks: List[Dict[str, Any]]) -> List[DuplicateGroup]:
        """Detect all types of duplicates in bookmark collection"""
        try:
            logger.info(f"Detecting duplicates in {len(bookmarks)} bookmarks")
            
            # Use the enhanced similarity calculator for detection
            duplicate_groups = self.similarity_calculator.detect_duplicates_enhanced(bookmarks)
            
            # Enhance groups with additional metadata
            enhanced_groups = []
            for group in duplicate_groups:
                enhanced_group = self._enhance_duplicate_group(group, bookmarks)
                enhanced_groups.append(enhanced_group)
            
            logger.info(f"Found {len(enhanced_groups)} duplicate groups")
            return enhanced_groups
            
        except Exception as e:
            logger.error(f"Error detecting duplicates: {e}")
            return []
    
    def detect_url_duplicates(self, bookmarks: List[Dict[str, Any]]) -> List[DuplicateGroup]:
        """Detect duplicates based primarily on URL similarity"""
        try:
            logger.info("Detecting URL-based duplicates")
            
            # Group bookmarks by normalized URLs
            url_groups = {}
            
            for bookmark in bookmarks:
                url = bookmark.get('url', '')
                if not url:
                    continue
                
                normalized = self.url_normalizer.normalize_url(url)
                normalized_url = normalized.normalized_url
                
                if normalized_url not in url_groups:
                    url_groups[normalized_url] = []
                url_groups[normalized_url].append(bookmark)
            
            # Create duplicate groups for URLs with multiple bookmarks
            duplicate_groups = []
            group_id = 0
            
            for normalized_url, group_bookmarks in url_groups.items():
                if len(group_bookmarks) > 1:
                    bookmark_ids = [b.get('id') for b in group_bookmarks if b.get('id') is not None]
                    
                    if len(bookmark_ids) > 1:
                        # Calculate similarity scores within group
                        similarity_scores = []
                        for i in range(len(group_bookmarks)):
                            for j in range(i + 1, len(group_bookmarks)):
                                score = self.similarity_calculator.calculate_comprehensive_similarity(
                                    group_bookmarks[i], group_bookmarks[j]
                                )
                                similarity_scores.append(score.overall_similarity)
                        
                        avg_similarity = np.mean(similarity_scores) if similarity_scores else 1.0
                        
                        group = DuplicateGroup(
                            group_id=group_id,
                            bookmarks=bookmark_ids,
                            similarity_scores=[avg_similarity] * len(bookmark_ids),
                            group_type="url_duplicate",
                            representative_bookmark=bookmark_ids[0]
                        )
                        
                        duplicate_groups.append(group)
                        group_id += 1
            
            logger.info(f"Found {len(duplicate_groups)} URL duplicate groups")
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Error detecting URL duplicates: {e}")
            return []
    
    def resolve_duplicates(self, 
                          duplicate_groups: List[DuplicateGroup], 
                          bookmarks: List[Dict[str, Any]],
                          strategy: Optional[ResolutionStrategy] = None) -> List[DuplicateResolution]:
        """Resolve duplicate groups using specified strategy"""
        try:
            if strategy is None:
                strategy = self.default_strategy
            
            logger.info(f"Resolving {len(duplicate_groups)} duplicate groups using {strategy.value}")
            
            # Create bookmark lookup
            bookmark_lookup = {b.get('id'): b for b in bookmarks if b.get('id') is not None}
            
            resolutions = []
            
            for group in duplicate_groups:
                try:
                    resolution = self._resolve_group(group, bookmark_lookup, strategy)
                    if resolution:
                        resolutions.append(resolution)
                except Exception as e:
                    logger.error(f"Error resolving group {group.group_id}: {e}")
                    continue
            
            logger.info(f"Generated {len(resolutions)} resolutions")
            return resolutions
            
        except Exception as e:
            logger.error(f"Error resolving duplicates: {e}")
            return []
    
    def get_interactive_choices(self, 
                               duplicate_groups: List[DuplicateGroup], 
                               bookmarks: List[Dict[str, Any]]) -> List[InteractiveChoice]:
        """Get interactive choices for ambiguous duplicate groups"""
        try:
            bookmark_lookup = {b.get('id'): b for b in bookmarks if b.get('id') is not None}
            interactive_choices = []
            
            for group in duplicate_groups:
                # Only create interactive choices for ambiguous cases
                avg_similarity = np.mean(group.similarity_scores) if group.similarity_scores else 0.0
                
                if avg_similarity < self.auto_resolve_threshold:
                    group_bookmarks = [bookmark_lookup[bid] for bid in group.bookmarks 
                                     if bid in bookmark_lookup]
                    
                    if len(group_bookmarks) > 1:
                        # Determine recommended action
                        recommended_action = self._get_recommended_action(group, group_bookmarks)
                        
                        # Generate options
                        options = [
                            "merge_all",
                            "keep_best_quality",
                            "keep_most_recent",
                            "keep_all_separate",
                            "manual_review"
                        ]
                        
                        choice = InteractiveChoice(
                            group_id=group.group_id,
                            bookmarks=group_bookmarks,
                            similarity_scores=group.similarity_scores,
                            recommended_action=recommended_action,
                            options=options
                        )
                        
                        interactive_choices.append(choice)
            
            return interactive_choices
            
        except Exception as e:
            logger.error(f"Error generating interactive choices: {e}")
            return []
    
    def calculate_quality_scores(self, bookmarks: List[Dict[str, Any]]) -> List[QualityScore]:
        """Calculate quality scores for bookmarks"""
        try:
            quality_scores = []
            
            for bookmark in bookmarks:
                bookmark_id = bookmark.get('id')
                if bookmark_id is None:
                    continue
                
                # Calculate component quality scores
                title_quality = self._assess_title_quality(bookmark)
                description_quality = self._assess_description_quality(bookmark)
                url_quality = self._assess_url_quality(bookmark)
                metadata_completeness = self._assess_metadata_completeness(bookmark)
                
                # Calculate overall quality
                overall_quality = (
                    title_quality * 0.3 +
                    description_quality * 0.25 +
                    url_quality * 0.2 +
                    metadata_completeness * 0.25
                )
                
                quality_score = QualityScore(
                    bookmark_id=bookmark_id,
                    title_quality=title_quality,
                    description_quality=description_quality,
                    url_quality=url_quality,
                    metadata_completeness=metadata_completeness,
                    overall_quality=overall_quality,
                    factors={
                        'title_length': len(bookmark.get('name', '') or ''),
                        'has_description': bool(bookmark.get('description', '')),
                        'has_tags': bool(bookmark.get('tags', [])),
                        'url_valid': self._is_valid_url(bookmark.get('url', ''))
                    }
                )
                
                quality_scores.append(quality_score)
            
            return quality_scores
            
        except Exception as e:
            logger.error(f"Error calculating quality scores: {e}")
            return []
    
    def _enhance_duplicate_group(self, group: DuplicateGroup, bookmarks: List[Dict[str, Any]]) -> DuplicateGroup:
        """Enhance duplicate group with additional metadata"""
        try:
            bookmark_lookup = {b.get('id'): b for b in bookmarks if b.get('id') is not None}
            
            # Calculate more detailed similarity scores
            group_bookmarks = [bookmark_lookup[bid] for bid in group.bookmarks if bid in bookmark_lookup]
            
            if len(group_bookmarks) > 1:
                # Recalculate similarity scores with comprehensive method
                detailed_scores = []
                for i in range(len(group_bookmarks)):
                    for j in range(i + 1, len(group_bookmarks)):
                        score = self.similarity_calculator.calculate_comprehensive_similarity(
                            group_bookmarks[i], group_bookmarks[j]
                        )
                        detailed_scores.append(score.overall_similarity)
                
                if detailed_scores:
                    group.similarity_scores = detailed_scores
                
                # Choose better representative based on quality
                quality_scores = self.calculate_quality_scores(group_bookmarks)
                if quality_scores:
                    best_quality = max(quality_scores, key=lambda x: x.overall_quality)
                    group.representative_bookmark = best_quality.bookmark_id
            
            return group
            
        except Exception as e:
            logger.warning(f"Error enhancing duplicate group: {e}")
            return group
    
    def _resolve_group(self, 
                      group: DuplicateGroup, 
                      bookmark_lookup: Dict[int, Dict[str, Any]], 
                      strategy: ResolutionStrategy) -> Optional[DuplicateResolution]:
        """Resolve a single duplicate group"""
        try:
            group_bookmarks = [bookmark_lookup[bid] for bid in group.bookmarks if bid in bookmark_lookup]
            
            if len(group_bookmarks) < 2:
                return None
            
            avg_similarity = np.mean(group.similarity_scores) if group.similarity_scores else 0.0
            
            if strategy == ResolutionStrategy.MERGE:
                return self._resolve_merge(group, group_bookmarks, avg_similarity)
            elif strategy == ResolutionStrategy.QUALITY_BASED:
                return self._resolve_quality_based(group, group_bookmarks, avg_similarity)
            elif strategy == ResolutionStrategy.RECENCY_BASED:
                return self._resolve_recency_based(group, group_bookmarks, avg_similarity)
            elif strategy == ResolutionStrategy.KEEP_FIRST:
                return self._resolve_keep_first(group, group_bookmarks, avg_similarity)
            elif strategy == ResolutionStrategy.USER_CHOICE:
                return self._resolve_user_choice(group, group_bookmarks, avg_similarity)
            else:
                return self._resolve_keep_all(group, group_bookmarks, avg_similarity)
            
        except Exception as e:
            logger.error(f"Error resolving group {group.group_id}: {e}")
            return None
    
    def _resolve_merge(self, 
                      group: DuplicateGroup, 
                      bookmarks: List[Dict[str, Any]], 
                      similarity: float) -> DuplicateResolution:
        """Resolve by merging all bookmarks in group"""
        # Choose primary bookmark (best quality or first)
        quality_scores = self.calculate_quality_scores(bookmarks)
        if quality_scores:
            primary = max(quality_scores, key=lambda x: x.overall_quality)
            primary_id = primary.bookmark_id
        else:
            primary_id = bookmarks[0].get('id')
        
        secondary_ids = [b.get('id') for b in bookmarks if b.get('id') != primary_id]
        
        # Create merged data
        merged_data = self._merge_bookmark_data(bookmarks, primary_id)
        
        return DuplicateResolution(
            group_id=group.group_id,
            strategy=ResolutionStrategy.MERGE,
            action="merge",
            primary_bookmark_id=primary_id,
            secondary_bookmark_ids=secondary_ids,
            merged_data=merged_data,
            confidence=similarity,
            reason=f"Merged {len(bookmarks)} similar bookmarks (similarity: {similarity:.2f})"
        )
    
    def _resolve_quality_based(self, 
                              group: DuplicateGroup, 
                              bookmarks: List[Dict[str, Any]], 
                              similarity: float) -> DuplicateResolution:
        """Resolve by keeping highest quality bookmark"""
        quality_scores = self.calculate_quality_scores(bookmarks)
        
        if not quality_scores:
            # Fallback to first bookmark
            primary_id = bookmarks[0].get('id')
        else:
            best_quality = max(quality_scores, key=lambda x: x.overall_quality)
            primary_id = best_quality.bookmark_id
        
        secondary_ids = [b.get('id') for b in bookmarks if b.get('id') != primary_id]
        
        return DuplicateResolution(
            group_id=group.group_id,
            strategy=ResolutionStrategy.QUALITY_BASED,
            action="keep",
            primary_bookmark_id=primary_id,
            secondary_bookmark_ids=secondary_ids,
            confidence=similarity,
            reason=f"Kept highest quality bookmark from {len(bookmarks)} duplicates"
        )
    
    def _resolve_recency_based(self, 
                              group: DuplicateGroup, 
                              bookmarks: List[Dict[str, Any]], 
                              similarity: float) -> DuplicateResolution:
        """Resolve by keeping most recent bookmark"""
        # Sort by creation/update time
        def get_timestamp(bookmark):
            created = bookmark.get('created_at') or bookmark.get('createdAt')
            updated = bookmark.get('updated_at') or bookmark.get('updatedAt')
            
            # Use updated time if available, otherwise created time
            timestamp = updated or created
            
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        # Try to parse ISO format
                        from datetime import datetime
                        return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    return timestamp
                except:
                    pass
            
            return datetime.min
        
        sorted_bookmarks = sorted(bookmarks, key=get_timestamp, reverse=True)
        primary_id = sorted_bookmarks[0].get('id')
        secondary_ids = [b.get('id') for b in sorted_bookmarks[1:]]
        
        return DuplicateResolution(
            group_id=group.group_id,
            strategy=ResolutionStrategy.RECENCY_BASED,
            action="keep",
            primary_bookmark_id=primary_id,
            secondary_bookmark_ids=secondary_ids,
            confidence=similarity,
            reason=f"Kept most recent bookmark from {len(bookmarks)} duplicates"
        )
    
    def _resolve_keep_first(self, 
                           group: DuplicateGroup, 
                           bookmarks: List[Dict[str, Any]], 
                           similarity: float) -> DuplicateResolution:
        """Resolve by keeping first bookmark"""
        primary_id = bookmarks[0].get('id')
        secondary_ids = [b.get('id') for b in bookmarks[1:]]
        
        return DuplicateResolution(
            group_id=group.group_id,
            strategy=ResolutionStrategy.KEEP_FIRST,
            action="keep",
            primary_bookmark_id=primary_id,
            secondary_bookmark_ids=secondary_ids,
            confidence=similarity,
            reason=f"Kept first bookmark from {len(bookmarks)} duplicates"
        )
    
    def _resolve_user_choice(self, 
                            group: DuplicateGroup, 
                            bookmarks: List[Dict[str, Any]], 
                            similarity: float) -> DuplicateResolution:
        """Resolve by requiring user choice"""
        return DuplicateResolution(
            group_id=group.group_id,
            strategy=ResolutionStrategy.USER_CHOICE,
            action="user_review",
            primary_bookmark_id=bookmarks[0].get('id'),
            secondary_bookmark_ids=[b.get('id') for b in bookmarks[1:]],
            confidence=similarity,
            reason=f"Requires user review for {len(bookmarks)} similar bookmarks"
        )
    
    def _resolve_keep_all(self, 
                         group: DuplicateGroup, 
                         bookmarks: List[Dict[str, Any]], 
                         similarity: float) -> DuplicateResolution:
        """Resolve by keeping all bookmarks separate"""
        return DuplicateResolution(
            group_id=group.group_id,
            strategy=ResolutionStrategy.KEEP_ALL,
            action="keep_all",
            primary_bookmark_id=bookmarks[0].get('id'),
            secondary_bookmark_ids=[b.get('id') for b in bookmarks[1:]],
            confidence=similarity,
            reason=f"Keeping all {len(bookmarks)} bookmarks separate"
        )
    
    def _merge_bookmark_data(self, bookmarks: List[Dict[str, Any]], primary_id: int) -> Dict[str, Any]:
        """Merge data from multiple bookmarks"""
        try:
            # Find primary bookmark
            primary = next((b for b in bookmarks if b.get('id') == primary_id), bookmarks[0])
            merged = primary.copy()
            
            # Merge tags from all bookmarks
            all_tags = set()
            for bookmark in bookmarks:
                tags = bookmark.get('tags', [])
                if isinstance(tags, list):
                    all_tags.update(tags)
                elif isinstance(tags, str):
                    all_tags.add(tags)
            
            merged['tags'] = list(all_tags)
            
            # Use best description (longest non-empty)
            descriptions = [b.get('description', '') for b in bookmarks if b.get('description', '')]
            if descriptions:
                merged['description'] = max(descriptions, key=len)
            
            # Use best title (longest non-generic)
            titles = [b.get('name', '') or b.get('title', '') for b in bookmarks 
                     if b.get('name', '') or b.get('title', '')]
            if titles:
                # Prefer non-generic titles
                non_generic = [t for t in titles if not self._is_generic_title(t)]
                if non_generic:
                    merged['name'] = max(non_generic, key=len)
                else:
                    merged['name'] = max(titles, key=len)
            
            # Merge collections (if multiple)
            collections = [b.get('collection') for b in bookmarks if b.get('collection')]
            if collections:
                merged['collection'] = collections[0]  # Use first non-empty collection
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging bookmark data: {e}")
            return bookmarks[0].copy() if bookmarks else {}
    
    def _get_recommended_action(self, group: DuplicateGroup, bookmarks: List[Dict[str, Any]]) -> str:
        """Get recommended action for a duplicate group"""
        avg_similarity = np.mean(group.similarity_scores) if group.similarity_scores else 0.0
        
        if avg_similarity > 0.9:
            return "merge_all"
        elif avg_similarity > 0.8:
            return "keep_best_quality"
        elif avg_similarity > 0.7:
            return "keep_most_recent"
        else:
            return "manual_review"
    
    def _assess_title_quality(self, bookmark: Dict[str, Any]) -> float:
        """Assess title quality (0.0 to 1.0)"""
        title = bookmark.get('name', '') or bookmark.get('title', '')
        
        if not title:
            return 0.0
        
        score = 0.0
        
        # Length score
        if 10 <= len(title) <= 100:
            score += 0.3
        elif len(title) > 5:
            score += 0.1
        
        # Non-generic score (but only if title is reasonable length)
        if not self._is_generic_title(title) and len(title) > 2:
            score += 0.4
        
        # Descriptiveness score
        words = title.split()
        if len(words) >= 2:
            score += 0.3
        
        return min(1.0, score)
    
    def _assess_description_quality(self, bookmark: Dict[str, Any]) -> float:
        """Assess description quality (0.0 to 1.0)"""
        description = bookmark.get('description', '')
        
        if not description:
            return 0.0
        
        score = 0.0
        
        # Length score
        if 50 <= len(description) <= 500:
            score += 0.5
        elif len(description) > 10:
            score += 0.2
        
        # Content quality
        words = description.split()
        if len(words) >= 5:
            score += 0.3
        
        # Not just URL
        if not description.startswith('http'):
            score += 0.2
        
        return min(1.0, score)
    
    def _assess_url_quality(self, bookmark: Dict[str, Any]) -> float:
        """Assess URL quality (0.0 to 1.0)"""
        url = bookmark.get('url', '')
        
        if not url:
            return 0.0
        
        score = 0.0
        
        # Valid URL
        if self._is_valid_url(url):
            score += 0.5
        
        # HTTPS
        if url.startswith('https://'):
            score += 0.2
        
        # Not too long
        if len(url) < 200:
            score += 0.2
        
        # Meaningful path
        if '/' in url[8:] and not url.endswith('/'):
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_metadata_completeness(self, bookmark: Dict[str, Any]) -> float:
        """Assess metadata completeness (0.0 to 1.0)"""
        score = 0.0
        
        # Has tags
        tags = bookmark.get('tags', [])
        if tags and len(tags) > 0:
            score += 0.3
        
        # Has collection
        if bookmark.get('collection'):
            score += 0.2
        
        # Has timestamps
        if bookmark.get('created_at') or bookmark.get('createdAt'):
            score += 0.2
        
        # Has content
        content = bookmark.get('content', {})
        if content and (content.get('text_content') or content.get('textContent')):
            score += 0.3
        
        return min(1.0, score)
    
    def _is_generic_title(self, title: str) -> bool:
        """Check if title is generic/low quality"""
        if not title:
            return True
        
        title_lower = title.lower().strip()
        
        generic_patterns = [
            'untitled', 'page', 'document', 'home', 'index', 'default',
            'new tab', 'blank page', 'loading', 'error', '404'
        ]
        
        return any(pattern in title_lower for pattern in generic_patterns)
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid"""
        if not url:
            return False
        
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False