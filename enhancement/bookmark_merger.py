"""Advanced bookmark merging logic for duplicate resolution"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

from enhancement.duplicate_detector import QualityScore
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class MergeResult:
    """Result of bookmark merging operation"""
    merged_bookmark: Dict[str, Any]
    source_bookmarks: List[int]
    merge_strategy: str
    conflicts_resolved: List[str]
    quality_improvement: float
    metadata_preserved: Dict[str, Any]


@dataclass
class MergeConflict:
    """Represents a conflict during merging"""
    field: str
    values: List[Any]
    resolution: str
    chosen_value: Any
    reason: str


class BookmarkMerger:
    """Advanced bookmark merging with intelligent conflict resolution"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize bookmark merger"""
        self.config = config or {}
        self.merge_config = self.config.get('merge', {})
        
        # Merge preferences
        self.prefer_longer_text = self.merge_config.get('prefer_longer_text', True)
        self.preserve_user_data = self.merge_config.get('preserve_user_data', True)
        self.merge_tags_strategy = self.merge_config.get('tags_strategy', 'union')  # union, intersection, quality_based
        self.title_selection_strategy = self.merge_config.get('title_strategy', 'quality_based')  # quality_based, longest, most_descriptive
        self.description_strategy = self.merge_config.get('description_strategy', 'best_quality')  # best_quality, longest, user_preferred
        
        logger.info("Bookmark merger initialized")
    
    def merge_bookmarks(self, 
                       bookmarks: List[Dict[str, Any]], 
                       primary_id: Optional[int] = None,
                       quality_scores: Optional[List[QualityScore]] = None) -> MergeResult:
        """Merge multiple bookmarks into a single bookmark"""
        if not bookmarks:
            raise ValueError("No bookmarks provided for merging")
        
        try:
            
            if len(bookmarks) == 1:
                return MergeResult(
                    merged_bookmark=bookmarks[0].copy(),
                    source_bookmarks=[bookmarks[0].get('id')],
                    merge_strategy="single_bookmark",
                    conflicts_resolved=[],
                    quality_improvement=0.0,
                    metadata_preserved={}
                )
            
            logger.info(f"Merging {len(bookmarks)} bookmarks")
            
            # Determine primary bookmark
            primary_bookmark = self._select_primary_bookmark(bookmarks, primary_id, quality_scores)
            
            # Start with primary bookmark as base
            merged = primary_bookmark.copy()
            conflicts_resolved = []
            
            # Merge each field intelligently
            merged_title, title_conflicts = self._merge_titles(bookmarks, primary_bookmark)
            merged['name'] = merged_title
            conflicts_resolved.extend(title_conflicts)
            
            merged_description, desc_conflicts = self._merge_descriptions(bookmarks, primary_bookmark)
            merged['description'] = merged_description
            conflicts_resolved.extend(desc_conflicts)
            
            merged_tags, tag_conflicts = self._merge_tags(bookmarks)
            merged['tags'] = merged_tags
            conflicts_resolved.extend(tag_conflicts)
            
            merged_url, url_conflicts = self._merge_urls(bookmarks, primary_bookmark)
            if merged_url != primary_bookmark.get('url', ''):
                merged['url'] = merged_url
                conflicts_resolved.extend(url_conflicts)
            
            merged_collection, collection_conflicts = self._merge_collections(bookmarks, primary_bookmark)
            if merged_collection != primary_bookmark.get('collection'):
                merged['collection'] = merged_collection
                conflicts_resolved.extend(collection_conflicts)
            
            merged_content, content_conflicts = self._merge_content(bookmarks, primary_bookmark)
            if merged_content != primary_bookmark.get('content', {}):
                merged['content'] = merged_content
                conflicts_resolved.extend(content_conflicts)
            
            # Merge timestamps (keep earliest created, latest updated)
            merged_timestamps = self._merge_timestamps(bookmarks)
            merged.update(merged_timestamps)
            
            # Calculate quality improvement
            quality_improvement = self._calculate_quality_improvement(
                primary_bookmark, merged, quality_scores
            )
            
            # Preserve metadata about the merge
            metadata_preserved = {
                'original_ids': [b.get('id') for b in bookmarks if b.get('id') is not None],
                'primary_id': primary_bookmark.get('id'),
                'merge_timestamp': datetime.now().isoformat(),
                'merge_strategy': 'intelligent_merge',
                'conflicts_count': len(conflicts_resolved)
            }
            
            return MergeResult(
                merged_bookmark=merged,
                source_bookmarks=[b.get('id') for b in bookmarks if b.get('id') is not None],
                merge_strategy="intelligent_merge",
                conflicts_resolved=[c.field for c in conflicts_resolved],
                quality_improvement=quality_improvement,
                metadata_preserved=metadata_preserved
            )
            
        except Exception as e:
            logger.error(f"Error merging bookmarks: {e}")
            # Return first bookmark as fallback
            return MergeResult(
                merged_bookmark=bookmarks[0].copy() if bookmarks else {},
                source_bookmarks=[bookmarks[0].get('id')] if bookmarks else [],
                merge_strategy="fallback",
                conflicts_resolved=[],
                quality_improvement=0.0,
                metadata_preserved={}
            )
    
    def _select_primary_bookmark(self, 
                                bookmarks: List[Dict[str, Any]], 
                                primary_id: Optional[int],
                                quality_scores: Optional[List[QualityScore]]) -> Dict[str, Any]:
        """Select the primary bookmark for merging"""
        if primary_id:
            primary = next((b for b in bookmarks if b.get('id') == primary_id), None)
            if primary:
                return primary
        
        if quality_scores:
            # Use highest quality bookmark
            quality_lookup = {q.bookmark_id: q for q in quality_scores}
            scored_bookmarks = [(b, quality_lookup.get(b.get('id'))) for b in bookmarks]
            scored_bookmarks = [(b, q) for b, q in scored_bookmarks if q is not None]
            
            if scored_bookmarks:
                best_bookmark, _ = max(scored_bookmarks, key=lambda x: x[1].overall_quality)
                return best_bookmark
        
        # Fallback to first bookmark
        return bookmarks[0]
    
    def _merge_titles(self, 
                     bookmarks: List[Dict[str, Any]], 
                     primary: Dict[str, Any]) -> Tuple[str, List[MergeConflict]]:
        """Merge titles from multiple bookmarks"""
        titles = []
        conflicts = []
        
        for bookmark in bookmarks:
            title = bookmark.get('name', '') or bookmark.get('title', '')
            if title and title not in titles:
                titles.append(title)
        
        if not titles:
            return '', []
        
        if len(titles) == 1:
            return titles[0], []
        
        # Apply title selection strategy
        if self.title_selection_strategy == 'longest':
            selected_title = max(titles, key=len)
            reason = "Selected longest title"
        elif self.title_selection_strategy == 'most_descriptive':
            selected_title = self._select_most_descriptive_title(titles)
            reason = "Selected most descriptive title"
        else:  # quality_based
            selected_title = self._select_highest_quality_title(titles)
            reason = "Selected highest quality title"
        
        if selected_title != primary.get('name', ''):
            conflict = MergeConflict(
                field='title',
                values=titles,
                resolution=self.title_selection_strategy,
                chosen_value=selected_title,
                reason=reason
            )
            conflicts.append(conflict)
        
        return selected_title, conflicts
    
    def _merge_descriptions(self, 
                           bookmarks: List[Dict[str, Any]], 
                           primary: Dict[str, Any]) -> Tuple[str, List[MergeConflict]]:
        """Merge descriptions from multiple bookmarks"""
        descriptions = []
        conflicts = []
        
        for bookmark in bookmarks:
            desc = bookmark.get('description', '')
            if desc and desc not in descriptions:
                descriptions.append(desc)
        
        if not descriptions:
            return '', []
        
        if len(descriptions) == 1:
            return descriptions[0], []
        
        # Apply description selection strategy
        if self.description_strategy == 'longest':
            selected_desc = max(descriptions, key=len)
            reason = "Selected longest description"
        elif self.description_strategy == 'user_preferred':
            # Prefer descriptions that don't look auto-generated
            user_descriptions = [d for d in descriptions if not self._looks_auto_generated(d)]
            if user_descriptions:
                selected_desc = max(user_descriptions, key=len)
                reason = "Selected user-written description"
            else:
                selected_desc = max(descriptions, key=len)
                reason = "Selected longest description (no user descriptions found)"
        else:  # best_quality
            selected_desc = self._select_best_quality_description(descriptions)
            reason = "Selected best quality description"
        
        if selected_desc != primary.get('description', ''):
            conflict = MergeConflict(
                field='description',
                values=descriptions,
                resolution=self.description_strategy,
                chosen_value=selected_desc,
                reason=reason
            )
            conflicts.append(conflict)
        
        return selected_desc, conflicts
    
    def _merge_tags(self, bookmarks: List[Dict[str, Any]]) -> Tuple[List[str], List[MergeConflict]]:
        """Merge tags from multiple bookmarks"""
        all_tags = set()
        tag_sources = {}
        conflicts = []
        
        for bookmark in bookmarks:
            tags = bookmark.get('tags', [])
            if isinstance(tags, str):
                tags = [tags]
            elif not isinstance(tags, list):
                continue
            
            for tag in tags:
                if isinstance(tag, str) and tag.strip():
                    clean_tag = tag.strip().lower()
                    all_tags.add(clean_tag)
                    if clean_tag not in tag_sources:
                        tag_sources[clean_tag] = []
                    tag_sources[clean_tag].append(bookmark.get('id'))
        
        if not all_tags:
            return [], []
        
        # Apply tag merging strategy
        if self.merge_tags_strategy == 'intersection':
            # Only keep tags that appear in all bookmarks
            common_tags = all_tags.copy()
            for bookmark in bookmarks:
                bookmark_tags = set()
                tags = bookmark.get('tags', [])
                if isinstance(tags, str):
                    tags = [tags]
                for tag in tags:
                    if isinstance(tag, str):
                        bookmark_tags.add(tag.strip().lower())
                common_tags &= bookmark_tags
            
            merged_tags = list(common_tags)
            reason = "Kept only common tags"
        elif self.merge_tags_strategy == 'quality_based':
            # Keep tags based on frequency and quality
            merged_tags = self._select_quality_tags(tag_sources, bookmarks)
            reason = "Selected tags based on quality and frequency"
        else:  # union (default)
            merged_tags = list(all_tags)
            reason = "Combined all unique tags"
        
        # Sort tags for consistency
        merged_tags.sort()
        
        if len(set(merged_tags)) > len(all_tags) or self.merge_tags_strategy != 'union':
            conflict = MergeConflict(
                field='tags',
                values=[b.get('tags', []) for b in bookmarks],
                resolution=self.merge_tags_strategy,
                chosen_value=merged_tags,
                reason=reason
            )
            conflicts.append(conflict)
        
        return merged_tags, conflicts
    
    def _merge_urls(self, 
                   bookmarks: List[Dict[str, Any]], 
                   primary: Dict[str, Any]) -> Tuple[str, List[MergeConflict]]:
        """Merge URLs, preferring HTTPS and canonical versions"""
        urls = [b.get('url', '') for b in bookmarks if b.get('url', '')]
        
        if not urls:
            return '', []
        
        if len(set(urls)) == 1:
            return urls[0], []
        
        # Prefer HTTPS URLs
        https_urls = [url for url in urls if url.startswith('https://')]
        if https_urls:
            selected_url = https_urls[0]
            reason = "Selected HTTPS URL"
        else:
            selected_url = urls[0]
            reason = "Selected first available URL"
        
        conflicts = []
        if selected_url != primary.get('url', ''):
            conflict = MergeConflict(
                field='url',
                values=urls,
                resolution='prefer_https',
                chosen_value=selected_url,
                reason=reason
            )
            conflicts.append(conflict)
        
        return selected_url, conflicts
    
    def _merge_collections(self, 
                          bookmarks: List[Dict[str, Any]], 
                          primary: Dict[str, Any]) -> Tuple[Optional[str], List[MergeConflict]]:
        """Merge collection assignments"""
        collections = [b.get('collection') for b in bookmarks if b.get('collection')]
        
        if not collections:
            return None, []
        
        if len(set(collections)) == 1:
            return collections[0], []
        
        # Prefer primary bookmark's collection, or most common
        from collections import Counter
        collection_counts = Counter(collections)
        most_common = collection_counts.most_common(1)[0][0]
        
        conflicts = []
        if most_common != primary.get('collection'):
            conflict = MergeConflict(
                field='collection',
                values=collections,
                resolution='most_common',
                chosen_value=most_common,
                reason="Selected most common collection"
            )
            conflicts.append(conflict)
        
        return most_common, conflicts
    
    def _merge_content(self, 
                      bookmarks: List[Dict[str, Any]], 
                      primary: Dict[str, Any]) -> Tuple[Dict[str, Any], List[MergeConflict]]:
        """Merge content data"""
        contents = [b.get('content', {}) for b in bookmarks if b.get('content')]
        
        if not contents:
            return {}, []
        
        if len(contents) == 1:
            return contents[0], []
        
        # Merge content fields
        merged_content = {}
        conflicts = []
        
        # Collect all content fields
        all_fields = set()
        for content in contents:
            if isinstance(content, dict):
                all_fields.update(content.keys())
        
        for field in all_fields:
            field_values = [c.get(field) for c in contents if isinstance(c, dict) and c.get(field)]
            
            if field_values:
                if field in ['text_content', 'textContent']:
                    # For text content, prefer longest
                    merged_content[field] = max(field_values, key=lambda x: len(str(x)))
                else:
                    # For other fields, use first non-empty value
                    merged_content[field] = field_values[0]
        
        return merged_content, conflicts
    
    def _merge_timestamps(self, bookmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge timestamp fields"""
        timestamps = {}
        
        # Collect all timestamps
        created_times = []
        updated_times = []
        
        for bookmark in bookmarks:
            created = bookmark.get('created_at') or bookmark.get('createdAt')
            updated = bookmark.get('updated_at') or bookmark.get('updatedAt')
            
            if created:
                created_times.append(created)
            if updated:
                updated_times.append(updated)
        
        # Use earliest created time
        if created_times:
            timestamps['created_at'] = min(created_times)
        
        # Use latest updated time
        if updated_times:
            timestamps['updated_at'] = max(updated_times)
        
        return timestamps
    
    def _select_most_descriptive_title(self, titles: List[str]) -> str:
        """Select the most descriptive title"""
        def descriptiveness_score(title: str) -> float:
            score = 0.0
            
            # Length score (moderate length is better)
            if 20 <= len(title) <= 80:
                score += 0.3
            elif 10 <= len(title) <= 100:
                score += 0.2
            
            # Word count score
            words = title.split()
            if 3 <= len(words) <= 10:
                score += 0.3
            elif len(words) >= 2:
                score += 0.2
            
            # Avoid generic words
            generic_words = {'page', 'document', 'untitled', 'home', 'index'}
            if not any(word.lower() in generic_words for word in words):
                score += 0.2
            
            # Prefer titles with descriptive words
            descriptive_words = {'descriptive', 'guide', 'tutorial', 'comprehensive', 'complete', 'detailed'}
            if any(word.lower() in descriptive_words for word in words):
                score += 0.3
            
            # Prefer titles with meaningful words
            meaningful_words = sum(1 for word in words if len(word) > 3)
            score += min(0.2, meaningful_words * 0.05)
            
            return score
        
        return max(titles, key=descriptiveness_score)
    
    def _select_highest_quality_title(self, titles: List[str]) -> str:
        """Select highest quality title based on multiple factors"""
        def quality_score(title: str) -> float:
            if not title:
                return 0.0
            
            score = 0.0
            
            # Length score (prefer longer titles up to a point)
            length = len(title)
            if 30 <= length <= 80:
                score += 0.4
            elif 15 <= length <= 100:
                score += 0.3
            elif 10 <= length <= 120:
                score += 0.2
            elif length > 5:
                score += 0.1
            
            # Word count (more words generally better)
            words = title.split()
            word_count = len(words)
            if word_count >= 5:
                score += 0.3
            elif word_count >= 3:
                score += 0.2
            elif word_count >= 2:
                score += 0.1
            
            # Avoid generic patterns
            generic_patterns = ['untitled', 'page', 'document', 'home', 'index']
            if not any(pattern in title.lower() for pattern in generic_patterns):
                score += 0.2
            
            # Prefer descriptive words
            descriptive_words = ['guide', 'tutorial', 'comprehensive', 'complete', 'detailed', 'descriptive']
            if any(word.lower() in descriptive_words for word in words):
                score += 0.1
            
            return score
        
        return max(titles, key=quality_score)
    
    def _select_best_quality_description(self, descriptions: List[str]) -> str:
        """Select best quality description"""
        def quality_score(desc: str) -> float:
            if not desc:
                return 0.0
            
            score = 0.0
            
            # Length score
            if 50 <= len(desc) <= 500:
                score += 0.4
            elif len(desc) > 20:
                score += 0.2
            
            # Word count
            words = desc.split()
            if len(words) >= 5:
                score += 0.3
            
            # Not just URL
            if not desc.startswith('http'):
                score += 0.3
            
            return score
        
        return max(descriptions, key=quality_score)
    
    def _select_quality_tags(self, tag_sources: Dict[str, List[int]], bookmarks: List[Dict[str, Any]]) -> List[str]:
        """Select tags based on quality and frequency"""
        tag_scores = {}
        
        for tag, sources in tag_sources.items():
            score = 0.0
            
            # Frequency score
            frequency = len(sources) / len(bookmarks)
            score += frequency * 0.5
            
            # Length score (avoid very short or very long tags)
            if 3 <= len(tag) <= 20:
                score += 0.3
            elif len(tag) >= 2:
                score += 0.1
            
            # Avoid generic tags
            generic_tags = {'tag', 'bookmark', 'link', 'page', 'site', 'web'}
            if tag not in generic_tags:
                score += 0.2
            
            tag_scores[tag] = score
        
        # Select top tags (limit to reasonable number)
        sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        return [tag for tag, score in sorted_tags[:15] if score > 0.3]
    
    def _looks_auto_generated(self, description: str) -> bool:
        """Check if description looks auto-generated"""
        if not description:
            return True
        
        # Check for URL-only descriptions
        if description.startswith('http'):
            return True
        
        # Check for very short descriptions
        if len(description) < 10:
            return True
        
        # Check for generic patterns
        generic_patterns = [
            r'^page \d+',
            r'^document',
            r'^untitled',
            r'^loading',
            r'^error'
        ]
        
        for pattern in generic_patterns:
            if re.match(pattern, description.lower()):
                return True
        
        return False
    
    def _calculate_quality_improvement(self, 
                                     original: Dict[str, Any], 
                                     merged: Dict[str, Any],
                                     quality_scores: Optional[List[QualityScore]]) -> float:
        """Calculate quality improvement from merging"""
        try:
            # Simple quality metrics
            original_score = 0.0
            merged_score = 0.0
            
            # Title quality
            orig_title = original.get('name', '') or original.get('title', '')
            merged_title = merged.get('name', '') or merged.get('title', '')
            
            if len(merged_title) > len(orig_title):
                merged_score += 0.2
            
            # Description quality
            orig_desc = original.get('description', '')
            merged_desc = merged.get('description', '')
            
            if len(merged_desc) > len(orig_desc):
                merged_score += 0.3
            
            # Tag count
            orig_tags = original.get('tags', [])
            merged_tags = merged.get('tags', [])
            
            if len(merged_tags) > len(orig_tags):
                merged_score += 0.2
            
            # Content completeness
            orig_content = original.get('content', {})
            merged_content = merged.get('content', {})
            
            if len(merged_content) > len(orig_content):
                merged_score += 0.3
            
            return merged_score - original_score
            
        except Exception as e:
            logger.warning(f"Error calculating quality improvement: {e}")
            return 0.0