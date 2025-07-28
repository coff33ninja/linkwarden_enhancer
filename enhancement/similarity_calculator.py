"""Enhanced similarity calculation engine for duplicate detection"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import re

from ai.similarity_engine import SimilarityEngine, SimilarityResult, DuplicateGroup
from enhancement.url_normalizer import URLNormalizer
from utils.logging_utils import get_logger
from utils.text_utils import TextUtils

logger = get_logger(__name__)


@dataclass
class ComprehensiveSimilarityScore:
    """Comprehensive similarity score with component breakdowns"""
    url_similarity: float
    title_similarity: float
    content_similarity: float
    semantic_similarity: float
    overall_similarity: float
    confidence: float
    
    def __post_init__(self):
        """Calculate overall similarity from components"""
        if self.overall_similarity == 0.0:
            # Weighted combination of similarity components
            self.overall_similarity = (
                self.url_similarity * 0.3 +
                self.title_similarity * 0.25 +
                self.content_similarity * 0.2 +
                self.semantic_similarity * 0.25
            )
        
        # Calculate confidence based on how many components have high scores
        high_scores = sum([
            1 for score in [self.url_similarity, self.title_similarity, 
                          self.content_similarity, self.semantic_similarity]
            if score > 0.7
        ])
        self.confidence = min(1.0, (high_scores + 1) / 5.0)


@dataclass
class EnhancedSimilarityResult(SimilarityResult):
    """Enhanced similarity result with detailed scoring"""
    detailed_score: ComprehensiveSimilarityScore = None
    url_normalized: str = ""
    title_cleaned: str = ""


class SimilarityCalculator:
    """Enhanced similarity calculation engine combining multiple similarity metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize similarity calculator"""
        self.config = config
        self.similarity_config = config.get('similarity', {})
        
        # Initialize components
        self.url_normalizer = URLNormalizer()
        self.semantic_engine = SimilarityEngine(config)
        
        # Similarity thresholds
        self.url_similarity_threshold = self.similarity_config.get('url_threshold', 0.8)
        self.title_similarity_threshold = self.similarity_config.get('title_threshold', 0.7)
        self.content_similarity_threshold = self.similarity_config.get('content_threshold', 0.6)
        self.semantic_similarity_threshold = self.similarity_config.get('semantic_threshold', 0.7)
        self.overall_similarity_threshold = self.similarity_config.get('overall_threshold', 0.75)
        
        # Text processing utilities
        self.text_utils = TextUtils()
        
        logger.info("Enhanced similarity calculator initialized")
    
    def calculate_url_similarity(self, url1: str, url2: str) -> float:
        """Calculate URL similarity using normalization and comparison"""
        try:
            return self.url_normalizer.get_url_similarity_score(url1, url2)
        except Exception as e:
            logger.warning(f"Error calculating URL similarity: {e}")
            return 0.0
    
    def calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity using multiple methods"""
        try:
            if not title1 or not title2:
                return 0.0
            
            # Clean and normalize titles
            clean_title1 = self._clean_title(title1)
            clean_title2 = self._clean_title(title2)
            
            if not clean_title1 or not clean_title2:
                return 0.0
            
            # Exact match
            if clean_title1.lower() == clean_title2.lower():
                return 1.0
            
            # Jaccard similarity (word overlap)
            words1 = set(clean_title1.lower().split())
            words2 = set(clean_title2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1 & words2
            union = words1 | words2
            jaccard_score = len(intersection) / len(union) if union else 0.0
            
            # Levenshtein similarity (character-level)
            levenshtein_score = self._levenshtein_similarity(clean_title1, clean_title2)
            
            # Longest common subsequence similarity
            lcs_score = self._lcs_similarity(clean_title1, clean_title2)
            
            # Combine scores with weights
            combined_score = (
                jaccard_score * 0.4 +
                levenshtein_score * 0.3 +
                lcs_score * 0.3
            )
            
            return min(1.0, combined_score)
            
        except Exception as e:
            logger.warning(f"Error calculating title similarity: {e}")
            return 0.0
    
    def calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using text analysis"""
        try:
            if not content1 or not content2:
                return 0.0
            
            # Clean and normalize content
            clean_content1 = self._clean_content(content1)
            clean_content2 = self._clean_content(content2)
            
            if not clean_content1 or not clean_content2:
                return 0.0
            
            # Extract keywords and calculate overlap
            keywords1 = self._extract_keywords(clean_content1)
            keywords2 = self._extract_keywords(clean_content2)
            
            if not keywords1 or not keywords2:
                return 0.0
            
            # Jaccard similarity for keywords
            intersection = keywords1 & keywords2
            union = keywords1 | keywords2
            keyword_similarity = len(intersection) / len(union) if union else 0.0
            
            # Character n-gram similarity
            ngram_similarity = self._ngram_similarity(clean_content1, clean_content2, n=3)
            
            # Combine scores
            combined_score = (keyword_similarity * 0.6 + ngram_similarity * 0.4)
            
            return min(1.0, combined_score)
            
        except Exception as e:
            logger.warning(f"Error calculating content similarity: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, bookmark1: Dict[str, Any], bookmark2: Dict[str, Any]) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            # Use the existing semantic engine
            bookmarks = [bookmark1, bookmark2]
            self.semantic_engine.compute_bookmark_embeddings(bookmarks)
            
            id1 = bookmark1.get('id')
            id2 = bookmark2.get('id')
            
            if id1 not in self.semantic_engine.bookmark_embeddings or id2 not in self.semantic_engine.bookmark_embeddings:
                return 0.0
            
            embedding1 = self.semantic_engine.bookmark_embeddings[id1]
            embedding2 = self.semantic_engine.bookmark_embeddings[id2]
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_comprehensive_similarity(self, bookmark1: Dict[str, Any], bookmark2: Dict[str, Any]) -> ComprehensiveSimilarityScore:
        """Calculate comprehensive similarity score with all components"""
        try:
            # Extract data from bookmarks
            url1 = bookmark1.get('url', '')
            url2 = bookmark2.get('url', '')
            title1 = bookmark1.get('name', '') or bookmark1.get('title', '')
            title2 = bookmark2.get('name', '') or bookmark2.get('title', '')
            
            # Get content
            content1 = self._extract_bookmark_content(bookmark1)
            content2 = self._extract_bookmark_content(bookmark2)
            
            # Calculate component similarities
            url_sim = self.calculate_url_similarity(url1, url2)
            title_sim = self.calculate_title_similarity(title1, title2)
            content_sim = self.calculate_content_similarity(content1, content2)
            semantic_sim = self.calculate_semantic_similarity(bookmark1, bookmark2)
            
            return ComprehensiveSimilarityScore(
                url_similarity=url_sim,
                title_similarity=title_sim,
                content_similarity=content_sim,
                semantic_similarity=semantic_sim,
                overall_similarity=0.0,  # Will be calculated in __post_init__
                confidence=0.0  # Will be calculated in __post_init__
            )
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive similarity: {e}")
            return ComprehensiveSimilarityScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def find_similar_bookmarks_enhanced(self, 
                                      target_bookmark: Dict[str, Any], 
                                      candidate_bookmarks: List[Dict[str, Any]], 
                                      limit: int = 10) -> List[EnhancedSimilarityResult]:
        """Find similar bookmarks with enhanced scoring"""
        try:
            results = []
            target_id = target_bookmark.get('id')
            
            for candidate in candidate_bookmarks:
                candidate_id = candidate.get('id')
                
                if candidate_id == target_id:
                    continue
                
                # Calculate comprehensive similarity
                similarity_score = self.calculate_comprehensive_similarity(target_bookmark, candidate)
                
                if similarity_score.overall_similarity >= self.overall_similarity_threshold:
                    # Normalize URLs for display
                    normalized_url = self.url_normalizer.normalize_url(candidate.get('url', '')).normalized_url
                    cleaned_title = self._clean_title(candidate.get('name', '') or candidate.get('title', ''))
                    
                    result = EnhancedSimilarityResult(
                        bookmark_id=candidate_id,
                        similarity_score=similarity_score.overall_similarity,
                        title=candidate.get('name', '') or candidate.get('title', ''),
                        url=candidate.get('url', ''),
                        description=candidate.get('description', ''),
                        match_type=self._determine_match_type(similarity_score),
                        detailed_score=similarity_score,
                        url_normalized=normalized_url,
                        title_cleaned=cleaned_title
                    )
                    
                    results.append(result)
            
            # Sort by overall similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar bookmarks: {e}")
            return []
    
    def detect_duplicates_enhanced(self, bookmarks: List[Dict[str, Any]]) -> List[DuplicateGroup]:
        """Enhanced duplicate detection with comprehensive similarity"""
        try:
            logger.info(f"Enhanced duplicate detection for {len(bookmarks)} bookmarks")
            
            duplicate_groups = []
            processed_bookmarks = set()
            
            # Create similarity matrix
            n = len(bookmarks)
            similarity_matrix = np.zeros((n, n))
            
            # Calculate pairwise similarities
            for i in range(n):
                for j in range(i + 1, n):
                    similarity_score = self.calculate_comprehensive_similarity(bookmarks[i], bookmarks[j])
                    similarity_matrix[i][j] = similarity_score.overall_similarity
                    similarity_matrix[j][i] = similarity_score.overall_similarity
            
            # Find groups using different thresholds
            group_id = 0
            
            # High similarity groups (likely duplicates)
            high_threshold = 0.9
            high_groups = self._find_similarity_groups(
                similarity_matrix, bookmarks, high_threshold, "exact", group_id
            )
            duplicate_groups.extend(high_groups)
            group_id += len(high_groups)
            
            for group in high_groups:
                processed_bookmarks.update(group.bookmarks)
            
            # Medium similarity groups (near duplicates)
            medium_threshold = 0.75
            medium_groups = self._find_similarity_groups(
                similarity_matrix, bookmarks, medium_threshold, "near_duplicate", 
                group_id, exclude_ids=processed_bookmarks
            )
            duplicate_groups.extend(medium_groups)
            group_id += len(medium_groups)
            
            for group in medium_groups:
                processed_bookmarks.update(group.bookmarks)
            
            # Lower similarity groups (similar content)
            low_threshold = 0.6
            low_groups = self._find_similarity_groups(
                similarity_matrix, bookmarks, low_threshold, "similar", 
                group_id, exclude_ids=processed_bookmarks
            )
            duplicate_groups.extend(low_groups)
            
            logger.info(f"Found {len(duplicate_groups)} duplicate groups")
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Error in enhanced duplicate detection: {e}")
            return []
    
    def _find_similarity_groups(self, 
                               similarity_matrix: np.ndarray, 
                               bookmarks: List[Dict[str, Any]], 
                               threshold: float, 
                               group_type: str,
                               start_group_id: int,
                               exclude_ids: set = None) -> List[DuplicateGroup]:
        """Find groups of similar bookmarks using threshold-based clustering"""
        if exclude_ids is None:
            exclude_ids = set()
        
        try:
            groups = []
            visited = set()
            
            for i, bookmark in enumerate(bookmarks):
                bookmark_id = bookmark.get('id')
                
                if bookmark_id in exclude_ids or i in visited:
                    continue
                
                # Find all bookmarks similar to this one
                similar_indices = []
                similar_scores = []
                
                for j, other_bookmark in enumerate(bookmarks):
                    other_id = other_bookmark.get('id')
                    
                    if i != j and other_id not in exclude_ids and j not in visited:
                        if similarity_matrix[i][j] >= threshold:
                            similar_indices.append(j)
                            similar_scores.append(similarity_matrix[i][j])
                
                # If we found similar bookmarks, create a group
                if similar_indices:
                    group_indices = [i] + similar_indices
                    group_bookmarks = [bookmarks[idx].get('id') for idx in group_indices]
                    group_scores = [1.0] + similar_scores  # Self-similarity is 1.0
                    
                    # Mark as visited
                    visited.update(group_indices)
                    
                    # Choose representative (highest average similarity to others)
                    representative_idx = i
                    best_avg_similarity = 0.0
                    
                    for idx in group_indices:
                        avg_sim = np.mean([similarity_matrix[idx][other_idx] 
                                         for other_idx in group_indices if other_idx != idx])
                        if avg_sim > best_avg_similarity:
                            best_avg_similarity = avg_sim
                            representative_idx = idx
                    
                    representative_id = bookmarks[representative_idx].get('id')
                    
                    group = DuplicateGroup(
                        group_id=start_group_id + len(groups),
                        bookmarks=group_bookmarks,
                        similarity_scores=group_scores,
                        group_type=group_type,
                        representative_bookmark=representative_id
                    )
                    
                    groups.append(group)
            
            return groups
            
        except Exception as e:
            logger.error(f"Error finding similarity groups: {e}")
            return []
    
    def _determine_match_type(self, similarity_score: ComprehensiveSimilarityScore) -> str:
        """Determine the type of match based on similarity components"""
        if similarity_score.url_similarity > 0.9:
            return "url_match"
        elif similarity_score.title_similarity > 0.9:
            return "title_match"
        elif similarity_score.semantic_similarity > 0.8:
            return "semantic_match"
        elif similarity_score.content_similarity > 0.8:
            return "content_match"
        else:
            return "mixed_match"
    
    def _clean_title(self, title: str) -> str:
        """Clean and normalize title for comparison"""
        if not title:
            return ""
        
        # Remove common site suffixes
        title = re.sub(r'\s*[-|–—]\s*[^-|–—]*$', '', title)
        
        # Remove extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove special characters for comparison
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        return title
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for comparison"""
        if not content:
            return ""
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Limit length for processing
        if len(content) > 1000:
            content = content[:1000]
        
        return content
    
    def _extract_bookmark_content(self, bookmark: Dict[str, Any]) -> str:
        """Extract content from bookmark for similarity comparison"""
        content_parts = []
        
        # Add description
        description = bookmark.get('description', '')
        if description:
            content_parts.append(description)
        
        # Add content from content field
        content = bookmark.get('content', {})
        if isinstance(content, dict):
            text_content = content.get('text_content', '') or content.get('textContent', '')
            if text_content:
                content_parts.append(text_content)
        elif isinstance(content, str):
            content_parts.append(content)
        
        return ' '.join(content_parts)
    
    def _extract_keywords(self, text: str, max_keywords: int = 20) -> set:
        """Extract keywords from text"""
        if not text:
            return set()
        
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return most frequent keywords
        from collections import Counter
        word_counts = Counter(keywords)
        return set([word for word, count in word_counts.most_common(max_keywords)])
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity (normalized edit distance)"""
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Calculate Levenshtein distance
        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1
        
        current_row = list(range(len1 + 1))
        for i in range(1, len2 + 1):
            previous_row, current_row = current_row, [i] + [0] * len1
            for j in range(1, len1 + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if s1[j - 1] != s2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)
        
        distance = current_row[len1]
        max_len = max(len(s1), len(s2))
        
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0
    
    def _lcs_similarity(self, s1: str, s2: str) -> float:
        """Calculate Longest Common Subsequence similarity"""
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Calculate LCS length
        len1, len2 = len(s1), len(s2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        lcs_length = dp[len1][len2]
        max_len = max(len1, len2)
        
        return lcs_length / max_len if max_len > 0 else 0.0
    
    def _ngram_similarity(self, s1: str, s2: str, n: int = 3) -> float:
        """Calculate n-gram similarity"""
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Generate n-grams
        def get_ngrams(text: str, n: int) -> set:
            if len(text) < n:
                return {text}
            return {text[i:i + n] for i in range(len(text) - n + 1)}
        
        ngrams1 = get_ngrams(s1.lower(), n)
        ngrams2 = get_ngrams(s2.lower(), n)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union) if union else 0.0