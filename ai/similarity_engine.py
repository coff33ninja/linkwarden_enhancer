"""Similarity engine for semantic similarity and duplicate detection using sentence transformers"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import pickle
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from ..utils.logging_utils import get_logger
from ..utils.text_utils import TextUtils

logger = get_logger(__name__)


@dataclass
class SimilarityResult:
    """Result of similarity comparison"""
    bookmark_id: int
    similarity_score: float
    title: str
    url: str
    description: str = ""
    match_type: str = "semantic"  # semantic, title, url, content


@dataclass
class DuplicateGroup:
    """Group of potentially duplicate bookmarks"""
    group_id: int
    bookmarks: List[int]
    similarity_scores: List[float]
    group_type: str  # exact, near_duplicate, similar
    representative_bookmark: int  # ID of the most representative bookmark


class SimilarityEngine:
    """Semantic similarity engine using sentence transformers"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize similarity engine"""
        self.config = config
        self.similarity_config = config.get('similarity', {})
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        # Model configuration
        self.model_name = self.similarity_config.get('model_name', 'all-MiniLM-L6-v2')
        self.cache_dir = Path(config.get('directories', {}).get('models_dir', 'models'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Similarity thresholds
        self.duplicate_threshold = self.similarity_config.get('duplicate_threshold', 0.95)
        self.near_duplicate_threshold = self.similarity_config.get('near_duplicate_threshold', 0.85)
        self.similar_threshold = self.similarity_config.get('similar_threshold', 0.7)
        self.recommendation_threshold = self.similarity_config.get('recommendation_threshold', 0.6)
        
        # Processing settings
        self.max_text_length = self.similarity_config.get('max_text_length', 512)
        self.batch_size = self.similarity_config.get('batch_size', 32)
        
        # Initialize model
        self.model = None
        self.embeddings_cache = {}
        self.bookmark_embeddings = {}
        
        self._initialize_model()
        
        logger.info(f"Similarity engine initialized with model: {self.model_name}")
    
    def _initialize_model(self) -> None:
        """Initialize sentence transformer model"""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {e}")
            raise
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a list of texts"""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                if text:
                    # Truncate if too long
                    if len(text) > self.max_text_length:
                        text = text[:self.max_text_length]
                    processed_texts.append(text)
                else:
                    processed_texts.append("")
            
            # Compute embeddings in batches
            embeddings = self.model.encode(
                processed_texts,
                batch_size=self.batch_size,
                show_progress_bar=len(processed_texts) > 100,
                convert_to_numpy=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            raise
    
    def compute_bookmark_embeddings(self, bookmarks: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
        """Compute embeddings for bookmarks"""
        try:
            logger.info(f"Computing embeddings for {len(bookmarks)} bookmarks")
            
            # Prepare texts for embedding
            texts = []
            bookmark_ids = []
            
            for bookmark in bookmarks:
                bookmark_id = bookmark.get('id')
                if bookmark_id is None:
                    continue
                
                # Combine title, description, and content for embedding
                title = bookmark.get('name', '') or ''
                description = bookmark.get('description', '') or ''
                content = bookmark.get('content', {})
                
                if isinstance(content, dict):
                    text_content = content.get('text_content', '') or ''
                else:
                    text_content = str(content) if content else ''
                
                # Create combined text for embedding
                combined_text = f"{title} {description} {text_content}".strip()
                
                if combined_text:
                    texts.append(combined_text)
                    bookmark_ids.append(bookmark_id)
            
            if not texts:
                logger.warning("No valid texts found for embedding computation")
                return {}
            
            # Compute embeddings
            embeddings = self.compute_embeddings(texts)
            
            # Create mapping
            bookmark_embeddings = {}
            for i, bookmark_id in enumerate(bookmark_ids):
                bookmark_embeddings[bookmark_id] = embeddings[i]
            
            # Cache embeddings
            self.bookmark_embeddings.update(bookmark_embeddings)
            
            logger.info(f"Computed embeddings for {len(bookmark_embeddings)} bookmarks")
            return bookmark_embeddings
            
        except Exception as e:
            logger.error(f"Failed to compute bookmark embeddings: {e}")
            return {}
    
    def find_similar_bookmarks(self, 
                              target_bookmark_id: int, 
                              bookmarks: List[Dict[str, Any]], 
                              limit: int = 10) -> List[SimilarityResult]:
        """Find bookmarks similar to the target bookmark"""
        
        try:
            # Ensure embeddings are computed
            if not self.bookmark_embeddings:
                self.compute_bookmark_embeddings(bookmarks)
            
            if target_bookmark_id not in self.bookmark_embeddings:
                logger.warning(f"No embedding found for bookmark {target_bookmark_id}")
                return []
            
            target_embedding = self.bookmark_embeddings[target_bookmark_id]
            target_bookmark = next((b for b in bookmarks if b.get('id') == target_bookmark_id), None)
            
            if not target_bookmark:
                logger.warning(f"Target bookmark {target_bookmark_id} not found")
                return []
            
            # Compute similarities
            similarities = []
            
            for bookmark in bookmarks:
                bookmark_id = bookmark.get('id')
                
                if bookmark_id == target_bookmark_id or bookmark_id not in self.bookmark_embeddings:
                    continue
                
                # Compute cosine similarity
                other_embedding = self.bookmark_embeddings[bookmark_id]
                similarity = cosine_similarity(
                    target_embedding.reshape(1, -1),
                    other_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity >= self.recommendation_threshold:
                    similarities.append(SimilarityResult(
                        bookmark_id=bookmark_id,
                        similarity_score=float(similarity),
                        title=bookmark.get('name', ''),
                        url=bookmark.get('url', ''),
                        description=bookmark.get('description', ''),
                        match_type='semantic'
                    ))
            
            # Sort by similarity score (highest first)
            similarities.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.debug(f"Found {len(similarities)} similar bookmarks for {target_bookmark_id}")
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar bookmarks: {e}")
            return []
    
    def detect_duplicates(self, bookmarks: List[Dict[str, Any]]) -> List[DuplicateGroup]:
        """Detect duplicate and near-duplicate bookmarks"""
        
        try:
            logger.info(f"Detecting duplicates in {len(bookmarks)} bookmarks")
            
            # Ensure embeddings are computed
            if not self.bookmark_embeddings:
                self.compute_bookmark_embeddings(bookmarks)
            
            # Create bookmark lookup
            bookmark_lookup = {b.get('id'): b for b in bookmarks if b.get('id') is not None}
            
            # Get valid bookmark IDs with embeddings
            valid_ids = [bid for bid in bookmark_lookup.keys() if bid in self.bookmark_embeddings]
            
            if len(valid_ids) < 2:
                logger.warning("Not enough bookmarks with embeddings for duplicate detection")
                return []
            
            # Compute pairwise similarities
            embeddings_matrix = np.array([self.bookmark_embeddings[bid] for bid in valid_ids])
            similarity_matrix = cosine_similarity(embeddings_matrix)
            
            # Find duplicate groups using different thresholds
            duplicate_groups = []
            processed_bookmarks = set()
            
            # Exact duplicates (very high similarity)
            exact_groups = self._find_duplicate_groups(
                similarity_matrix, valid_ids, self.duplicate_threshold, "exact"
            )
            for group in exact_groups:
                duplicate_groups.append(group)
                processed_bookmarks.update(group.bookmarks)
            
            # Near duplicates (high similarity)
            near_duplicate_groups = self._find_duplicate_groups(
                similarity_matrix, valid_ids, self.near_duplicate_threshold, "near_duplicate",
                exclude_ids=processed_bookmarks
            )
            for group in near_duplicate_groups:
                duplicate_groups.append(group)
                processed_bookmarks.update(group.bookmarks)
            
            # Similar bookmarks (moderate similarity)
            similar_groups = self._find_duplicate_groups(
                similarity_matrix, valid_ids, self.similar_threshold, "similar",
                exclude_ids=processed_bookmarks
            )
            for group in similar_groups:
                duplicate_groups.append(group)
            
            logger.info(f"Found {len(duplicate_groups)} duplicate groups")
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Failed to detect duplicates: {e}")
            return []
    
    def _find_duplicate_groups(self, 
                              similarity_matrix: np.ndarray, 
                              bookmark_ids: List[int], 
                              threshold: float, 
                              group_type: str,
                              exclude_ids: set = None) -> List[DuplicateGroup]:
        """Find groups of similar bookmarks using clustering"""
        
        if exclude_ids is None:
            exclude_ids = set()
        
        try:
            # Filter out excluded bookmarks
            valid_indices = [i for i, bid in enumerate(bookmark_ids) if bid not in exclude_ids]
            
            if len(valid_indices) < 2:
                return []
            
            # Create filtered similarity matrix
            filtered_similarity = similarity_matrix[np.ix_(valid_indices, valid_indices)]
            filtered_ids = [bookmark_ids[i] for i in valid_indices]
            
            # Convert similarity to distance for clustering
            distance_matrix = 1 - filtered_similarity
            
            # Use DBSCAN clustering to find groups
            # eps = 1 - threshold (distance threshold)
            eps = 1 - threshold
            clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Group bookmarks by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # -1 is noise in DBSCAN
                    clusters[label].append(i)
            
            # Create duplicate groups
            duplicate_groups = []
            group_id = 0
            
            for cluster_indices in clusters.values():
                if len(cluster_indices) >= 2:
                    bookmark_group = [filtered_ids[i] for i in cluster_indices]
                    
                    # Calculate average similarity scores within group
                    similarities = []
                    for i in range(len(cluster_indices)):
                        for j in range(i + 1, len(cluster_indices)):
                            idx1, idx2 = cluster_indices[i], cluster_indices[j]
                            similarities.append(filtered_similarity[idx1][idx2])
                    
                    avg_similarity = np.mean(similarities) if similarities else 0.0
                    
                    # Choose representative bookmark (first one for now)
                    representative = bookmark_group[0]
                    
                    duplicate_group = DuplicateGroup(
                        group_id=group_id,
                        bookmarks=bookmark_group,
                        similarity_scores=[avg_similarity] * len(bookmark_group),
                        group_type=group_type,
                        representative_bookmark=representative
                    )
                    
                    duplicate_groups.append(duplicate_group)
                    group_id += 1
            
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Failed to find duplicate groups: {e}")
            return []
    
    def recommend_similar_bookmarks(self, 
                                   user_bookmarks: List[Dict[str, Any]], 
                                   candidate_bookmarks: List[Dict[str, Any]], 
                                   limit: int = 20) -> List[SimilarityResult]:
        """Recommend bookmarks based on user's existing bookmarks"""
        
        try:
            logger.info(f"Generating recommendations from {len(candidate_bookmarks)} candidates")
            
            # Compute embeddings for all bookmarks
            all_bookmarks = user_bookmarks + candidate_bookmarks
            self.compute_bookmark_embeddings(all_bookmarks)
            
            # Get user bookmark embeddings
            user_ids = [b.get('id') for b in user_bookmarks if b.get('id') is not None]
            user_embeddings = [self.bookmark_embeddings[bid] for bid in user_ids if bid in self.bookmark_embeddings]
            
            if not user_embeddings:
                logger.warning("No valid user bookmark embeddings found")
                return []
            
            # Compute user profile as average of user bookmark embeddings
            user_profile = np.mean(user_embeddings, axis=0)
            
            # Score candidate bookmarks
            recommendations = []
            candidate_ids = {b.get('id') for b in candidate_bookmarks if b.get('id') is not None}
            
            for bookmark in candidate_bookmarks:
                bookmark_id = bookmark.get('id')
                
                if bookmark_id not in self.bookmark_embeddings or bookmark_id in user_ids:
                    continue
                
                # Compute similarity to user profile
                candidate_embedding = self.bookmark_embeddings[bookmark_id]
                similarity = cosine_similarity(
                    user_profile.reshape(1, -1),
                    candidate_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity >= self.recommendation_threshold:
                    recommendations.append(SimilarityResult(
                        bookmark_id=bookmark_id,
                        similarity_score=float(similarity),
                        title=bookmark.get('name', ''),
                        url=bookmark.get('url', ''),
                        description=bookmark.get('description', ''),
                        match_type='recommendation'
                    ))
            
            # Sort by similarity score
            recommendations.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    def save_embeddings(self, file_path: Optional[str] = None) -> bool:
        """Save computed embeddings to disk"""
        try:
            if file_path is None:
                file_path = self.cache_dir / 'bookmark_embeddings.pkl'
            else:
                file_path = Path(file_path)
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(self.bookmark_embeddings, f)
            
            logger.info(f"Saved {len(self.bookmark_embeddings)} embeddings to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
            return False
    
    def load_embeddings(self, file_path: Optional[str] = None) -> bool:
        """Load embeddings from disk"""
        try:
            if file_path is None:
                file_path = self.cache_dir / 'bookmark_embeddings.pkl'
            else:
                file_path = Path(file_path)
            
            if not file_path.exists():
                logger.info("No saved embeddings found")
                return False
            
            with open(file_path, 'rb') as f:
                self.bookmark_embeddings = pickle.load(f)
            
            logger.info(f"Loaded {len(self.bookmark_embeddings)} embeddings from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return False
    
    def get_similarity_stats(self) -> Dict[str, Any]:
        """Get similarity engine statistics"""
        return {
            'model_name': self.model_name,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'model_loaded': self.model is not None,
            'cached_embeddings': len(self.bookmark_embeddings),
            'thresholds': {
                'duplicate': self.duplicate_threshold,
                'near_duplicate': self.near_duplicate_threshold,
                'similar': self.similar_threshold,
                'recommendation': self.recommendation_threshold
            },
            'settings': {
                'max_text_length': self.max_text_length,
                'batch_size': self.batch_size
            }
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache"""
        self.bookmark_embeddings.clear()
        self.embeddings_cache.clear()
        logger.info("Embedding cache cleared")