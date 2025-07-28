"""Clustering engine for bookmark organization using scikit-learn"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils.logging_utils import get_logger
from utils.text_utils import TextUtils
from utils.url_utils import UrlUtils

logger = get_logger(__name__)


@dataclass
class BookmarkCluster:
    """Represents a cluster of bookmarks"""
    cluster_id: int
    bookmarks: List[int]  # Bookmark indices
    centroid_keywords: List[str]
    suggested_collection_name: str
    coherence_score: float
    size: int
    dominant_domains: List[str]
    common_tags: List[str]


@dataclass
class DomainCluster:
    """Represents a cluster based on domain patterns"""
    cluster_id: int
    domains: List[str]
    bookmark_count: int
    suggested_category: str
    confidence: float


@dataclass
class ClusteringResult:
    """Result of clustering analysis"""
    clusters: List[BookmarkCluster]
    optimal_cluster_count: int
    silhouette_score: float
    total_bookmarks: int
    clustered_bookmarks: int
    noise_bookmarks: int
    algorithm_used: str


class ClusteringEngine:
    """Advanced clustering engine for bookmark organization"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize clustering engine"""
        self.config = config
        self.ai_config = config.get('ai', {})
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for clustering. Install with: pip install scikit-learn")
        
        # Clustering parameters
        self.max_clusters = self.ai_config.get('max_clusters', 50)
        self.min_cluster_size = 3
        self.max_features = 1000
        
        # Models
        self.vectorizer = None
        self.scaler = StandardScaler()
        
        # Results cache
        self.clustering_cache = {}
        
        logger.info("Clustering engine initialized")
    
    def cluster_by_content_similarity(self, bookmarks: List[Dict[str, Any]]) -> ClusteringResult:
        """Cluster bookmarks based on content similarity"""
        
        if len(bookmarks) < self.min_cluster_size:
            logger.warning(f"Insufficient bookmarks for clustering (need at least {self.min_cluster_size})")
            return self._create_empty_result(bookmarks, "insufficient_data")
        
        try:
            logger.info(f"Clustering {len(bookmarks)} bookmarks by content similarity")
            
            # Extract text content for clustering
            documents = []
            valid_indices = []
            
            for idx, bookmark in enumerate(bookmarks):
                title = bookmark.get('name', '')
                content = bookmark.get('content', {}).get('text_content', '') or ''
                url = bookmark.get('url', '')
                
                # Combine text sources
                text = f"{title} {content}".strip()
                
                if len(text) > 10:  # Minimum content threshold
                    documents.append(text)
                    valid_indices.append(idx)
            
            if len(documents) < self.min_cluster_size:
                logger.warning("Insufficient content for clustering")
                return self._create_empty_result(bookmarks, "insufficient_content")
            
            # Vectorize documents
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # Find optimal number of clusters
            optimal_k = self._find_optimal_cluster_count(tfidf_matrix)
            
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=optimal_k,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
            
            # Create cluster objects
            clusters = self._create_content_clusters(
                bookmarks, valid_indices, cluster_labels, optimal_k
            )
            
            result = ClusteringResult(
                clusters=clusters,
                optimal_cluster_count=optimal_k,
                silhouette_score=silhouette_avg,
                total_bookmarks=len(bookmarks),
                clustered_bookmarks=len(valid_indices),
                noise_bookmarks=len(bookmarks) - len(valid_indices),
                algorithm_used="kmeans"
            )
            
            logger.info(f"Content clustering completed: {len(clusters)} clusters, silhouette score: {silhouette_avg:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Content clustering failed: {e}")
            return self._create_empty_result(bookmarks, "error")
    
    def cluster_by_domain_patterns(self, bookmarks: List[Dict[str, Any]]) -> List[DomainCluster]:
        """Cluster bookmarks based on domain patterns"""
        
        try:
            logger.info(f"Clustering {len(bookmarks)} bookmarks by domain patterns")
            
            # Extract domains and group bookmarks
            domain_bookmarks = defaultdict(list)
            
            for idx, bookmark in enumerate(bookmarks):
                url = bookmark.get('url', '')
                if url:
                    domain = UrlUtils.extract_domain(url)
                    if domain:
                        domain_bookmarks[domain].append(idx)
            
            # Create domain clusters
            clusters = []
            cluster_id = 0
            
            for domain, bookmark_indices in domain_bookmarks.items():
                if len(bookmark_indices) >= 2:  # At least 2 bookmarks per domain cluster
                    
                    # Analyze domain characteristics
                    category = self._classify_domain_category(domain, bookmarks, bookmark_indices)
                    confidence = min(1.0, len(bookmark_indices) / 10.0)  # Scale confidence by count
                    
                    cluster = DomainCluster(
                        cluster_id=cluster_id,
                        domains=[domain],
                        bookmark_count=len(bookmark_indices),
                        suggested_category=category,
                        confidence=confidence
                    )
                    
                    clusters.append(cluster)
                    cluster_id += 1
            
            # Sort by bookmark count (largest first)
            clusters.sort(key=lambda c: c.bookmark_count, reverse=True)
            
            logger.info(f"Domain clustering completed: {len(clusters)} domain clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Domain clustering failed: {e}")
            return []
    
    def cluster_by_tags(self, bookmarks: List[Dict[str, Any]]) -> List[BookmarkCluster]:
        """Cluster bookmarks based on tag similarity"""
        
        try:
            logger.info(f"Clustering {len(bookmarks)} bookmarks by tag similarity")
            
            # Extract tag vectors
            all_tags = set()
            bookmark_tags = []
            
            for bookmark in bookmarks:
                tags = bookmark.get('tags', [])
                tag_names = []
                
                for tag in tags:
                    if isinstance(tag, dict):
                        tag_name = tag.get('name', '')
                    else:
                        tag_name = str(tag)
                    
                    if tag_name:
                        tag_names.append(tag_name.lower())
                        all_tags.add(tag_name.lower())
                
                bookmark_tags.append(tag_names)
            
            if not all_tags:
                logger.warning("No tags found for clustering")
                return []
            
            # Create tag vectors (binary encoding)
            tag_list = sorted(list(all_tags))
            tag_vectors = []
            
            for tags in bookmark_tags:
                vector = [1 if tag in tags else 0 for tag in tag_list]
                tag_vectors.append(vector)
            
            tag_matrix = np.array(tag_vectors)
            
            # Use DBSCAN for tag-based clustering (handles varying cluster sizes better)
            dbscan = DBSCAN(eps=0.5, min_samples=2, metric='jaccard')
            cluster_labels = dbscan.fit_predict(tag_matrix)
            
            # Create clusters
            clusters = self._create_tag_clusters(bookmarks, cluster_labels, tag_list, tag_vectors)
            
            logger.info(f"Tag clustering completed: {len(clusters)} clusters")
            return clusters
            
        except Exception as e:
            logger.error(f"Tag clustering failed: {e}")
            return []
    
    def _find_optimal_cluster_count(self, data_matrix) -> int:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        
        n_samples = data_matrix.shape[0]
        max_k = min(self.max_clusters, n_samples // 2)
        min_k = 2
        
        if max_k <= min_k:
            return min_k
        
        try:
            # Test different cluster counts
            silhouette_scores = []
            inertias = []
            k_range = range(min_k, min(max_k + 1, 11))  # Limit to 10 for performance
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
                cluster_labels = kmeans.fit_predict(data_matrix)
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(data_matrix, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                inertias.append(kmeans.inertia_)
            
            # Find best k based on silhouette score
            best_k_idx = np.argmax(silhouette_scores)
            optimal_k = list(k_range)[best_k_idx]
            
            logger.debug(f"Optimal cluster count: {optimal_k} (silhouette: {silhouette_scores[best_k_idx]:.3f})")
            return optimal_k
            
        except Exception as e:
            logger.warning(f"Failed to find optimal cluster count: {e}")
            return min(5, max_k)  # Fallback
    
    def _create_content_clusters(self, 
                               bookmarks: List[Dict[str, Any]], 
                               valid_indices: List[int],
                               cluster_labels: np.ndarray,
                               n_clusters: int) -> List[BookmarkCluster]:
        """Create BookmarkCluster objects from clustering results"""
        
        clusters = []
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in range(n_clusters):
            # Get bookmarks in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_bookmark_indices = [valid_indices[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
            
            if len(cluster_bookmark_indices) < 2:  # Skip tiny clusters
                continue
            
            # Extract cluster characteristics
            cluster_bookmarks = [bookmarks[i] for i in cluster_bookmark_indices]
            
            # Get centroid keywords
            centroid_keywords = self._extract_cluster_keywords(cluster_bookmarks)
            
            # Suggest collection name
            collection_name = self._suggest_collection_name(cluster_bookmarks, centroid_keywords)
            
            # Calculate coherence score
            coherence_score = self._calculate_cluster_coherence(cluster_bookmarks)
            
            # Extract dominant domains
            domains = []
            for bookmark in cluster_bookmarks:
                url = bookmark.get('url', '')
                if url:
                    domain = UrlUtils.extract_domain(url)
                    if domain:
                        domains.append(domain)
            
            domain_counts = Counter(domains)
            dominant_domains = [domain for domain, count in domain_counts.most_common(3)]
            
            # Extract common tags
            all_tags = []
            for bookmark in cluster_bookmarks:
                tags = bookmark.get('tags', [])
                for tag in tags:
                    if isinstance(tag, dict):
                        tag_name = tag.get('name', '')
                    else:
                        tag_name = str(tag)
                    if tag_name:
                        all_tags.append(tag_name)
            
            tag_counts = Counter(all_tags)
            common_tags = [tag for tag, count in tag_counts.most_common(5)]
            
            cluster = BookmarkCluster(
                cluster_id=cluster_id,
                bookmarks=cluster_bookmark_indices,
                centroid_keywords=centroid_keywords,
                suggested_collection_name=collection_name,
                coherence_score=coherence_score,
                size=len(cluster_bookmark_indices),
                dominant_domains=dominant_domains,
                common_tags=common_tags
            )
            
            clusters.append(cluster)
        
        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: c.size, reverse=True)
        return clusters
    
    def _create_tag_clusters(self, 
                           bookmarks: List[Dict[str, Any]], 
                           cluster_labels: np.ndarray,
                           tag_list: List[str],
                           tag_vectors: List[List[int]]) -> List[BookmarkCluster]:
        """Create clusters based on tag similarity"""
        
        clusters = []
        unique_labels = set(cluster_labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
            
            # Get bookmarks in this cluster
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            
            if len(cluster_indices) < 2:
                continue
            
            cluster_bookmarks = [bookmarks[i] for i in cluster_indices]
            
            # Find common tags for this cluster
            cluster_tag_vectors = [tag_vectors[i] for i in cluster_indices]
            tag_sums = np.sum(cluster_tag_vectors, axis=0)
            
            # Get tags that appear in at least half the bookmarks in the cluster
            threshold = len(cluster_indices) / 2
            common_tag_indices = [i for i, count in enumerate(tag_sums) if count >= threshold]
            common_tags = [tag_list[i] for i in common_tag_indices]
            
            # Use common tags as keywords
            centroid_keywords = common_tags[:10]
            
            # Suggest collection name based on tags
            collection_name = self._suggest_collection_name_from_tags(common_tags)
            
            cluster = BookmarkCluster(
                cluster_id=cluster_id,
                bookmarks=cluster_indices,
                centroid_keywords=centroid_keywords,
                suggested_collection_name=collection_name,
                coherence_score=len(common_tags) / len(tag_list),  # Simple coherence measure
                size=len(cluster_indices),
                dominant_domains=[],
                common_tags=common_tags
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _extract_cluster_keywords(self, cluster_bookmarks: List[Dict[str, Any]]) -> List[str]:
        """Extract representative keywords for a cluster"""
        
        # Combine all text from cluster bookmarks
        all_text = []
        for bookmark in cluster_bookmarks:
            title = bookmark.get('name', '')
            content = bookmark.get('content', {}).get('text_content', '') or ''
            all_text.append(f"{title} {content}")
        
        combined_text = " ".join(all_text)
        
        # Extract keywords
        keywords = TextUtils.extract_keywords(combined_text, max_keywords=10)
        return keywords
    
    def _suggest_collection_name(self, 
                                cluster_bookmarks: List[Dict[str, Any]], 
                                keywords: List[str]) -> str:
        """Suggest a collection name for a cluster"""
        
        # Try to find patterns in URLs
        domains = []
        for bookmark in cluster_bookmarks:
            url = bookmark.get('url', '')
            if url:
                domain = UrlUtils.extract_domain(url)
                if domain:
                    domains.append(domain)
        
        domain_counts = Counter(domains)
        
        # If dominated by one domain, use domain-based name
        if domain_counts and domain_counts.most_common(1)[0][1] >= len(cluster_bookmarks) * 0.6:
            dominant_domain = domain_counts.most_common(1)[0][0]
            
            # Map common domains to readable names
            domain_names = {
                'github.com': 'GitHub Projects',
                'youtube.com': 'YouTube Videos',
                'reddit.com': 'Reddit Posts',
                'stackoverflow.com': 'Stack Overflow',
                'medium.com': 'Medium Articles',
                'dev.to': 'Dev.to Articles'
            }
            
            if dominant_domain in domain_names:
                return domain_names[dominant_domain]
            else:
                return f"{dominant_domain.replace('.com', '').title()} Resources"
        
        # Use keywords to suggest name
        if keywords:
            # Look for technology/topic patterns
            tech_keywords = ['ai', 'ml', 'python', 'javascript', 'react', 'vue', 'docker', 'kubernetes']
            topic_keywords = ['gaming', 'music', 'art', 'design', 'news', 'tutorial', 'tool']
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in tech_keywords:
                    return f"{keyword.title()} Resources"
                elif keyword_lower in topic_keywords:
                    return f"{keyword.title()} Collection"
            
            # Use first keyword as base
            return f"{keywords[0].title()} Resources"
        
        return f"Cluster {len(cluster_bookmarks)} Items"
    
    def _suggest_collection_name_from_tags(self, common_tags: List[str]) -> str:
        """Suggest collection name based on common tags"""
        
        if not common_tags:
            return "Tagged Collection"
        
        # Use most descriptive tag
        priority_tags = ['ai', 'gaming', 'development', 'music', 'art', 'news', 'tutorial']
        
        for tag in common_tags:
            if tag.lower() in priority_tags:
                return f"{tag.title()} Collection"
        
        # Use first tag
        return f"{common_tags[0].title()} Collection"
    
    def _calculate_cluster_coherence(self, cluster_bookmarks: List[Dict[str, Any]]) -> float:
        """Calculate coherence score for a cluster"""
        
        # Simple coherence based on tag overlap
        all_tags = []
        bookmark_tag_sets = []
        
        for bookmark in cluster_bookmarks:
            tags = bookmark.get('tags', [])
            tag_set = set()
            
            for tag in tags:
                if isinstance(tag, dict):
                    tag_name = tag.get('name', '')
                else:
                    tag_name = str(tag)
                
                if tag_name:
                    tag_name = tag_name.lower()
                    all_tags.append(tag_name)
                    tag_set.add(tag_name)
            
            bookmark_tag_sets.append(tag_set)
        
        if not all_tags:
            return 0.5  # Neutral score if no tags
        
        # Calculate average pairwise tag similarity
        similarities = []
        for i in range(len(bookmark_tag_sets)):
            for j in range(i + 1, len(bookmark_tag_sets)):
                set1, set2 = bookmark_tag_sets[i], bookmark_tag_sets[j]
                if set1 or set2:
                    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
                    similarities.append(jaccard)
        
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.5
    
    def _classify_domain_category(self, 
                                 domain: str, 
                                 bookmarks: List[Dict[str, Any]], 
                                 bookmark_indices: List[int]) -> str:
        """Classify domain into a category"""
        
        # Simple domain classification
        domain_lower = domain.lower()
        
        if any(keyword in domain_lower for keyword in ['github', 'gitlab', 'bitbucket']):
            return 'Development'
        elif any(keyword in domain_lower for keyword in ['youtube', 'twitch', 'netflix']):
            return 'Entertainment'
        elif any(keyword in domain_lower for keyword in ['reddit', 'twitter', 'facebook']):
            return 'Social Media'
        elif any(keyword in domain_lower for keyword in ['aws', 'azure', 'cloud', 'docker']):
            return 'Cloud & Infrastructure'
        elif any(keyword in domain_lower for keyword in ['game', 'steam', 'itch']):
            return 'Gaming'
        else:
            return 'General'
    
    def _create_empty_result(self, bookmarks: List[Dict[str, Any]], reason: str) -> ClusteringResult:
        """Create empty clustering result"""
        return ClusteringResult(
            clusters=[],
            optimal_cluster_count=0,
            silhouette_score=0.0,
            total_bookmarks=len(bookmarks),
            clustered_bookmarks=0,
            noise_bookmarks=len(bookmarks),
            algorithm_used=f"none_{reason}"
        )
    
    def get_clustering_stats(self) -> Dict[str, Any]:
        """Get clustering engine statistics"""
        
        return {
            'sklearn_available': SKLEARN_AVAILABLE,
            'max_clusters': self.max_clusters,
            'min_cluster_size': self.min_cluster_size,
            'max_features': self.max_features,
            'has_vectorizer': self.vectorizer is not None,
            'cache_size': len(self.clustering_cache),
            'vectorizer_features': len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 0
        }