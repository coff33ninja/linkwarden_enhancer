"""Network analysis for bookmark relationships using NetworkX"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from urllib.parse import urlparse
import logging

try:
    import networkx as nx
    from networkx.algorithms import community
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from ..utils.logging_utils import get_logger
    from ..utils.url_utils import UrlUtils
    from ..utils.text_utils import TextUtils
    logger = get_logger(__name__)
except ImportError:
    # Fallback for standalone testing
    import logging
    logger = logging.getLogger(__name__)
    
    class UrlUtils:
        @staticmethod
        def extract_domain(url):
            from urllib.parse import urlparse
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
            except:
                return ""
    
    class TextUtils:
        @staticmethod
        def clean_text(text):
            return text.strip() if text else ""


@dataclass
class BookmarkNode:
    """Represents a bookmark node in the network"""
    bookmark_id: int
    url: str
    title: str
    domain: str
    tags: List[str]
    collection_id: Optional[int] = None
    content_hash: Optional[str] = None
    
    
@dataclass
class Community:
    """Represents a detected community of bookmarks"""
    community_id: int
    bookmark_ids: List[int]
    dominant_domains: List[str]
    common_tags: List[str]
    suggested_collection_name: str
    cohesion_score: float
    size: int
    hub_bookmarks: List[int]  # Most connected bookmarks in community


@dataclass
class HubBookmark:
    """Represents an important hub bookmark"""
    bookmark_id: int
    centrality_score: float
    degree: int
    betweenness_centrality: float
    pagerank_score: float
    influence_type: str  # 'domain_hub', 'tag_hub', 'content_hub'
    connected_bookmarks: List[int]


@dataclass
class CollectionStructure:
    """Optimized collection structure based on network analysis"""
    suggested_collections: List[Dict[str, Any]]
    collection_hierarchy: Dict[int, List[int]]  # parent -> children
    bookmark_assignments: Dict[int, int]  # bookmark_id -> collection_id
    optimization_score: float
    reasoning: str


@dataclass
class NetworkAnalysisResult:
    """Complete result of network analysis"""
    total_nodes: int
    total_edges: int
    communities: List[Community]
    hub_bookmarks: List[HubBookmark]
    suggested_structure: CollectionStructure
    network_metrics: Dict[str, float]
    analysis_time: float


class NetworkAnalyzer:
    """Network analysis for bookmark relationships using NetworkX"""
    
    def __init__(self, similarity_threshold: float = 0.3, min_community_size: int = 3):
        """
        Initialize the network analyzer
        
        Args:
            similarity_threshold: Minimum similarity score to create edges
            min_community_size: Minimum size for detected communities
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for network analysis. Install with: pip install networkx")
            
        self.similarity_threshold = similarity_threshold
        self.min_community_size = min_community_size
        self.graph = nx.Graph()
        self.bookmark_nodes = {}
        self.url_utils = UrlUtils()
        self.text_utils = TextUtils()
        
    def build_bookmark_network(self, bookmarks: List[dict]) -> nx.Graph:
        """
        Build a network graph from bookmarks based on various relationships
        
        Args:
            bookmarks: List of bookmark dictionaries
            
        Returns:
            NetworkX graph representing bookmark relationships
        """
        logger.info(f"Building bookmark network from {len(bookmarks)} bookmarks")
        
        # Clear previous graph
        self.graph.clear()
        self.bookmark_nodes.clear()
        
        # Create nodes for each bookmark
        for i, bookmark in enumerate(bookmarks):
            node = self._create_bookmark_node(i, bookmark)
            self.bookmark_nodes[i] = node
            
            # Add node to graph with attributes
            self.graph.add_node(i, **{
                'url': node.url,
                'title': node.title,
                'domain': node.domain,
                'tags': node.tags,
                'collection_id': node.collection_id
            })
        
        # Create edges based on relationships
        self._create_domain_edges()
        self._create_tag_edges()
        self._create_content_similarity_edges(bookmarks)
        self._create_collection_edges()
        
        logger.info(f"Created network with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _create_bookmark_node(self, index: int, bookmark: dict) -> BookmarkNode:
        """Create a BookmarkNode from bookmark data"""
        url = bookmark.get('url', '')
        title = bookmark.get('name', bookmark.get('title', ''))
        domain = self.url_utils.extract_domain(url)
        tags = [tag.get('name', '') for tag in bookmark.get('tags', [])]
        collection_id = bookmark.get('collectionId')
        
        return BookmarkNode(
            bookmark_id=index,
            url=url,
            title=title,
            domain=domain,
            tags=tags,
            collection_id=collection_id,
            content_hash=self._generate_content_hash(title, url, tags)
        )
    
    def _generate_content_hash(self, title: str, url: str, tags: List[str]) -> str:
        """Generate a content hash for similarity comparison"""
        content = f"{title} {url} {' '.join(tags)}".lower()
        return str(hash(content))
    
    def _create_domain_edges(self):
        """Create edges between bookmarks from the same domain"""
        domain_groups = defaultdict(list)
        
        for node_id, node in self.bookmark_nodes.items():
            if node.domain:
                domain_groups[node.domain].append(node_id)
        
        # Connect bookmarks within the same domain
        for domain, node_ids in domain_groups.items():
            if len(node_ids) > 1:
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        self.graph.add_edge(node_ids[i], node_ids[j], 
                                          relationship='domain', 
                                          weight=0.5,
                                          domain=domain)
    
    def _create_tag_edges(self):
        """Create edges between bookmarks sharing tags"""
        for i in range(len(self.bookmark_nodes)):
            for j in range(i + 1, len(self.bookmark_nodes)):
                node_i = self.bookmark_nodes[i]
                node_j = self.bookmark_nodes[j]
                
                # Calculate tag similarity
                tags_i = set(node_i.tags)
                tags_j = set(node_j.tags)
                
                if tags_i and tags_j:
                    intersection = tags_i.intersection(tags_j)
                    union = tags_i.union(tags_j)
                    
                    if intersection and len(intersection) / len(union) >= 0.2:  # Jaccard similarity
                        weight = len(intersection) / len(union)
                        self.graph.add_edge(i, j, 
                                          relationship='tags', 
                                          weight=weight,
                                          shared_tags=list(intersection))
    
    def _create_content_similarity_edges(self, bookmarks: List[dict]):
        """Create edges based on content similarity"""
        # Simple content similarity based on title and description
        for i in range(len(bookmarks)):
            for j in range(i + 1, len(bookmarks)):
                bookmark_i = bookmarks[i]
                bookmark_j = bookmarks[j]
                
                title_i = bookmark_i.get('name', '').lower()
                title_j = bookmark_j.get('name', '').lower()
                
                desc_i = bookmark_i.get('description', '').lower()
                desc_j = bookmark_j.get('description', '').lower()
                
                # Simple text similarity
                similarity = self._calculate_text_similarity(
                    f"{title_i} {desc_i}", 
                    f"{title_j} {desc_j}"
                )
                
                if similarity >= self.similarity_threshold:
                    self.graph.add_edge(i, j, 
                                      relationship='content', 
                                      weight=similarity,
                                      similarity_score=similarity)
    
    def _create_collection_edges(self):
        """Create edges between bookmarks in the same collection"""
        collection_groups = defaultdict(list)
        
        for node_id, node in self.bookmark_nodes.items():
            if node.collection_id:
                collection_groups[node.collection_id].append(node_id)
        
        # Connect bookmarks within the same collection
        for collection_id, node_ids in collection_groups.items():
            if len(node_ids) > 1:
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        self.graph.add_edge(node_ids[i], node_ids[j], 
                                          relationship='collection', 
                                          weight=0.8,
                                          collection_id=collection_id)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def find_bookmark_communities(self) -> List[Community]:
        """
        Detect communities in the bookmark network using various algorithms
        
        Returns:
            List of detected communities
        """
        logger.info("Detecting bookmark communities")
        
        if self.graph.number_of_nodes() < self.min_community_size:
            logger.warning("Not enough nodes for community detection")
            return []
        
        communities = []
        
        try:
            # Use Louvain algorithm for community detection
            partition = community.louvain_communities(self.graph, resolution=1.0)
            
            for i, community_nodes in enumerate(partition):
                if len(community_nodes) >= self.min_community_size:
                    comm = self._analyze_community(i, list(community_nodes))
                    communities.append(comm)
            
            logger.info(f"Detected {len(communities)} communities")
            
        except Exception as e:
            logger.error(f"Error in community detection: {e}")
            # Fallback to simple clustering based on connected components
            communities = self._fallback_community_detection()
        
        return communities
    
    def _analyze_community(self, community_id: int, node_ids: List[int]) -> Community:
        """Analyze a detected community to extract characteristics"""
        domains = []
        tags = []
        
        for node_id in node_ids:
            node = self.bookmark_nodes[node_id]
            if node.domain:
                domains.append(node.domain)
            tags.extend(node.tags)
        
        # Find dominant domains and common tags
        domain_counts = Counter(domains)
        tag_counts = Counter(tags)
        
        dominant_domains = [domain for domain, count in domain_counts.most_common(3)]
        common_tags = [tag for tag, count in tag_counts.most_common(5) if count > 1]
        
        # Calculate cohesion score based on internal connections
        subgraph = self.graph.subgraph(node_ids)
        possible_edges = len(node_ids) * (len(node_ids) - 1) / 2
        actual_edges = subgraph.number_of_edges()
        cohesion_score = actual_edges / possible_edges if possible_edges > 0 else 0
        
        # Find hub bookmarks within community
        hub_bookmarks = self._find_community_hubs(node_ids)
        
        # Suggest collection name
        suggested_name = self._suggest_collection_name(dominant_domains, common_tags)
        
        return Community(
            community_id=community_id,
            bookmark_ids=node_ids,
            dominant_domains=dominant_domains,
            common_tags=common_tags,
            suggested_collection_name=suggested_name,
            cohesion_score=cohesion_score,
            size=len(node_ids),
            hub_bookmarks=hub_bookmarks
        )
    
    def _fallback_community_detection(self) -> List[Community]:
        """Fallback community detection using connected components"""
        communities = []
        connected_components = list(nx.connected_components(self.graph))
        
        for i, component in enumerate(connected_components):
            if len(component) >= self.min_community_size:
                comm = self._analyze_community(i, list(component))
                communities.append(comm)
        
        return communities
    
    def _find_community_hubs(self, node_ids: List[int]) -> List[int]:
        """Find hub nodes within a community"""
        subgraph = self.graph.subgraph(node_ids)
        
        # Calculate degree centrality within the community
        centrality = nx.degree_centrality(subgraph)
        
        # Return top 3 most central nodes
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in sorted_nodes[:3]]
    
    def _suggest_collection_name(self, domains: List[str], tags: List[str]) -> str:
        """Suggest a collection name based on community characteristics"""
        if domains:
            # Use most common domain as base
            main_domain = domains[0]
            if 'github' in main_domain:
                return "Development Resources"
            elif any(gaming in main_domain for gaming in ['steam', 'itch', 'twitch', 'gaming']):
                return "Gaming"
            elif any(ai in main_domain for ai in ['openai', 'huggingface', 'anthropic']):
                return "AI & Machine Learning"
            else:
                return f"{main_domain.replace('.com', '').replace('.org', '').title()} Resources"
        
        if tags:
            # Use most common tag
            main_tag = tags[0]
            return f"{main_tag.title()} Collection"
        
        return "Uncategorized Collection"
    
    def identify_hub_bookmarks(self) -> List[HubBookmark]:
        """
        Identify important hub bookmarks based on various centrality measures
        
        Returns:
            List of hub bookmarks sorted by importance
        """
        logger.info("Identifying hub bookmarks")
        
        if self.graph.number_of_nodes() == 0:
            return []
        
        hub_bookmarks = []
        
        try:
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(self.graph)
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            pagerank = nx.pagerank(self.graph)
            
            # Identify hubs based on different criteria
            for node_id in self.graph.nodes():
                node = self.bookmark_nodes[node_id]
                degree = self.graph.degree(node_id)
                
                # Only consider nodes with significant connections
                if degree >= 2:
                    # Determine hub type
                    influence_type = self._determine_influence_type(node_id, node)
                    
                    # Get connected bookmarks
                    connected = list(self.graph.neighbors(node_id))
                    
                    hub = HubBookmark(
                        bookmark_id=node_id,
                        centrality_score=degree_centrality[node_id],
                        degree=degree,
                        betweenness_centrality=betweenness_centrality[node_id],
                        pagerank_score=pagerank[node_id],
                        influence_type=influence_type,
                        connected_bookmarks=connected
                    )
                    
                    hub_bookmarks.append(hub)
            
            # Sort by combined importance score
            hub_bookmarks.sort(key=lambda h: (h.centrality_score + h.pagerank_score + h.betweenness_centrality), reverse=True)
            
            logger.info(f"Identified {len(hub_bookmarks)} hub bookmarks")
            
        except Exception as e:
            logger.error(f"Error identifying hub bookmarks: {e}")
        
        return hub_bookmarks[:20]  # Return top 20 hubs
    
    def _determine_influence_type(self, node_id: int, node: BookmarkNode) -> str:
        """Determine the type of influence a hub bookmark has"""
        neighbors = list(self.graph.neighbors(node_id))
        
        # Check if it's a domain hub (connects many bookmarks from same domain)
        same_domain_count = sum(1 for neighbor_id in neighbors 
                               if self.bookmark_nodes[neighbor_id].domain == node.domain)
        
        if same_domain_count >= len(neighbors) * 0.7:
            return 'domain_hub'
        
        # Check if it's a tag hub (connects bookmarks with similar tags)
        node_tags = set(node.tags)
        tag_overlap_count = 0
        for neighbor_id in neighbors:
            neighbor_tags = set(self.bookmark_nodes[neighbor_id].tags)
            if node_tags.intersection(neighbor_tags):
                tag_overlap_count += 1
        
        if tag_overlap_count >= len(neighbors) * 0.6:
            return 'tag_hub'
        
        return 'content_hub'
    
    def suggest_collection_structure(self, communities: List[Community]) -> CollectionStructure:
        """
        Suggest an optimized collection structure based on network analysis
        
        Args:
            communities: Detected communities from network analysis
            
        Returns:
            Optimized collection structure
        """
        logger.info("Suggesting collection structure based on network analysis")
        
        suggested_collections = []
        bookmark_assignments = {}
        collection_hierarchy = {}
        
        # Create collections based on communities
        for i, community in enumerate(communities):
            collection = {
                'id': i + 1,
                'name': community.suggested_collection_name,
                'description': f"Auto-generated collection based on network analysis. "
                             f"Contains {community.size} bookmarks with {community.cohesion_score:.2f} cohesion score.",
                'bookmark_count': community.size,
                'dominant_domains': community.dominant_domains,
                'common_tags': community.common_tags,
                'cohesion_score': community.cohesion_score
            }
            
            suggested_collections.append(collection)
            
            # Assign bookmarks to collections
            for bookmark_id in community.bookmark_ids:
                bookmark_assignments[bookmark_id] = i + 1
        
        # Create hierarchy based on domain relationships
        domain_collections = defaultdict(list)
        for collection in suggested_collections:
            for domain in collection['dominant_domains']:
                domain_collections[domain].append(collection['id'])
        
        # Group related collections under parent categories
        parent_id = len(suggested_collections) + 1
        for domain, collection_ids in domain_collections.items():
            if len(collection_ids) > 1:
                # Create parent collection
                parent_name = self._get_parent_category_name(domain)
                parent_collection = {
                    'id': parent_id,
                    'name': parent_name,
                    'description': f"Parent category for {domain}-related collections",
                    'is_parent': True
                }
                suggested_collections.append(parent_collection)
                collection_hierarchy[parent_id] = collection_ids
                parent_id += 1
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(communities, suggested_collections)
        
        reasoning = (f"Created {len(suggested_collections)} collections based on network analysis. "
                    f"Detected {len(communities)} communities with average cohesion score of "
                    f"{np.mean([c.cohesion_score for c in communities]):.2f}.")
        
        return CollectionStructure(
            suggested_collections=suggested_collections,
            collection_hierarchy=collection_hierarchy,
            bookmark_assignments=bookmark_assignments,
            optimization_score=optimization_score,
            reasoning=reasoning
        )
    
    def _get_parent_category_name(self, domain: str) -> str:
        """Get parent category name based on domain"""
        if 'github' in domain or 'gitlab' in domain:
            return "Development"
        elif any(gaming in domain for gaming in ['steam', 'itch', 'twitch', 'gaming']):
            return "Gaming"
        elif any(ai in domain for ai in ['openai', 'huggingface', 'anthropic']):
            return "AI & Machine Learning"
        elif any(social in domain for social in ['twitter', 'facebook', 'reddit']):
            return "Social Media"
        else:
            return f"{domain.replace('.com', '').replace('.org', '').title()} Category"
    
    def _calculate_optimization_score(self, communities: List[Community], collections: List[dict]) -> float:
        """Calculate how well the suggested structure optimizes bookmark organization"""
        if not communities:
            return 0.0
        
        # Score based on community cohesion and size distribution
        cohesion_scores = [c.cohesion_score for c in communities]
        avg_cohesion = np.mean(cohesion_scores)
        
        # Penalty for very small or very large collections
        sizes = [c.size for c in communities]
        size_variance = np.var(sizes)
        size_penalty = min(size_variance / 100, 0.3)  # Cap penalty at 0.3
        
        optimization_score = avg_cohesion - size_penalty
        return max(0.0, min(1.0, optimization_score))  # Clamp between 0 and 1
    
    def analyze_network(self, bookmarks: List[dict]) -> NetworkAnalysisResult:
        """
        Perform complete network analysis on bookmarks
        
        Args:
            bookmarks: List of bookmark dictionaries
            
        Returns:
            Complete network analysis result
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting network analysis of {len(bookmarks)} bookmarks")
        
        # Build the network
        graph = self.build_bookmark_network(bookmarks)
        
        # Detect communities
        communities = self.find_bookmark_communities()
        
        # Identify hub bookmarks
        hub_bookmarks = self.identify_hub_bookmarks()
        
        # Suggest collection structure
        suggested_structure = self.suggest_collection_structure(communities)
        
        # Calculate network metrics
        network_metrics = self._calculate_network_metrics()
        
        analysis_time = time.time() - start_time
        
        result = NetworkAnalysisResult(
            total_nodes=graph.number_of_nodes(),
            total_edges=graph.number_of_edges(),
            communities=communities,
            hub_bookmarks=hub_bookmarks,
            suggested_structure=suggested_structure,
            network_metrics=network_metrics,
            analysis_time=analysis_time
        )
        
        logger.info(f"Network analysis completed in {analysis_time:.2f} seconds")
        return result
    
    def _calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate various network metrics"""
        if self.graph.number_of_nodes() == 0:
            return {}
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['density'] = nx.density(self.graph)
            metrics['average_clustering'] = nx.average_clustering(self.graph)
            
            # Connectivity metrics
            if nx.is_connected(self.graph):
                metrics['diameter'] = nx.diameter(self.graph)
                metrics['average_shortest_path_length'] = nx.average_shortest_path_length(self.graph)
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(self.graph), key=len)
                subgraph = self.graph.subgraph(largest_cc)
                if len(largest_cc) > 1:
                    metrics['diameter'] = nx.diameter(subgraph)
                    metrics['average_shortest_path_length'] = nx.average_shortest_path_length(subgraph)
            
            # Centralization metrics
            degree_centrality = nx.degree_centrality(self.graph)
            metrics['max_degree_centrality'] = max(degree_centrality.values()) if degree_centrality else 0
            metrics['average_degree_centrality'] = np.mean(list(degree_centrality.values())) if degree_centrality else 0
            
            # Component analysis
            metrics['number_of_components'] = nx.number_connected_components(self.graph)
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
        
        return metrics
    
    def export_network_data(self, output_path: str, format: str = 'gexf'):
        """
        Export network data for visualization in external tools
        
        Args:
            output_path: Path to save the network file
            format: Export format ('gexf', 'graphml', 'gml')
        """
        try:
            if format.lower() == 'gexf':
                nx.write_gexf(self.graph, output_path)
            elif format.lower() == 'graphml':
                nx.write_graphml(self.graph, output_path)
            elif format.lower() == 'gml':
                nx.write_gml(self.graph, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Network exported to {output_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Error exporting network: {e}")
            raise