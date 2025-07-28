"""Continuous learning system for adaptive intelligence improvement"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field

from utils.logging_utils import get_logger
from utils.text_utils import TextUtils
from utils.url_utils import UrlUtils

logger = get_logger(__name__)


@dataclass
class LearningMetrics:
    """Metrics for tracking learning performance"""
    total_bookmarks_processed: int = 0
    patterns_learned: int = 0
    patterns_updated: int = 0
    accuracy_improvements: float = 0.0
    learning_sessions: int = 0
    last_learning_time: Optional[datetime] = None
    average_learning_time: float = 0.0
    pattern_reliability_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class PatternStrength:
    """Represents the strength and reliability of a learned pattern"""
    pattern_id: str
    strength: float
    confidence: float
    usage_count: int
    success_rate: float
    last_used: datetime
    created_at: datetime
    pattern_type: str  # 'category', 'tag', 'domain', 'content'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningSession:
    """Information about a learning session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    bookmarks_processed: int = 0
    new_patterns: int = 0
    updated_patterns: int = 0
    accuracy_before: float = 0.0
    accuracy_after: float = 0.0
    session_type: str = "incremental"  # 'incremental', 'batch', 'feedback'
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ContinuousLearner:
    """Continuous learning system for adaptive intelligence improvement"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize continuous learner"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Learning state
        self.pattern_strengths = {}  # pattern_id -> PatternStrength
        self.learning_metrics = LearningMetrics()
        self.learning_sessions = []
        self.active_session = None
        
        # Pattern tracking
        self.category_patterns = defaultdict(lambda: defaultdict(float))
        self.tag_patterns = defaultdict(lambda: defaultdict(float))
        self.domain_patterns = defaultdict(lambda: defaultdict(float))
        self.content_patterns = defaultdict(lambda: defaultdict(float))
        
        # Performance tracking
        self.prediction_history = []
        self.accuracy_tracking = defaultdict(list)
        self.pattern_usage_stats = defaultdict(int)
        
        # Configuration
        self.min_pattern_strength = 0.1
        self.max_patterns_per_type = 1000
        self.pattern_decay_rate = 0.95  # Decay unused patterns
        self.learning_rate = 0.1
        
        # Load existing learning data
        self._load_learning_data()
        
        logger.info("Continuous learner initialized")
    
    def start_learning_session(self, session_type: str = "incremental") -> str:
        """Start a new learning session"""
        
        session_id = f"session_{int(time.time())}"
        
        self.active_session = LearningSession(
            session_id=session_id,
            start_time=datetime.now(),
            session_type=session_type
        )
        
        logger.info(f"Started learning session: {session_id} ({session_type})")
        return session_id
    
    def learn_from_new_bookmarks(self, bookmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn from newly added bookmarks"""
        
        if not self.active_session:
            self.start_learning_session("incremental")
        
        session = self.active_session
        learning_results = {
            'session_id': session.session_id,
            'bookmarks_processed': 0,
            'new_patterns_learned': 0,
            'patterns_updated': 0,
            'learning_improvements': [],
            'errors': []
        }
        
        try:
            logger.info(f"Learning from {len(bookmarks)} new bookmarks")
            
            for bookmark in bookmarks:
                try:
                    # Extract bookmark features
                    url = bookmark.get('url', '')
                    title = bookmark.get('name', '')
                    content = bookmark.get('content', {}).get('text_content', '') or ''
                    collection_name = bookmark.get('collection_name', '')
                    tags = bookmark.get('tags', [])
                    
                    if not url:
                        continue
                    
                    # Learn patterns from this bookmark
                    self._learn_category_patterns(url, title, content, collection_name)
                    self._learn_tag_patterns(url, title, content, tags)
                    self._learn_domain_patterns(url, collection_name, tags)
                    self._learn_content_patterns(title, content, tags, collection_name)
                    
                    session.bookmarks_processed += 1
                    learning_results['bookmarks_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to learn from bookmark {bookmark.get('id', 'unknown')}: {e}"
                    session.errors.append(error_msg)
                    learning_results['errors'].append(error_msg)
                    logger.warning(error_msg)
            
            # Update pattern strengths
            new_patterns, updated_patterns = self._update_pattern_strengths()
            session.new_patterns = new_patterns
            session.updated_patterns = updated_patterns
            learning_results['new_patterns_learned'] = new_patterns
            learning_results['patterns_updated'] = updated_patterns
            
            # Calculate learning improvements
            improvements = self._calculate_learning_improvements()
            learning_results['learning_improvements'] = improvements
            
            logger.info(f"Learning completed: {learning_results}")
            return learning_results
            
        except Exception as e:
            error_msg = f"Learning session failed: {e}"
            logger.error(error_msg)
            session.errors.append(error_msg)
            learning_results['errors'].append(error_msg)
            return learning_results
    
    def _learn_category_patterns(self, url: str, title: str, content: str, collection_name: str) -> None:
        """Learn category prediction patterns"""
        
        if not collection_name:
            return
        
        try:
            # Extract features for category learning
            domain = UrlUtils.extract_domain(url)
            path_segments = UrlUtils.extract_path_segments(url)
            keywords = TextUtils.extract_keywords(f"{title} {content}")
            
            # Learn domain -> category associations
            if domain:
                pattern_id = f"domain_category:{domain}"
                self.category_patterns[pattern_id][collection_name] += 1
                self._update_pattern_strength(pattern_id, "category", domain, collection_name)
            
            # Learn path -> category associations
            for segment in path_segments[:3]:  # Top 3 path segments
                if len(segment) > 2:
                    pattern_id = f"path_category:{segment}"
                    self.category_patterns[pattern_id][collection_name] += 1
                    self._update_pattern_strength(pattern_id, "category", segment, collection_name)
            
            # Learn keyword -> category associations
            for keyword in keywords[:5]:  # Top 5 keywords
                pattern_id = f"keyword_category:{keyword}"
                self.category_patterns[pattern_id][collection_name] += 1
                self._update_pattern_strength(pattern_id, "category", keyword, collection_name)
                
        except Exception as e:
            logger.warning(f"Failed to learn category patterns: {e}")
    
    def _learn_tag_patterns(self, url: str, title: str, content: str, tags: List[Any]) -> None:
        """Learn tag prediction patterns"""
        
        if not tags:
            return
        
        try:
            # Extract tag names
            tag_names = []
            for tag in tags:
                if isinstance(tag, dict):
                    tag_name = tag.get('name', '')
                else:
                    tag_name = str(tag)
                if tag_name:
                    tag_names.append(tag_name)
            
            if not tag_names:
                return
            
            # Extract features for tag learning
            keywords = TextUtils.extract_keywords(f"{title} {content}")
            domain = UrlUtils.extract_domain(url)
            
            # Learn keyword -> tag associations
            for keyword in keywords[:10]:  # Top 10 keywords
                for tag_name in tag_names:
                    pattern_id = f"keyword_tag:{keyword}"
                    self.tag_patterns[pattern_id][tag_name] += 1
                    self._update_pattern_strength(pattern_id, "tag", keyword, tag_name)
            
            # Learn domain -> tag associations
            if domain:
                for tag_name in tag_names:
                    pattern_id = f"domain_tag:{domain}"
                    self.tag_patterns[pattern_id][tag_name] += 1
                    self._update_pattern_strength(pattern_id, "tag", domain, tag_name)
                    
        except Exception as e:
            logger.warning(f"Failed to learn tag patterns: {e}")
    
    def _learn_domain_patterns(self, url: str, collection_name: str, tags: List[Any]) -> None:
        """Learn domain classification patterns"""
        
        try:
            domain = UrlUtils.extract_domain(url)
            if not domain:
                return
            
            # Learn domain characteristics
            domain_features = {
                'collection': collection_name,
                'tag_count': len(tags),
                'has_gaming_indicators': self._has_gaming_indicators(url, tags),
                'has_dev_indicators': self._has_dev_indicators(url, tags),
                'has_ai_indicators': self._has_ai_indicators(url, tags)
            }
            
            pattern_id = f"domain_features:{domain}"
            self.domain_patterns[pattern_id].update(domain_features)
            
            # Update domain pattern strength
            self._update_pattern_strength(pattern_id, "domain", domain, str(domain_features))
            
        except Exception as e:
            logger.warning(f"Failed to learn domain patterns: {e}")
    
    def _learn_content_patterns(self, title: str, content: str, tags: List[Any], collection_name: str) -> None:
        """Learn content-based patterns"""
        
        try:
            # Extract content features
            text_content = f"{title} {content}"
            keywords = TextUtils.extract_keywords(text_content)
            word_count = len(text_content.split())
            
            # Learn content type patterns
            content_type = self._classify_content_type(text_content)
            if content_type and collection_name:
                pattern_id = f"content_type:{content_type}"
                self.content_patterns[pattern_id][collection_name] += 1
                self._update_pattern_strength(pattern_id, "content", content_type, collection_name)
            
            # Learn content length patterns
            length_category = self._categorize_content_length(word_count)
            if length_category and collection_name:
                pattern_id = f"content_length:{length_category}"
                self.content_patterns[pattern_id][collection_name] += 1
                self._update_pattern_strength(pattern_id, "content", length_category, collection_name)
                
        except Exception as e:
            logger.warning(f"Failed to learn content patterns: {e}")
    
    def _update_pattern_strength(self, pattern_id: str, pattern_type: str, 
                                feature: str, target: str) -> None:
        """Update the strength of a learned pattern"""
        
        try:
            current_time = datetime.now()
            
            if pattern_id in self.pattern_strengths:
                # Update existing pattern
                pattern = self.pattern_strengths[pattern_id]
                pattern.usage_count += 1
                pattern.last_used = current_time
                
                # Update strength using learning rate
                pattern.strength = min(1.0, pattern.strength + self.learning_rate)
                
            else:
                # Create new pattern
                pattern = PatternStrength(
                    pattern_id=pattern_id,
                    strength=self.learning_rate,
                    confidence=0.5,  # Initial confidence
                    usage_count=1,
                    success_rate=0.5,  # Will be updated with feedback
                    last_used=current_time,
                    created_at=current_time,
                    pattern_type=pattern_type,
                    metadata={
                        'feature': feature,
                        'target': target
                    }
                )
                self.pattern_strengths[pattern_id] = pattern
            
        except Exception as e:
            logger.warning(f"Failed to update pattern strength for {pattern_id}: {e}")
    
    def _update_pattern_strengths(self) -> Tuple[int, int]:
        """Update all pattern strengths and return counts of new/updated patterns"""
        
        new_patterns = 0
        updated_patterns = 0
        current_time = datetime.now()
        
        try:
            # Apply decay to unused patterns
            for pattern_id, pattern in list(self.pattern_strengths.items()):
                days_since_use = (current_time - pattern.last_used).days
                
                if days_since_use > 30:  # Decay patterns not used in 30 days
                    decay_factor = self.pattern_decay_rate ** (days_since_use / 30)
                    pattern.strength *= decay_factor
                    
                    # Remove very weak patterns
                    if pattern.strength < self.min_pattern_strength:
                        del self.pattern_strengths[pattern_id]
                        continue
                
                # Count as updated if recently modified
                if (current_time - pattern.created_at).days < 1:
                    if pattern.usage_count == 1:
                        new_patterns += 1
                    else:
                        updated_patterns += 1
            
            # Limit number of patterns per type
            self._limit_patterns_per_type()
            
            logger.debug(f"Pattern strength update: {new_patterns} new, {updated_patterns} updated")
            return new_patterns, updated_patterns
            
        except Exception as e:
            logger.error(f"Failed to update pattern strengths: {e}")
            return 0, 0
    
    def _limit_patterns_per_type(self) -> None:
        """Limit the number of patterns per type to prevent memory bloat"""
        
        try:
            patterns_by_type = defaultdict(list)
            
            # Group patterns by type
            for pattern_id, pattern in self.pattern_strengths.items():
                patterns_by_type[pattern.pattern_type].append((pattern_id, pattern))
            
            # Limit each type
            for pattern_type, patterns in patterns_by_type.items():
                if len(patterns) > self.max_patterns_per_type:
                    # Sort by strength and keep the strongest
                    patterns.sort(key=lambda x: x[1].strength, reverse=True)
                    
                    # Remove weakest patterns
                    for pattern_id, _ in patterns[self.max_patterns_per_type:]:
                        del self.pattern_strengths[pattern_id]
                    
                    logger.debug(f"Limited {pattern_type} patterns to {self.max_patterns_per_type}")
                    
        except Exception as e:
            logger.warning(f"Failed to limit patterns per type: {e}")
    
    def predict_category(self, url: str, title: str = "", content: str = "") -> List[Tuple[str, float]]:
        """Predict category using learned patterns"""
        
        try:
            predictions = defaultdict(float)
            
            # Extract features
            domain = UrlUtils.extract_domain(url)
            path_segments = UrlUtils.extract_path_segments(url)
            keywords = TextUtils.extract_keywords(f"{title} {content}")
            
            # Use domain patterns
            if domain:
                pattern_id = f"domain_category:{domain}"
                if pattern_id in self.pattern_strengths:
                    pattern = self.pattern_strengths[pattern_id]
                    for category, count in self.category_patterns[pattern_id].items():
                        confidence = pattern.strength * (count / sum(self.category_patterns[pattern_id].values()))
                        predictions[category] += confidence
            
            # Use path patterns
            for segment in path_segments[:3]:
                pattern_id = f"path_category:{segment}"
                if pattern_id in self.pattern_strengths:
                    pattern = self.pattern_strengths[pattern_id]
                    for category, count in self.category_patterns[pattern_id].items():
                        confidence = pattern.strength * (count / sum(self.category_patterns[pattern_id].values())) * 0.7
                        predictions[category] += confidence
            
            # Use keyword patterns
            for keyword in keywords[:5]:
                pattern_id = f"keyword_category:{keyword}"
                if pattern_id in self.pattern_strengths:
                    pattern = self.pattern_strengths[pattern_id]
                    for category, count in self.category_patterns[pattern_id].items():
                        confidence = pattern.strength * (count / sum(self.category_patterns[pattern_id].values())) * 0.5
                        predictions[category] += confidence
            
            # Sort by confidence
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            return sorted_predictions[:5]  # Top 5 predictions
            
        except Exception as e:
            logger.error(f"Failed to predict category: {e}")
            return []
    
    def predict_tags(self, url: str, title: str = "", content: str = "") -> List[Tuple[str, float]]:
        """Predict tags using learned patterns"""
        
        try:
            predictions = defaultdict(float)
            
            # Extract features
            domain = UrlUtils.extract_domain(url)
            keywords = TextUtils.extract_keywords(f"{title} {content}")
            
            # Use keyword patterns
            for keyword in keywords[:10]:
                pattern_id = f"keyword_tag:{keyword}"
                if pattern_id in self.pattern_strengths:
                    pattern = self.pattern_strengths[pattern_id]
                    for tag, count in self.tag_patterns[pattern_id].items():
                        confidence = pattern.strength * (count / sum(self.tag_patterns[pattern_id].values()))
                        predictions[tag] += confidence
            
            # Use domain patterns
            if domain:
                pattern_id = f"domain_tag:{domain}"
                if pattern_id in self.pattern_strengths:
                    pattern = self.pattern_strengths[pattern_id]
                    for tag, count in self.tag_patterns[pattern_id].items():
                        confidence = pattern.strength * (count / sum(self.tag_patterns[pattern_id].values())) * 0.8
                        predictions[tag] += confidence
            
            # Sort by confidence
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            return sorted_predictions[:10]  # Top 10 predictions
            
        except Exception as e:
            logger.error(f"Failed to predict tags: {e}")
            return []
    
    def track_prediction_feedback(self, prediction_type: str, predicted: List[str], 
                                 actual: List[str], context: Dict[str, Any]) -> None:
        """Track feedback on predictions to improve accuracy"""
        
        try:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'prediction_type': prediction_type,
                'predicted': predicted,
                'actual': actual,
                'context': context,
                'accuracy': self._calculate_prediction_accuracy(predicted, actual)
            }
            
            self.prediction_history.append(feedback_entry)
            
            # Update pattern success rates based on feedback
            self._update_pattern_success_rates(prediction_type, predicted, actual, context)
            
            # Track accuracy trends
            self.accuracy_tracking[prediction_type].append(feedback_entry['accuracy'])
            
            logger.debug(f"Tracked prediction feedback: {prediction_type}, accuracy: {feedback_entry['accuracy']:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to track prediction feedback: {e}")
    
    def _update_pattern_success_rates(self, prediction_type: str, predicted: List[str], 
                                     actual: List[str], context: Dict[str, Any]) -> None:
        """Update success rates of patterns based on prediction feedback"""
        
        try:
            url = context.get('url', '')
            title = context.get('title', '')
            content = context.get('content', '')
            
            if prediction_type == 'category':
                # Update category pattern success rates
                domain = UrlUtils.extract_domain(url)
                if domain:
                    pattern_id = f"domain_category:{domain}"
                    if pattern_id in self.pattern_strengths:
                        pattern = self.pattern_strengths[pattern_id]
                        accuracy = self._calculate_prediction_accuracy(predicted, actual)
                        pattern.success_rate = (pattern.success_rate + accuracy) / 2
                        pattern.confidence = min(1.0, pattern.confidence + (accuracy - 0.5) * 0.1)
            
            elif prediction_type == 'tags':
                # Update tag pattern success rates
                keywords = TextUtils.extract_keywords(f"{title} {content}")
                for keyword in keywords[:5]:
                    pattern_id = f"keyword_tag:{keyword}"
                    if pattern_id in self.pattern_strengths:
                        pattern = self.pattern_strengths[pattern_id]
                        accuracy = self._calculate_prediction_accuracy(predicted, actual)
                        pattern.success_rate = (pattern.success_rate + accuracy) / 2
                        pattern.confidence = min(1.0, pattern.confidence + (accuracy - 0.5) * 0.1)
            
        except Exception as e:
            logger.warning(f"Failed to update pattern success rates: {e}")
    
    def _calculate_prediction_accuracy(self, predicted: List[str], actual: List[str]) -> float:
        """Calculate accuracy of predictions"""
        
        if not actual:
            return 0.0
        
        if not predicted:
            return 0.0
        
        # Calculate intersection over union (Jaccard similarity)
        predicted_set = set(predicted)
        actual_set = set(actual)
        
        intersection = len(predicted_set.intersection(actual_set))
        union = len(predicted_set.union(actual_set))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_learning_improvements(self) -> List[Dict[str, Any]]:
        """Calculate improvements from learning"""
        
        improvements = []
        
        try:
            # Calculate accuracy improvements
            for prediction_type, accuracies in self.accuracy_tracking.items():
                if len(accuracies) >= 2:
                    recent_accuracy = sum(accuracies[-10:]) / len(accuracies[-10:])
                    older_accuracy = sum(accuracies[-20:-10]) / len(accuracies[-20:-10]) if len(accuracies) >= 20 else recent_accuracy
                    
                    improvement = recent_accuracy - older_accuracy
                    if abs(improvement) > 0.01:  # Significant improvement
                        improvements.append({
                            'type': 'accuracy_improvement',
                            'prediction_type': prediction_type,
                            'improvement': improvement,
                            'recent_accuracy': recent_accuracy,
                            'previous_accuracy': older_accuracy
                        })
            
            # Calculate pattern strength improvements
            strong_patterns = [p for p in self.pattern_strengths.values() if p.strength > 0.7]
            if strong_patterns:
                improvements.append({
                    'type': 'pattern_strength',
                    'strong_patterns_count': len(strong_patterns),
                    'average_strength': sum(p.strength for p in strong_patterns) / len(strong_patterns)
                })
            
        except Exception as e:
            logger.warning(f"Failed to calculate learning improvements: {e}")
        
        return improvements
    
    def end_learning_session(self) -> Optional[LearningSession]:
        """End the current learning session"""
        
        if not self.active_session:
            return None
        
        try:
            session = self.active_session
            session.end_time = datetime.now()
            
            # Calculate session metrics
            session_duration = (session.end_time - session.start_time).total_seconds()
            
            # Update global metrics
            self.learning_metrics.total_bookmarks_processed += session.bookmarks_processed
            self.learning_metrics.patterns_learned += session.new_patterns
            self.learning_metrics.patterns_updated += session.updated_patterns
            self.learning_metrics.learning_sessions += 1
            self.learning_metrics.last_learning_time = session.end_time
            
            # Update average learning time
            if self.learning_metrics.learning_sessions > 0:
                total_time = self.learning_metrics.average_learning_time * (self.learning_metrics.learning_sessions - 1) + session_duration
                self.learning_metrics.average_learning_time = total_time / self.learning_metrics.learning_sessions
            
            # Store session
            self.learning_sessions.append(session)
            
            # Save learning data
            self._save_learning_data()
            
            logger.info(f"Learning session completed: {session.session_id}, duration: {session_duration:.2f}s")
            
            self.active_session = None
            return session
            
        except Exception as e:
            logger.error(f"Failed to end learning session: {e}")
            return None
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        
        try:
            # Calculate pattern statistics
            pattern_stats = defaultdict(int)
            for pattern in self.pattern_strengths.values():
                pattern_stats[pattern.pattern_type] += 1
            
            # Calculate accuracy statistics
            accuracy_stats = {}
            for prediction_type, accuracies in self.accuracy_tracking.items():
                if accuracies:
                    accuracy_stats[prediction_type] = {
                        'current_accuracy': accuracies[-1] if accuracies else 0.0,
                        'average_accuracy': sum(accuracies) / len(accuracies),
                        'best_accuracy': max(accuracies),
                        'predictions_tracked': len(accuracies)
                    }
            
            return {
                'learning_metrics': {
                    'total_bookmarks_processed': self.learning_metrics.total_bookmarks_processed,
                    'patterns_learned': self.learning_metrics.patterns_learned,
                    'patterns_updated': self.learning_metrics.patterns_updated,
                    'learning_sessions': self.learning_metrics.learning_sessions,
                    'last_learning_time': self.learning_metrics.last_learning_time.isoformat() if self.learning_metrics.last_learning_time else None,
                    'average_learning_time': self.learning_metrics.average_learning_time
                },
                'pattern_statistics': dict(pattern_stats),
                'total_patterns': len(self.pattern_strengths),
                'strong_patterns': len([p for p in self.pattern_strengths.values() if p.strength > 0.7]),
                'accuracy_statistics': accuracy_stats,
                'recent_sessions': [
                    {
                        'session_id': s.session_id,
                        'start_time': s.start_time.isoformat(),
                        'bookmarks_processed': s.bookmarks_processed,
                        'new_patterns': s.new_patterns,
                        'session_type': s.session_type
                    }
                    for s in self.learning_sessions[-5:]  # Last 5 sessions
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning statistics: {e}")
            return {'error': str(e)}
    
    def _has_gaming_indicators(self, url: str, tags: List[Any]) -> bool:
        """Check if URL/tags indicate gaming content"""
        
        gaming_domains = ['twitch.tv', 'steam', 'itch.io', 'gamebanana.com']
        gaming_keywords = ['game', 'gaming', 'genshin', 'achievement', 'mod']
        
        url_lower = url.lower()
        tag_text = ' '.join(str(tag).lower() for tag in tags)
        
        return (any(domain in url_lower for domain in gaming_domains) or
                any(keyword in url_lower or keyword in tag_text for keyword in gaming_keywords))
    
    def _has_dev_indicators(self, url: str, tags: List[Any]) -> bool:
        """Check if URL/tags indicate development content"""
        
        dev_domains = ['github.com', 'stackoverflow.com', 'npmjs.com']
        dev_keywords = ['code', 'programming', 'api', 'framework', 'library']
        
        url_lower = url.lower()
        tag_text = ' '.join(str(tag).lower() for tag in tags)
        
        return (any(domain in url_lower for domain in dev_domains) or
                any(keyword in url_lower or keyword in tag_text for keyword in dev_keywords))
    
    def _has_ai_indicators(self, url: str, tags: List[Any]) -> bool:
        """Check if URL/tags indicate AI/ML content"""
        
        ai_domains = ['openai.com', 'huggingface.co', 'arxiv.org']
        ai_keywords = ['ai', 'ml', 'machine learning', 'neural', 'llm']
        
        url_lower = url.lower()
        tag_text = ' '.join(str(tag).lower() for tag in tags)
        
        return (any(domain in url_lower for domain in ai_domains) or
                any(keyword in url_lower or keyword in tag_text for keyword in ai_keywords))
    
    def _classify_content_type(self, text_content: str) -> str:
        """Classify content type based on text"""
        
        text_lower = text_content.lower()
        
        if any(keyword in text_lower for keyword in ['tutorial', 'guide', 'how to']):
            return 'tutorial'
        elif any(keyword in text_lower for keyword in ['documentation', 'docs', 'api']):
            return 'documentation'
        elif any(keyword in text_lower for keyword in ['news', 'article', 'blog']):
            return 'article'
        elif any(keyword in text_lower for keyword in ['tool', 'utility', 'app']):
            return 'tool'
        else:
            return 'general'
    
    def _categorize_content_length(self, word_count: int) -> str:
        """Categorize content by length"""
        
        if word_count < 50:
            return 'short'
        elif word_count < 200:
            return 'medium'
        elif word_count < 1000:
            return 'long'
        else:
            return 'very_long'
    
    def _save_learning_data(self) -> None:
        """Save learning data to disk"""
        
        try:
            learning_data = {
                'pattern_strengths': {
                    pattern_id: {
                        'pattern_id': pattern.pattern_id,
                        'strength': pattern.strength,
                        'confidence': pattern.confidence,
                        'usage_count': pattern.usage_count,
                        'success_rate': pattern.success_rate,
                        'last_used': pattern.last_used.isoformat(),
                        'created_at': pattern.created_at.isoformat(),
                        'pattern_type': pattern.pattern_type,
                        'metadata': pattern.metadata
                    }
                    for pattern_id, pattern in self.pattern_strengths.items()
                },
                'learning_metrics': {
                    'total_bookmarks_processed': self.learning_metrics.total_bookmarks_processed,
                    'patterns_learned': self.learning_metrics.patterns_learned,
                    'patterns_updated': self.learning_metrics.patterns_updated,
                    'learning_sessions': self.learning_metrics.learning_sessions,
                    'last_learning_time': self.learning_metrics.last_learning_time.isoformat() if self.learning_metrics.last_learning_time else None,
                    'average_learning_time': self.learning_metrics.average_learning_time
                },
                'category_patterns': {k: dict(v) for k, v in self.category_patterns.items()},
                'tag_patterns': {k: dict(v) for k, v in self.tag_patterns.items()},
                'domain_patterns': {k: dict(v) for k, v in self.domain_patterns.items()},
                'content_patterns': {k: dict(v) for k, v in self.content_patterns.items()},
                'accuracy_tracking': {k: list(v) for k, v in self.accuracy_tracking.items()},
                'version': '1.0',
                'saved_at': datetime.now().isoformat()
            }
            
            learning_file = self.data_dir / 'continuous_learning.json'
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved learning data to {learning_file}")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def _load_learning_data(self) -> None:
        """Load learning data from disk"""
        
        try:
            learning_file = self.data_dir / 'continuous_learning.json'
            
            if not learning_file.exists():
                logger.info("No existing learning data found")
                return
            
            with open(learning_file, 'r', encoding='utf-8') as f:
                learning_data = json.load(f)
            
            # Load pattern strengths
            if 'pattern_strengths' in learning_data:
                for pattern_id, pattern_data in learning_data['pattern_strengths'].items():
                    self.pattern_strengths[pattern_id] = PatternStrength(
                        pattern_id=pattern_data['pattern_id'],
                        strength=pattern_data['strength'],
                        confidence=pattern_data['confidence'],
                        usage_count=pattern_data['usage_count'],
                        success_rate=pattern_data['success_rate'],
                        last_used=datetime.fromisoformat(pattern_data['last_used']),
                        created_at=datetime.fromisoformat(pattern_data['created_at']),
                        pattern_type=pattern_data['pattern_type'],
                        metadata=pattern_data.get('metadata', {})
                    )
            
            # Load learning metrics
            if 'learning_metrics' in learning_data:
                metrics_data = learning_data['learning_metrics']
                self.learning_metrics = LearningMetrics(
                    total_bookmarks_processed=metrics_data.get('total_bookmarks_processed', 0),
                    patterns_learned=metrics_data.get('patterns_learned', 0),
                    patterns_updated=metrics_data.get('patterns_updated', 0),
                    learning_sessions=metrics_data.get('learning_sessions', 0),
                    last_learning_time=datetime.fromisoformat(metrics_data['last_learning_time']) if metrics_data.get('last_learning_time') else None,
                    average_learning_time=metrics_data.get('average_learning_time', 0.0)
                )
            
            # Load pattern data
            if 'category_patterns' in learning_data:
                self.category_patterns.update({k: defaultdict(float, v) for k, v in learning_data['category_patterns'].items()})
            
            if 'tag_patterns' in learning_data:
                self.tag_patterns.update({k: defaultdict(float, v) for k, v in learning_data['tag_patterns'].items()})
            
            if 'domain_patterns' in learning_data:
                self.domain_patterns.update({k: defaultdict(float, v) for k, v in learning_data['domain_patterns'].items()})
            
            if 'content_patterns' in learning_data:
                self.content_patterns.update({k: defaultdict(float, v) for k, v in learning_data['content_patterns'].items()})
            
            if 'accuracy_tracking' in learning_data:
                self.accuracy_tracking.update({k: list(v) for k, v in learning_data['accuracy_tracking'].items()})
            
            logger.info(f"Loaded learning data: {len(self.pattern_strengths)} patterns, {self.learning_metrics.learning_sessions} sessions")
            
        except Exception as e:
            logger.warning(f"Failed to load learning data: {e}")
    
    def retrain_models_incrementally(self, new_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Incrementally retrain models with new data"""
        
        if not self.active_session:
            self.start_learning_session("incremental_retrain")
        
        retrain_results = {
            'session_id': self.active_session.session_id,
            'models_retrained': 0,
            'accuracy_improvements': {},
            'pattern_updates': 0,
            'errors': []
        }
        
        try:
            logger.info(f"Starting incremental model retraining with {len(new_data)} samples")
            
            # Store accuracy before retraining
            accuracy_before = self._calculate_current_accuracy()
            
            # Learn from new data
            learning_results = self.learn_from_new_bookmarks(new_data)
            
            # Update model parameters based on new patterns
            self._update_model_parameters()
            
            # Calculate accuracy after retraining
            accuracy_after = self._calculate_current_accuracy()
            
            # Track improvements
            for prediction_type in accuracy_before.keys():
                improvement = accuracy_after.get(prediction_type, 0) - accuracy_before.get(prediction_type, 0)
                retrain_results['accuracy_improvements'][prediction_type] = improvement
            
            retrain_results['models_retrained'] = len(accuracy_before)
            retrain_results['pattern_updates'] = learning_results.get('patterns_updated', 0)
            
            logger.info(f"Incremental retraining completed: {retrain_results}")
            return retrain_results
            
        except Exception as e:
            error_msg = f"Incremental retraining failed: {e}"
            logger.error(error_msg)
            retrain_results['errors'].append(error_msg)
            return retrain_results
    
    def _calculate_current_accuracy(self) -> Dict[str, float]:
        """Calculate current model accuracy across prediction types"""
        
        accuracy_scores = {}
        
        try:
            # Calculate accuracy for each prediction type
            for prediction_type, accuracies in self.accuracy_tracking.items():
                if accuracies:
                    # Use recent accuracy (last 10 predictions)
                    recent_accuracies = accuracies[-10:]
                    accuracy_scores[prediction_type] = sum(recent_accuracies) / len(recent_accuracies)
                else:
                    accuracy_scores[prediction_type] = 0.0
            
            return accuracy_scores
            
        except Exception as e:
            logger.warning(f"Failed to calculate current accuracy: {e}")
            return {}
    
    def _update_model_parameters(self) -> None:
        """Update model parameters based on learned patterns"""
        
        try:
            # Adjust learning rate based on pattern reliability
            reliable_patterns = [p for p in self.pattern_strengths.values() if p.success_rate > 0.7]
            unreliable_patterns = [p for p in self.pattern_strengths.values() if p.success_rate < 0.3]
            
            if len(reliable_patterns) > len(unreliable_patterns):
                # Increase learning rate when patterns are reliable
                self.learning_rate = min(0.2, self.learning_rate * 1.1)
            else:
                # Decrease learning rate when patterns are unreliable
                self.learning_rate = max(0.05, self.learning_rate * 0.9)
            
            # Adjust pattern decay rate based on usage patterns
            active_patterns = [p for p in self.pattern_strengths.values() 
                             if (datetime.now() - p.last_used).days < 7]
            
            if len(active_patterns) > len(self.pattern_strengths) * 0.5:
                # Slower decay when patterns are actively used
                self.pattern_decay_rate = min(0.98, self.pattern_decay_rate + 0.01)
            else:
                # Faster decay when patterns are not used
                self.pattern_decay_rate = max(0.90, self.pattern_decay_rate - 0.01)
            
            logger.debug(f"Updated model parameters: learning_rate={self.learning_rate:.3f}, decay_rate={self.pattern_decay_rate:.3f}")
            
        except Exception as e:
            logger.warning(f"Failed to update model parameters: {e}")
    
    def analyze_pattern_reliability(self) -> Dict[str, Any]:
        """Analyze the reliability of learned patterns"""
        
        try:
            reliability_analysis = {
                'total_patterns': len(self.pattern_strengths),
                'reliable_patterns': 0,
                'unreliable_patterns': 0,
                'pattern_types': defaultdict(lambda: {'count': 0, 'avg_reliability': 0.0}),
                'reliability_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'recommendations': []
            }
            
            # Analyze each pattern
            for pattern in self.pattern_strengths.values():
                # Calculate overall reliability score
                reliability_score = (pattern.strength * 0.4 + 
                                   pattern.confidence * 0.3 + 
                                   pattern.success_rate * 0.3)
                
                # Categorize reliability
                if reliability_score > 0.7:
                    reliability_analysis['reliable_patterns'] += 1
                    reliability_analysis['reliability_distribution']['high'] += 1
                elif reliability_score > 0.4:
                    reliability_analysis['reliability_distribution']['medium'] += 1
                else:
                    reliability_analysis['unreliable_patterns'] += 1
                    reliability_analysis['reliability_distribution']['low'] += 1
                
                # Track by pattern type
                pattern_type_stats = reliability_analysis['pattern_types'][pattern.pattern_type]
                pattern_type_stats['count'] += 1
                pattern_type_stats['avg_reliability'] = (
                    (pattern_type_stats['avg_reliability'] * (pattern_type_stats['count'] - 1) + reliability_score) /
                    pattern_type_stats['count']
                )
            
            # Generate recommendations
            total_patterns = reliability_analysis['total_patterns']
            if total_patterns > 0:
                unreliable_ratio = reliability_analysis['unreliable_patterns'] / total_patterns
                
                if unreliable_ratio > 0.3:
                    reliability_analysis['recommendations'].append(
                        "High number of unreliable patterns detected. Consider increasing minimum pattern strength threshold."
                    )
                
                if reliability_analysis['reliable_patterns'] < 10:
                    reliability_analysis['recommendations'].append(
                        "Low number of reliable patterns. System needs more training data to improve accuracy."
                    )
                
                # Check pattern type balance
                for pattern_type, stats in reliability_analysis['pattern_types'].items():
                    if stats['avg_reliability'] < 0.3:
                        reliability_analysis['recommendations'].append(
                            f"Pattern type '{pattern_type}' has low reliability. Consider reviewing learning algorithm for this type."
                        )
            
            return reliability_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze pattern reliability: {e}")
            return {'error': str(e)}
    
    def get_learning_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive learning performance metrics"""
        
        try:
            current_time = datetime.now()
            
            # Calculate time-based metrics
            if self.learning_sessions:
                recent_sessions = [s for s in self.learning_sessions 
                                 if (current_time - s.start_time).days < 30]
                
                session_metrics = {
                    'total_sessions': len(self.learning_sessions),
                    'recent_sessions': len(recent_sessions),
                    'avg_bookmarks_per_session': (
                        sum(s.bookmarks_processed for s in self.learning_sessions) / len(self.learning_sessions)
                        if self.learning_sessions else 0
                    ),
                    'avg_patterns_per_session': (
                        sum(s.new_patterns + s.updated_patterns for s in self.learning_sessions) / len(self.learning_sessions)
                        if self.learning_sessions else 0
                    )
                }
            else:
                session_metrics = {
                    'total_sessions': 0,
                    'recent_sessions': 0,
                    'avg_bookmarks_per_session': 0,
                    'avg_patterns_per_session': 0
                }
            
            # Calculate pattern growth metrics
            pattern_age_distribution = {'new': 0, 'mature': 0, 'old': 0}
            for pattern in self.pattern_strengths.values():
                age_days = (current_time - pattern.created_at).days
                if age_days < 7:
                    pattern_age_distribution['new'] += 1
                elif age_days < 30:
                    pattern_age_distribution['mature'] += 1
                else:
                    pattern_age_distribution['old'] += 1
            
            # Calculate learning velocity
            learning_velocity = {
                'patterns_per_day': 0,
                'accuracy_improvement_rate': 0
            }
            
            if self.learning_sessions and len(self.learning_sessions) > 1:
                first_session = min(self.learning_sessions, key=lambda s: s.start_time)
                last_session = max(self.learning_sessions, key=lambda s: s.start_time)
                
                days_elapsed = (last_session.start_time - first_session.start_time).days
                if days_elapsed > 0:
                    total_patterns = sum(s.new_patterns + s.updated_patterns for s in self.learning_sessions)
                    learning_velocity['patterns_per_day'] = total_patterns / days_elapsed
            
            return {
                'session_metrics': session_metrics,
                'pattern_age_distribution': pattern_age_distribution,
                'learning_velocity': learning_velocity,
                'current_model_parameters': {
                    'learning_rate': self.learning_rate,
                    'pattern_decay_rate': self.pattern_decay_rate,
                    'min_pattern_strength': self.min_pattern_strength,
                    'max_patterns_per_type': self.max_patterns_per_type
                },
                'reliability_analysis': self.analyze_pattern_reliability()
            }
            
        except Exception as e:
            logger.error(f"Failed to get learning performance metrics: {e}")
            return {'error': str(e)}
    
    def optimize_learning_parameters(self) -> Dict[str, Any]:
        """Optimize learning parameters based on performance history"""
        
        try:
            optimization_results = {
                'parameters_changed': [],
                'performance_before': {},
                'performance_after': {},
                'recommendations': []
            }
            
            # Get current performance
            current_accuracy = self._calculate_current_accuracy()
            optimization_results['performance_before'] = current_accuracy
            
            # Analyze pattern reliability
            reliability_analysis = self.analyze_pattern_reliability()
            
            # Optimize based on reliability
            if reliability_analysis.get('unreliable_patterns', 0) > reliability_analysis.get('reliable_patterns', 0):
                # Too many unreliable patterns - increase minimum strength threshold
                old_threshold = self.min_pattern_strength
                self.min_pattern_strength = min(0.3, self.min_pattern_strength * 1.2)
                
                if self.min_pattern_strength != old_threshold:
                    optimization_results['parameters_changed'].append({
                        'parameter': 'min_pattern_strength',
                        'old_value': old_threshold,
                        'new_value': self.min_pattern_strength,
                        'reason': 'Too many unreliable patterns'
                    })
            
            # Optimize learning rate based on accuracy trends
            for prediction_type, accuracies in self.accuracy_tracking.items():
                if len(accuracies) >= 10:
                    recent_trend = sum(accuracies[-5:]) / 5 - sum(accuracies[-10:-5]) / 5
                    
                    if recent_trend < -0.05:  # Accuracy declining
                        old_rate = self.learning_rate
                        self.learning_rate = max(0.05, self.learning_rate * 0.8)
                        
                        if self.learning_rate != old_rate:
                            optimization_results['parameters_changed'].append({
                                'parameter': 'learning_rate',
                                'old_value': old_rate,
                                'new_value': self.learning_rate,
                                'reason': f'Declining accuracy in {prediction_type}'
                            })
                    
                    elif recent_trend > 0.05:  # Accuracy improving
                        old_rate = self.learning_rate
                        self.learning_rate = min(0.2, self.learning_rate * 1.1)
                        
                        if self.learning_rate != old_rate:
                            optimization_results['parameters_changed'].append({
                                'parameter': 'learning_rate',
                                'old_value': old_rate,
                                'new_value': self.learning_rate,
                                'reason': f'Improving accuracy in {prediction_type}'
                            })
            
            # Clean up weak patterns if too many exist
            weak_patterns = [p for p in self.pattern_strengths.values() if p.strength < self.min_pattern_strength]
            if len(weak_patterns) > len(self.pattern_strengths) * 0.2:
                removed_count = 0
                for pattern_id, pattern in list(self.pattern_strengths.items()):
                    if pattern.strength < self.min_pattern_strength:
                        del self.pattern_strengths[pattern_id]
                        removed_count += 1
                
                optimization_results['parameters_changed'].append({
                    'parameter': 'pattern_cleanup',
                    'old_value': len(self.pattern_strengths) + removed_count,
                    'new_value': len(self.pattern_strengths),
                    'reason': f'Removed {removed_count} weak patterns'
                })
            
            # Generate recommendations
            if not optimization_results['parameters_changed']:
                optimization_results['recommendations'].append("No parameter changes needed - system is performing well")
            else:
                optimization_results['recommendations'].append("Parameters optimized based on performance analysis")
            
            # Save optimized parameters
            self._save_learning_data()
            
            logger.info(f"Learning parameter optimization completed: {len(optimization_results['parameters_changed'])} changes made")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Failed to optimize learning parameters: {e}")
            return {'error': str(e)}