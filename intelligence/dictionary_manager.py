"""Smart Dictionary Manager - Orchestrates all intelligence components"""

import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

from .category_dictionary import CategoryDictionary
from .tag_dictionary import TagDictionary
from .pattern_learner import PatternLearner
from .domain_classifier import DomainClassifier
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class SmartDictionaryManager:
    """Central manager for all intelligent categorization and tagging"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize smart dictionary manager"""
        self.config = config
        self.intelligence_config = config.get('intelligence', {})
        self.data_dir = config.get('directories', {}).get('data_dir', 'data')
        
        # Initialize components
        self.category_dictionary = CategoryDictionary()
        self.tag_dictionary = TagDictionary()
        self.pattern_learner = PatternLearner(self.data_dir)
        self.domain_classifier = DomainClassifier()
        
        # Learning state
        self.is_trained = False
        self.last_training_time = None
        self.training_stats = {}
        
        logger.info("Smart Dictionary Manager initialized")
    
    def learn_from_bookmark_data(self, bookmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn from existing bookmark data to improve intelligence"""
        
        logger.info(f"Starting comprehensive learning from {len(bookmarks)} bookmarks")
        
        learning_results = {
            'total_bookmarks': len(bookmarks),
            'learning_start_time': datetime.now().isoformat(),
            'components_trained': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Learn category patterns
            logger.info("Learning category patterns...")
            self.category_dictionary.learn_from_bookmark_data(bookmarks)
            learning_results['components_trained'].append('category_dictionary')
            
            # Learn tag patterns
            logger.info("Learning tag patterns...")
            self.tag_dictionary.learn_from_bookmark_tags(bookmarks)
            learning_results['components_trained'].append('tag_dictionary')
            
            # Learn general patterns
            logger.info("Learning general patterns...")
            pattern_stats = self.pattern_learner.learn_from_bookmark_history(bookmarks)
            learning_results['pattern_learning_stats'] = pattern_stats
            learning_results['components_trained'].append('pattern_learner')
            
            # Learn domain classifications
            logger.info("Learning domain classifications...")
            self._learn_domain_classifications(bookmarks)
            learning_results['components_trained'].append('domain_classifier')
            
            # Update training state
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.training_stats = learning_results
            
            learning_results['learning_end_time'] = datetime.now().isoformat()
            learning_results['success'] = True
            
            logger.info(f"Learning completed successfully. Trained {len(learning_results['components_trained'])} components")
            
            return learning_results
            
        except Exception as e:
            error_msg = f"Learning failed: {e}"
            logger.error(error_msg)
            learning_results['errors'].append(error_msg)
            learning_results['success'] = False
            return learning_results
    
    def _learn_domain_classifications(self, bookmarks: List[Dict[str, Any]]) -> None:
        """Learn domain classifications from bookmark data"""
        
        for bookmark in bookmarks:
            try:
                url = bookmark.get('url', '')
                collection_name = bookmark.get('collection_name', '')
                
                if url and collection_name:
                    # Learn domain -> collection associations
                    self.domain_classifier.learn_from_classifications(
                        url, collection_name, confidence=0.8
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to learn domain classification: {e}")
                continue
    
    def suggest_categories_for_bookmark(self, 
                                      url: str, 
                                      title: str = "", 
                                      content: str = "") -> List[Tuple[str, float]]:
        """Get comprehensive category suggestions for a bookmark"""
        
        try:
            suggestions = []
            
            # Get suggestions from category dictionary
            category_suggestions = self.category_dictionary.suggest_categories_for_url(url, content)
            suggestions.extend([(cat, conf, 'category_dict') for cat, conf in category_suggestions])
            
            # Get suggestions from domain classifier
            domain_suggestions = self.domain_classifier.classify_domain(url)
            suggestions.extend([(cat, conf, 'domain_classifier') for cat, conf in domain_suggestions])
            
            # Get suggestions from pattern learner (if trained)
            if self.is_trained:
                pattern_suggestions = self.pattern_learner.predict_category_for_url(url)
                suggestions.extend([(cat, conf, 'pattern_learner') for cat, conf in pattern_suggestions])
            
            # Get learned suggestions
            learned_suggestions = self.category_dictionary.get_learned_suggestions(url)
            suggestions.extend([(cat, conf, 'learned') for cat, conf in learned_suggestions])
            
            # Aggregate and rank suggestions
            aggregated_suggestions = self._aggregate_category_suggestions(suggestions)
            
            logger.debug(f"Generated {len(aggregated_suggestions)} category suggestions for {url}")
            return aggregated_suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest categories for {url}: {e}")
            return [("General", 0.1)]
    
    def suggest_tags_for_bookmark(self, 
                                 url: str, 
                                 title: str = "", 
                                 content: str = "",
                                 existing_tags: List[str] = None) -> List[Tuple[str, float]]:
        """Get comprehensive tag suggestions for a bookmark"""
        
        if existing_tags is None:
            existing_tags = []
        
        try:
            suggestions = []
            
            # Get suggestions from tag dictionary
            tag_suggestions = self.tag_dictionary.suggest_tags_for_content(
                title, content, url, existing_tags
            )
            suggestions.extend([(tag, conf, 'tag_dict') for tag, conf in tag_suggestions])
            
            # Get suggestions from pattern learner (if trained)
            if self.is_trained and (content or title):
                pattern_suggestions = self.pattern_learner.predict_tags_for_content(
                    f"{title} {content}"
                )
                suggestions.extend([(tag, conf, 'pattern_learner') for tag, conf in pattern_suggestions])
            
            # Aggregate and rank suggestions
            aggregated_suggestions = self._aggregate_tag_suggestions(suggestions, existing_tags)
            
            logger.debug(f"Generated {len(aggregated_suggestions)} tag suggestions for {url}")
            return aggregated_suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest tags for {url}: {e}")
            return []
    
    def _aggregate_category_suggestions(self, suggestions: List[Tuple[str, float, str]]) -> List[Tuple[str, float]]:
        """Aggregate category suggestions from multiple sources"""
        
        # Weight different sources
        source_weights = {
            'category_dict': 1.0,
            'domain_classifier': 0.9,
            'pattern_learner': 0.8,
            'learned': 0.95
        }
        
        # Aggregate scores
        category_scores = {}
        category_sources = {}
        
        for category, confidence, source in suggestions:
            weight = source_weights.get(source, 0.5)
            weighted_score = confidence * weight
            
            if category in category_scores:
                # Use maximum score from multiple sources
                category_scores[category] = max(category_scores[category], weighted_score)
                category_sources[category].add(source)
            else:
                category_scores[category] = weighted_score
                category_sources[category] = {source}
        
        # Boost categories suggested by multiple sources
        for category in category_scores:
            if len(category_sources[category]) > 1:
                category_scores[category] *= 1.2  # 20% boost for multiple sources
        
        # Sort by score and return top suggestions
        sorted_suggestions = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:5]  # Return top 5
    
    def _aggregate_tag_suggestions(self, suggestions: List[Tuple[str, float, str]], existing_tags: List[str]) -> List[Tuple[str, float]]:
        """Aggregate tag suggestions from multiple sources"""
        
        # Weight different sources
        source_weights = {
            'tag_dict': 1.0,
            'pattern_learner': 0.8
        }
        
        # Aggregate scores
        tag_scores = {}
        existing_tags_lower = {tag.lower() for tag in existing_tags}
        
        for tag, confidence, source in suggestions:
            # Skip existing tags
            if tag.lower() in existing_tags_lower:
                continue
            
            weight = source_weights.get(source, 0.5)
            weighted_score = confidence * weight
            
            if tag in tag_scores:
                tag_scores[tag] = max(tag_scores[tag], weighted_score)
            else:
                tag_scores[tag] = weighted_score
        
        # Sort by score and return top suggestions
        sorted_suggestions = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_suggestions[:10]  # Return top 10
    
    def get_related_tags(self, tag: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get tags that are related to the given tag"""
        
        try:
            return self.tag_dictionary.get_related_tags(tag, limit)
        except Exception as e:
            logger.error(f"Failed to get related tags for '{tag}': {e}")
            return []
    
    def track_user_feedback(self, feedback_type: str, original: str, modified: str, context: Dict[str, Any]) -> None:
        """Track user feedback to improve suggestions"""
        
        try:
            # Track feedback in pattern learner
            self.pattern_learner.track_user_feedback(feedback_type, original, modified, context)
            
            logger.debug(f"Tracked user feedback: {feedback_type}")
            
        except Exception as e:
            logger.error(f"Failed to track user feedback: {e}")
    
    def get_intelligence_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the intelligence system"""
        
        try:
            stats = {
                'system_info': {
                    'is_trained': self.is_trained,
                    'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                    'training_stats': self.training_stats
                },
                'category_dictionary': self.category_dictionary.get_category_stats(),
                'tag_dictionary': self.tag_dictionary.get_tag_stats(),
                'pattern_learner': self.pattern_learner.get_learning_stats(),
                'domain_classifier': self.domain_classifier.get_domain_stats()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get intelligence stats: {e}")
            return {'error': str(e)}
    
    def export_intelligence_data(self) -> Dict[str, Any]:
        """Export all learned intelligence data for backup"""
        
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'system_info': {
                    'is_trained': self.is_trained,
                    'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                    'training_stats': self.training_stats
                },
                'category_patterns': self.category_dictionary.export_learned_patterns(),
                'pattern_learning': self.pattern_learner.export_learned_patterns(),
                'domain_classifications': self.domain_classifier.export_learned_domains()
            }
            
            logger.info("Exported intelligence data for backup")
            return export_data
            
        except Exception as e:
            logger.error(f"Failed to export intelligence data: {e}")
            return {'error': str(e)}
    
    def import_intelligence_data(self, data: Dict[str, Any]) -> bool:
        """Import previously exported intelligence data"""
        
        try:
            success = True
            
            # Import category patterns
            if 'category_patterns' in data:
                if not self.category_dictionary.import_learned_patterns(data['category_patterns']):
                    success = False
            
            # Import pattern learning data
            if 'pattern_learning' in data:
                if not self.pattern_learner.import_learned_patterns(data['pattern_learning']):
                    success = False
            
            # Import domain classifications
            if 'domain_classifications' in data:
                if not self.domain_classifier.import_learned_domains(data['domain_classifications']):
                    success = False
            
            # Update system info
            if 'system_info' in data:
                system_info = data['system_info']
                self.is_trained = system_info.get('is_trained', False)
                if system_info.get('last_training_time'):
                    self.last_training_time = datetime.fromisoformat(system_info['last_training_time'])
                self.training_stats = system_info.get('training_stats', {})
            
            if success:
                logger.info(f"Successfully imported intelligence data from version {data.get('version', 'unknown')}")
            else:
                logger.warning("Intelligence data import completed with some errors")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to import intelligence data: {e}")
            return False
    
    def save_intelligence_data(self, file_path: Optional[str] = None) -> bool:
        """Save intelligence data to file"""
        
        try:
            if file_path is None:
                file_path = Path(self.data_dir) / "intelligence_data.json"
            else:
                file_path = Path(file_path)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export and save data
            data = self.export_intelligence_data()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved intelligence data to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save intelligence data: {e}")
            return False
    
    def load_intelligence_data(self, file_path: Optional[str] = None) -> bool:
        """Load intelligence data from file"""
        
        try:
            if file_path is None:
                file_path = Path(self.data_dir) / "intelligence_data.json"
            else:
                file_path = Path(file_path)
            
            if not file_path.exists():
                logger.info("No existing intelligence data found")
                return True
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            success = self.import_intelligence_data(data)
            
            if success:
                logger.info(f"Loaded intelligence data from {file_path}")
            else:
                logger.warning(f"Loaded intelligence data from {file_path} with errors")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load intelligence data: {e}")
            return False