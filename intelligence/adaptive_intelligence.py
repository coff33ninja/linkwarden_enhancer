"""Adaptive intelligence system based on user behavior and feedback"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    SUGGESTION_ACCEPTED = "suggestion_accepted"
    SUGGESTION_REJECTED = "suggestion_rejected"
    SUGGESTION_MODIFIED = "suggestion_modified"
    CATEGORY_CORRECTED = "category_corrected"
    TAG_CORRECTED = "tag_corrected"
    COLLECTION_MOVED = "collection_moved"
    BOOKMARK_DELETED = "bookmark_deleted"
    BOOKMARK_FAVORITED = "bookmark_favorited"


@dataclass
class UserFeedback:
    """Represents user feedback on system suggestions"""
    feedback_id: str
    timestamp: datetime
    feedback_type: FeedbackType
    context: Dict[str, Any]
    original_suggestion: Any
    user_action: Any
    confidence_before: float
    confidence_after: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreference:
    """Represents learned user preferences"""
    preference_id: str
    preference_type: str  # 'category', 'tag', 'collection_structure', 'content_type'
    pattern: str
    strength: float
    confidence: float
    usage_count: int
    success_rate: float
    last_reinforced: datetime
    created_at: datetime
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PersonalizationProfile:
    """User's personalization profile"""
    user_id: str
    created_at: datetime
    last_updated: datetime
    total_interactions: int
    preferences: Dict[str, UserPreference] = field(default_factory=dict)
    behavioral_patterns: Dict[str, Any] = field(default_factory=dict)
    suggestion_accuracy: Dict[str, float] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)


class AdaptiveIntelligence:
    """Adaptive intelligence system that learns from user behavior"""
    
    def __init__(self, data_dir: str = "data", user_id: str = "default"):
        """Initialize adaptive intelligence system"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.user_id = user_id
        
        # User feedback and preferences
        self.feedback_history = deque(maxlen=10000)  # Keep last 10k feedback items
        self.user_preferences = {}
        self.personalization_profile = None
        
        # Behavioral tracking
        self.interaction_patterns = defaultdict(list)
        self.suggestion_performance = defaultdict(lambda: {'accepted': 0, 'rejected': 0, 'modified': 0})
        self.temporal_patterns = defaultdict(list)  # Time-based usage patterns
        
        # Adaptation parameters
        self.learning_rate = 0.1
        self.preference_decay_rate = 0.95
        self.min_preference_strength = 0.1
        self.max_preferences_per_type = 500
        
        # Load existing data
        self._load_user_data()
        
        logger.info(f"Adaptive intelligence initialized for user: {user_id}")
    
    def track_user_feedback(self, feedback_type: FeedbackType, context: Dict[str, Any],
                           original_suggestion: Any, user_action: Any,
                           confidence_before: float = 0.5) -> str:
        """Track user feedback on system suggestions"""
        
        try:
            feedback_id = f"feedback_{int(time.time())}_{len(self.feedback_history)}"
            
            feedback = UserFeedback(
                feedback_id=feedback_id,
                timestamp=datetime.now(),
                feedback_type=feedback_type,
                context=context,
                original_suggestion=original_suggestion,
                user_action=user_action,
                confidence_before=confidence_before,
                metadata={}
            )
            
            self.feedback_history.append(feedback)
            
            # Update personalization profile
            self._update_personalization_profile(feedback)
            
            # Learn from this feedback
            self._learn_from_feedback(feedback)
            
            # Update suggestion performance metrics
            self._update_suggestion_performance(feedback)
            
            logger.debug(f"Tracked user feedback: {feedback_type.value}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Failed to track user feedback: {e}")
            return ""
    
    def get_personalized_suggestions(self, suggestion_type: str, context: Dict[str, Any],
                                   base_suggestions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Get personalized suggestions based on user preferences"""
        
        try:
            if not base_suggestions:
                return []
            
            personalized_suggestions = []
            
            for suggestion, base_confidence in base_suggestions:
                # Apply user preference adjustments
                adjusted_confidence = self._apply_preference_adjustments(
                    suggestion_type, suggestion, base_confidence, context
                )
                
                # Apply behavioral pattern adjustments
                adjusted_confidence = self._apply_behavioral_adjustments(
                    suggestion_type, suggestion, adjusted_confidence, context
                )
                
                # Apply temporal pattern adjustments
                adjusted_confidence = self._apply_temporal_adjustments(
                    suggestion_type, suggestion, adjusted_confidence, context
                )
                
                personalized_suggestions.append((suggestion, adjusted_confidence))
            
            # Sort by adjusted confidence
            personalized_suggestions.sort(key=lambda x: x[1], reverse=True)
            
            # Track this personalization for learning
            self._track_personalization_event(suggestion_type, context, base_suggestions, personalized_suggestions)
            
            logger.debug(f"Generated {len(personalized_suggestions)} personalized {suggestion_type} suggestions")
            return personalized_suggestions
            
        except Exception as e:
            logger.error(f"Failed to get personalized suggestions: {e}")
            return base_suggestions
    
    def _apply_preference_adjustments(self, suggestion_type: str, suggestion: str,
                                    base_confidence: float, context: Dict[str, Any]) -> float:
        """Apply user preference adjustments to suggestion confidence"""
        
        try:
            adjusted_confidence = base_confidence
            
            # Find relevant preferences
            relevant_preferences = self._find_relevant_preferences(suggestion_type, suggestion, context)
            
            for preference in relevant_preferences:
                # Calculate preference influence
                influence = preference.strength * preference.confidence
                
                # Apply positive or negative adjustment based on success rate
                if preference.success_rate > 0.6:  # Positive preference
                    adjustment = influence * 0.3  # Max 30% boost
                    adjusted_confidence = min(1.0, adjusted_confidence + adjustment)
                elif preference.success_rate < 0.4:  # Negative preference
                    adjustment = influence * 0.3  # Max 30% reduction
                    adjusted_confidence = max(0.0, adjusted_confidence - adjustment)
            
            return adjusted_confidence
            
        except Exception as e:
            logger.warning(f"Failed to apply preference adjustments: {e}")
            return base_confidence
    
    def _apply_behavioral_adjustments(self, suggestion_type: str, suggestion: str,
                                    base_confidence: float, context: Dict[str, Any]) -> float:
        """Apply behavioral pattern adjustments"""
        
        try:
            adjusted_confidence = base_confidence
            
            # Check if user typically accepts/rejects this type of suggestion
            performance = self.suggestion_performance[suggestion_type]
            total_interactions = sum(performance.values())
            
            if total_interactions > 10:  # Enough data for behavioral adjustment
                acceptance_rate = performance['accepted'] / total_interactions
                
                if acceptance_rate > 0.7:  # User typically accepts this type
                    adjusted_confidence *= 1.1  # 10% boost
                elif acceptance_rate < 0.3:  # User typically rejects this type
                    adjusted_confidence *= 0.9  # 10% reduction
            
            # Check for specific patterns in the suggestion
            if self._matches_user_behavioral_pattern(suggestion, context):
                adjusted_confidence *= 1.15  # 15% boost for matching patterns
            
            return min(1.0, max(0.0, adjusted_confidence))
            
        except Exception as e:
            logger.warning(f"Failed to apply behavioral adjustments: {e}")
            return base_confidence
    
    def _apply_temporal_adjustments(self, suggestion_type: str, suggestion: str,
                                  base_confidence: float, context: Dict[str, Any]) -> float:
        """Apply temporal pattern adjustments based on time of day/week"""
        
        try:
            current_time = datetime.now()
            hour = current_time.hour
            weekday = current_time.weekday()
            
            # Check if user is more active with certain types at certain times
            temporal_key = f"{suggestion_type}_{hour}_{weekday}"
            
            if temporal_key in self.temporal_patterns:
                pattern_strength = len(self.temporal_patterns[temporal_key]) / 100.0  # Normalize
                if pattern_strength > 0.1:  # Significant temporal pattern
                    base_confidence *= (1.0 + pattern_strength * 0.1)  # Small temporal boost
            
            return min(1.0, base_confidence)
            
        except Exception as e:
            logger.warning(f"Failed to apply temporal adjustments: {e}")
            return base_confidence
    
    def _find_relevant_preferences(self, suggestion_type: str, suggestion: str,
                                 context: Dict[str, Any]) -> List[UserPreference]:
        """Find user preferences relevant to the current suggestion"""
        
        relevant_preferences = []
        
        try:
            for preference_id, preference in self.user_preferences.items():
                if preference.preference_type == suggestion_type:
                    # Check if preference pattern matches the suggestion
                    if self._preference_matches_suggestion(preference, suggestion, context):
                        relevant_preferences.append(preference)
            
            # Sort by strength and confidence
            relevant_preferences.sort(key=lambda p: p.strength * p.confidence, reverse=True)
            
            return relevant_preferences[:5]  # Top 5 most relevant
            
        except Exception as e:
            logger.warning(f"Failed to find relevant preferences: {e}")
            return []
    
    def _preference_matches_suggestion(self, preference: UserPreference, suggestion: str,
                                     context: Dict[str, Any]) -> bool:
        """Check if a preference pattern matches the current suggestion"""
        
        try:
            # Simple pattern matching - can be enhanced with more sophisticated matching
            pattern_lower = preference.pattern.lower()
            suggestion_lower = suggestion.lower()
            
            # Direct match
            if pattern_lower == suggestion_lower:
                return True
            
            # Substring match
            if pattern_lower in suggestion_lower or suggestion_lower in pattern_lower:
                return True
            
            # Context-based matching
            url = context.get('url', '').lower()
            title = context.get('title', '').lower()
            
            if pattern_lower in url or pattern_lower in title:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to match preference pattern: {e}")
            return False
    
    def _matches_user_behavioral_pattern(self, suggestion: str, context: Dict[str, Any]) -> bool:
        """Check if suggestion matches user's behavioral patterns"""
        
        try:
            # Check interaction patterns
            url = context.get('url', '')
            domain = url.split('/')[2] if '://' in url else ''
            
            # Check if user frequently interacts with this domain
            domain_interactions = self.interaction_patterns.get(f"domain:{domain}", [])
            if len(domain_interactions) > 5:  # User has significant history with this domain
                return True
            
            # Check if suggestion matches frequently used patterns
            suggestion_lower = suggestion.lower()
            for pattern_key, interactions in self.interaction_patterns.items():
                if len(interactions) > 3 and pattern_key.split(':')[-1].lower() in suggestion_lower:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to check behavioral patterns: {e}")
            return False
    
    def _learn_from_feedback(self, feedback: UserFeedback) -> None:
        """Learn user preferences from feedback"""
        
        try:
            context = feedback.context
            suggestion_type = context.get('suggestion_type', 'unknown')
            
            if feedback.feedback_type == FeedbackType.SUGGESTION_ACCEPTED:
                self._reinforce_positive_preference(feedback, suggestion_type)
            elif feedback.feedback_type == FeedbackType.SUGGESTION_REJECTED:
                self._reinforce_negative_preference(feedback, suggestion_type)
            elif feedback.feedback_type == FeedbackType.SUGGESTION_MODIFIED:
                self._learn_from_modification(feedback, suggestion_type)
            elif feedback.feedback_type in [FeedbackType.CATEGORY_CORRECTED, FeedbackType.TAG_CORRECTED]:
                self._learn_from_correction(feedback, suggestion_type)
            
            # Update behavioral patterns
            self._update_behavioral_patterns(feedback)
            
        except Exception as e:
            logger.error(f"Failed to learn from feedback: {e}")
    
    def _reinforce_positive_preference(self, feedback: UserFeedback, suggestion_type: str) -> None:
        """Reinforce positive preference based on accepted suggestion"""
        
        try:
            suggestion = str(feedback.original_suggestion)
            context = feedback.context
            
            # Create or update preference
            preference_id = f"{suggestion_type}:{suggestion}"
            
            if preference_id in self.user_preferences:
                preference = self.user_preferences[preference_id]
                preference.strength = min(1.0, preference.strength + self.learning_rate)
                preference.usage_count += 1
                preference.success_rate = (preference.success_rate * (preference.usage_count - 1) + 1.0) / preference.usage_count
                preference.last_reinforced = datetime.now()
            else:
                preference = UserPreference(
                    preference_id=preference_id,
                    preference_type=suggestion_type,
                    pattern=suggestion,
                    strength=self.learning_rate,
                    confidence=0.6,
                    usage_count=1,
                    success_rate=1.0,
                    last_reinforced=datetime.now(),
                    created_at=datetime.now(),
                    examples=[context]
                )
                self.user_preferences[preference_id] = preference
            
            logger.debug(f"Reinforced positive preference: {preference_id}")
            
        except Exception as e:
            logger.warning(f"Failed to reinforce positive preference: {e}")
    
    def _reinforce_negative_preference(self, feedback: UserFeedback, suggestion_type: str) -> None:
        """Reinforce negative preference based on rejected suggestion"""
        
        try:
            suggestion = str(feedback.original_suggestion)
            
            # Create or update negative preference
            preference_id = f"{suggestion_type}:NOT:{suggestion}"
            
            if preference_id in self.user_preferences:
                preference = self.user_preferences[preference_id]
                preference.strength = min(1.0, preference.strength + self.learning_rate)
                preference.usage_count += 1
                preference.success_rate = (preference.success_rate * (preference.usage_count - 1) + 0.0) / preference.usage_count
                preference.last_reinforced = datetime.now()
            else:
                preference = UserPreference(
                    preference_id=preference_id,
                    preference_type=suggestion_type,
                    pattern=f"NOT:{suggestion}",
                    strength=self.learning_rate,
                    confidence=0.6,
                    usage_count=1,
                    success_rate=0.0,
                    last_reinforced=datetime.now(),
                    created_at=datetime.now(),
                    examples=[feedback.context]
                )
                self.user_preferences[preference_id] = preference
            
            logger.debug(f"Reinforced negative preference: {preference_id}")
            
        except Exception as e:
            logger.warning(f"Failed to reinforce negative preference: {e}")
    
    def _learn_from_modification(self, feedback: UserFeedback, suggestion_type: str) -> None:
        """Learn from user modifications to suggestions"""
        
        try:
            original = str(feedback.original_suggestion)
            modified = str(feedback.user_action)
            
            # Create preference for the modified version
            self._reinforce_positive_preference(
                UserFeedback(
                    feedback_id=feedback.feedback_id + "_modified",
                    timestamp=feedback.timestamp,
                    feedback_type=FeedbackType.SUGGESTION_ACCEPTED,
                    context=feedback.context,
                    original_suggestion=modified,
                    user_action=modified,
                    confidence_before=feedback.confidence_before
                ),
                suggestion_type
            )
            
            # Weaken preference for the original version
            self._reinforce_negative_preference(feedback, suggestion_type)
            
            logger.debug(f"Learned from modification: {original} -> {modified}")
            
        except Exception as e:
            logger.warning(f"Failed to learn from modification: {e}")
    
    def _learn_from_correction(self, feedback: UserFeedback, suggestion_type: str) -> None:
        """Learn from user corrections"""
        
        try:
            # Similar to modification learning
            self._learn_from_modification(feedback, suggestion_type)
            
            # Additionally, learn contextual patterns
            context = feedback.context
            corrected_value = str(feedback.user_action)
            
            # Learn context -> correction patterns
            url = context.get('url', '')
            if url:
                domain = url.split('/')[2] if '://' in url else ''
                if domain:
                    pattern_id = f"domain_correction:{domain}:{corrected_value}"
                    self._create_or_update_preference(pattern_id, suggestion_type, corrected_value, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to learn from correction: {e}")
    
    def _create_or_update_preference(self, preference_id: str, preference_type: str,
                                   pattern: str, success_rate: float) -> None:
        """Create or update a user preference"""
        
        try:
            if preference_id in self.user_preferences:
                preference = self.user_preferences[preference_id]
                preference.strength = min(1.0, preference.strength + self.learning_rate)
                preference.usage_count += 1
                preference.success_rate = (preference.success_rate + success_rate) / 2
                preference.last_reinforced = datetime.now()
            else:
                preference = UserPreference(
                    preference_id=preference_id,
                    preference_type=preference_type,
                    pattern=pattern,
                    strength=self.learning_rate,
                    confidence=0.6,
                    usage_count=1,
                    success_rate=success_rate,
                    last_reinforced=datetime.now(),
                    created_at=datetime.now()
                )
                self.user_preferences[preference_id] = preference
            
        except Exception as e:
            logger.warning(f"Failed to create/update preference: {e}")
    
    def _update_behavioral_patterns(self, feedback: UserFeedback) -> None:
        """Update behavioral patterns based on feedback"""
        
        try:
            context = feedback.context
            timestamp = feedback.timestamp
            
            # Track domain interactions
            url = context.get('url', '')
            if url:
                domain = url.split('/')[2] if '://' in url else ''
                if domain:
                    self.interaction_patterns[f"domain:{domain}"].append(timestamp)
            
            # Track temporal patterns
            hour = timestamp.hour
            weekday = timestamp.weekday()
            suggestion_type = context.get('suggestion_type', 'unknown')
            
            temporal_key = f"{suggestion_type}_{hour}_{weekday}"
            self.temporal_patterns[temporal_key].append(timestamp)
            
            # Limit pattern history to prevent memory bloat
            for key in list(self.interaction_patterns.keys()):
                if len(self.interaction_patterns[key]) > 100:
                    self.interaction_patterns[key] = self.interaction_patterns[key][-50:]
            
            for key in list(self.temporal_patterns.keys()):
                if len(self.temporal_patterns[key]) > 100:
                    self.temporal_patterns[key] = self.temporal_patterns[key][-50:]
            
        except Exception as e:
            logger.warning(f"Failed to update behavioral patterns: {e}")
    
    def _update_suggestion_performance(self, feedback: UserFeedback) -> None:
        """Update suggestion performance metrics"""
        
        try:
            suggestion_type = feedback.context.get('suggestion_type', 'unknown')
            
            if feedback.feedback_type == FeedbackType.SUGGESTION_ACCEPTED:
                self.suggestion_performance[suggestion_type]['accepted'] += 1
            elif feedback.feedback_type == FeedbackType.SUGGESTION_REJECTED:
                self.suggestion_performance[suggestion_type]['rejected'] += 1
            elif feedback.feedback_type == FeedbackType.SUGGESTION_MODIFIED:
                self.suggestion_performance[suggestion_type]['modified'] += 1
            
        except Exception as e:
            logger.warning(f"Failed to update suggestion performance: {e}")
    
    def _update_personalization_profile(self, feedback: UserFeedback) -> None:
        """Update the user's personalization profile"""
        
        try:
            if not self.personalization_profile:
                self.personalization_profile = PersonalizationProfile(
                    user_id=self.user_id,
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    total_interactions=0
                )
            
            profile = self.personalization_profile
            profile.last_updated = datetime.now()
            profile.total_interactions += 1
            
            # Add to interaction history (keep last 1000)
            interaction = {
                'timestamp': feedback.timestamp.isoformat(),
                'feedback_type': feedback.feedback_type.value,
                'suggestion_type': feedback.context.get('suggestion_type', 'unknown'),
                'confidence_before': feedback.confidence_before
            }
            
            profile.interaction_history.append(interaction)
            if len(profile.interaction_history) > 1000:
                profile.interaction_history = profile.interaction_history[-500:]
            
            # Update suggestion accuracy
            suggestion_type = feedback.context.get('suggestion_type', 'unknown')
            if suggestion_type not in profile.suggestion_accuracy:
                profile.suggestion_accuracy[suggestion_type] = 0.5
            
            # Update accuracy based on feedback
            if feedback.feedback_type == FeedbackType.SUGGESTION_ACCEPTED:
                profile.suggestion_accuracy[suggestion_type] = min(1.0, profile.suggestion_accuracy[suggestion_type] + 0.05)
            elif feedback.feedback_type == FeedbackType.SUGGESTION_REJECTED:
                profile.suggestion_accuracy[suggestion_type] = max(0.0, profile.suggestion_accuracy[suggestion_type] - 0.05)
            
        except Exception as e:
            logger.warning(f"Failed to update personalization profile: {e}")
    
    def _track_personalization_event(self, suggestion_type: str, context: Dict[str, Any],
                                   base_suggestions: List[Tuple[str, float]],
                                   personalized_suggestions: List[Tuple[str, float]]) -> None:
        """Track personalization events for analysis"""
        
        try:
            # Calculate personalization impact
            base_top = base_suggestions[0][0] if base_suggestions else ""
            personalized_top = personalized_suggestions[0][0] if personalized_suggestions else ""
            
            personalization_changed = base_top != personalized_top
            
            # Store event for future analysis
            event = {
                'timestamp': datetime.now().isoformat(),
                'suggestion_type': suggestion_type,
                'personalization_changed': personalization_changed,
                'base_top_suggestion': base_top,
                'personalized_top_suggestion': personalized_top,
                'context_domain': context.get('url', '').split('/')[2] if '://' in context.get('url', '') else ''
            }
            
            # Add to behavioral patterns for analysis
            self.interaction_patterns[f"personalization:{suggestion_type}"].append(event)
            
        except Exception as e:
            logger.warning(f"Failed to track personalization event: {e}")
    
    def decay_preferences(self) -> int:
        """Apply decay to unused preferences and remove weak ones"""
        
        try:
            current_time = datetime.now()
            removed_count = 0
            
            for preference_id in list(self.user_preferences.keys()):
                preference = self.user_preferences[preference_id]
                
                # Calculate days since last reinforcement
                days_since_use = (current_time - preference.last_reinforced).days
                
                if days_since_use > 30:  # Apply decay to preferences not used in 30 days
                    decay_factor = self.preference_decay_rate ** (days_since_use / 30)
                    preference.strength *= decay_factor
                    
                    # Remove very weak preferences
                    if preference.strength < self.min_preference_strength:
                        del self.user_preferences[preference_id]
                        removed_count += 1
            
            # Limit preferences per type
            self._limit_preferences_per_type()
            
            logger.debug(f"Preference decay: removed {removed_count} weak preferences")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to decay preferences: {e}")
            return 0
    
    def _limit_preferences_per_type(self) -> None:
        """Limit number of preferences per type to prevent memory bloat"""
        
        try:
            preferences_by_type = defaultdict(list)
            
            # Group preferences by type
            for preference_id, preference in self.user_preferences.items():
                preferences_by_type[preference.preference_type].append((preference_id, preference))
            
            # Limit each type
            for preference_type, preferences in preferences_by_type.items():
                if len(preferences) > self.max_preferences_per_type:
                    # Sort by strength and keep the strongest
                    preferences.sort(key=lambda x: x[1].strength, reverse=True)
                    
                    # Remove weakest preferences
                    for preference_id, _ in preferences[self.max_preferences_per_type:]:
                        del self.user_preferences[preference_id]
                    
                    logger.debug(f"Limited {preference_type} preferences to {self.max_preferences_per_type}")
            
        except Exception as e:
            logger.warning(f"Failed to limit preferences per type: {e}")
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive adaptation statistics"""
        
        try:
            # Calculate preference statistics
            preference_stats = defaultdict(int)
            strong_preferences = 0
            
            for preference in self.user_preferences.values():
                preference_stats[preference.preference_type] += 1
                if preference.strength > 0.7:
                    strong_preferences += 1
            
            # Calculate suggestion performance
            performance_stats = {}
            for suggestion_type, performance in self.suggestion_performance.items():
                total = sum(performance.values())
                if total > 0:
                    performance_stats[suggestion_type] = {
                        'acceptance_rate': performance['accepted'] / total,
                        'rejection_rate': performance['rejected'] / total,
                        'modification_rate': performance['modified'] / total,
                        'total_interactions': total
                    }
            
            # Calculate recent activity
            recent_feedback = [f for f in self.feedback_history if (datetime.now() - f.timestamp).days < 7]
            
            return {
                'user_id': self.user_id,
                'total_preferences': len(self.user_preferences),
                'strong_preferences': strong_preferences,
                'preference_breakdown': dict(preference_stats),
                'total_feedback_items': len(self.feedback_history),
                'recent_feedback_count': len(recent_feedback),
                'suggestion_performance': performance_stats,
                'behavioral_patterns_count': len(self.interaction_patterns),
                'temporal_patterns_count': len(self.temporal_patterns),
                'personalization_profile': {
                    'total_interactions': self.personalization_profile.total_interactions if self.personalization_profile else 0,
                    'suggestion_accuracy': dict(self.personalization_profile.suggestion_accuracy) if self.personalization_profile else {},
                    'created_at': self.personalization_profile.created_at.isoformat() if self.personalization_profile else None,
                    'last_updated': self.personalization_profile.last_updated.isoformat() if self.personalization_profile else None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get adaptation statistics: {e}")
            return {'error': str(e)}
    
    def export_user_data(self) -> Dict[str, Any]:
        """Export user adaptation data for backup"""
        
        try:
            return {
                'user_id': self.user_id,
                'export_timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'user_preferences': {
                    preference_id: {
                        'preference_id': pref.preference_id,
                        'preference_type': pref.preference_type,
                        'pattern': pref.pattern,
                        'strength': pref.strength,
                        'confidence': pref.confidence,
                        'usage_count': pref.usage_count,
                        'success_rate': pref.success_rate,
                        'last_reinforced': pref.last_reinforced.isoformat(),
                        'created_at': pref.created_at.isoformat(),
                        'examples': pref.examples
                    }
                    for preference_id, pref in self.user_preferences.items()
                },
                'suggestion_performance': dict(self.suggestion_performance),
                'interaction_patterns': {k: [t.isoformat() if isinstance(t, datetime) else t for t in v] 
                                       for k, v in self.interaction_patterns.items()},
                'temporal_patterns': {k: [t.isoformat() if isinstance(t, datetime) else t for t in v] 
                                    for k, v in self.temporal_patterns.items()},
                'personalization_profile': {
                    'user_id': self.personalization_profile.user_id,
                    'created_at': self.personalization_profile.created_at.isoformat(),
                    'last_updated': self.personalization_profile.last_updated.isoformat(),
                    'total_interactions': self.personalization_profile.total_interactions,
                    'suggestion_accuracy': self.personalization_profile.suggestion_accuracy,
                    'interaction_history': self.personalization_profile.interaction_history
                } if self.personalization_profile else None
            }
            
        except Exception as e:
            logger.error(f"Failed to export user data: {e}")
            return {'error': str(e)}
    
    def import_user_data(self, data: Dict[str, Any]) -> bool:
        """Import user adaptation data from backup"""
        
        try:
            # Import user preferences
            if 'user_preferences' in data:
                for preference_id, pref_data in data['user_preferences'].items():
                    self.user_preferences[preference_id] = UserPreference(
                        preference_id=pref_data['preference_id'],
                        preference_type=pref_data['preference_type'],
                        pattern=pref_data['pattern'],
                        strength=pref_data['strength'],
                        confidence=pref_data['confidence'],
                        usage_count=pref_data['usage_count'],
                        success_rate=pref_data['success_rate'],
                        last_reinforced=datetime.fromisoformat(pref_data['last_reinforced']),
                        created_at=datetime.fromisoformat(pref_data['created_at']),
                        examples=pref_data.get('examples', [])
                    )
            
            # Import performance data
            if 'suggestion_performance' in data:
                self.suggestion_performance.update(data['suggestion_performance'])
            
            # Import patterns
            if 'interaction_patterns' in data:
                for key, values in data['interaction_patterns'].items():
                    self.interaction_patterns[key] = [
                        datetime.fromisoformat(v) if isinstance(v, str) and 'T' in v else v
                        for v in values
                    ]
            
            if 'temporal_patterns' in data:
                for key, values in data['temporal_patterns'].items():
                    self.temporal_patterns[key] = [
                        datetime.fromisoformat(v) if isinstance(v, str) and 'T' in v else v
                        for v in values
                    ]
            
            # Import personalization profile
            if 'personalization_profile' in data and data['personalization_profile']:
                profile_data = data['personalization_profile']
                self.personalization_profile = PersonalizationProfile(
                    user_id=profile_data['user_id'],
                    created_at=datetime.fromisoformat(profile_data['created_at']),
                    last_updated=datetime.fromisoformat(profile_data['last_updated']),
                    total_interactions=profile_data['total_interactions'],
                    suggestion_accuracy=profile_data.get('suggestion_accuracy', {}),
                    interaction_history=profile_data.get('interaction_history', [])
                )
            
            logger.info(f"Imported user data: {len(self.user_preferences)} preferences")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import user data: {e}")
            return False
    
    def _save_user_data(self) -> None:
        """Save user adaptation data to disk"""
        
        try:
            user_data = self.export_user_data()
            
            user_file = self.data_dir / f'adaptive_intelligence_{self.user_id}.json'
            with open(user_file, 'w', encoding='utf-8') as f:
                json.dump(user_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved user adaptation data to {user_file}")
            
        except Exception as e:
            logger.error(f"Failed to save user data: {e}")
    
    def _load_user_data(self) -> None:
        """Load user adaptation data from disk"""
        
        try:
            user_file = self.data_dir / f'adaptive_intelligence_{self.user_id}.json'
            
            if not user_file.exists():
                logger.info(f"No existing user data found for {self.user_id}")
                return
            
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            success = self.import_user_data(user_data)
            
            if success:
                logger.info(f"Loaded user adaptation data for {self.user_id}")
            else:
                logger.warning(f"Failed to load user adaptation data for {self.user_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load user data: {e}")
    
    def save_data(self) -> None:
        """Save all user data to disk"""
        self._save_user_data()
    
    def __del__(self):
        """Destructor to save data on cleanup"""
        try:
            self._save_user_data()
        except:
            pass  # Ignore errors during cleanup