"""Pattern learning system for continuous improvement"""

import json
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

from ..utils.logging_utils import get_logger
from ..utils.url_utils import UrlUtils
from ..utils.text_utils import TextUtils

logger = get_logger(__name__)


class PatternLearner:
    """Learn and adapt patterns from user behavior and bookmark data"""

    def __init__(self, data_dir: str = "data"):
        """Initialize pattern learner"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Pattern storage
        self.url_category_patterns = defaultdict(list)
        self.content_tag_patterns = defaultdict(list)
        self.user_preferences = {}
        self.domain_collection_mapping = defaultdict(Counter)
        self.tag_content_associations = defaultdict(Counter)

        # Learning statistics
        self.learning_sessions = []
        self.pattern_strength = defaultdict(float)
        self.accuracy_metrics = defaultdict(list)

        # Load existing patterns
        self._load_learned_patterns()

        logger.info("Pattern learner initialized")

    def learn_from_bookmark_history(self, bookmarks: List[Dict]) -> Dict[str, Any]:
        """Learn patterns from existing bookmark categorization"""

        logger.info(f"Learning patterns from {len(bookmarks)} bookmarks")

        learning_stats = {
            "bookmarks_processed": 0,
            "domains_learned": 0,
            "tag_patterns_learned": 0,
            "collection_associations_learned": 0,
            "errors": [],
        }

        for bookmark in bookmarks:
            try:
                url = bookmark.get("url", "")
                collection_name = bookmark.get("collection_name", "")
                collection_id = bookmark.get("collection_id")
                tags = bookmark.get("tags", [])
                title = bookmark.get("name", "")
                content = bookmark.get("content", {}).get("text_content", "") or ""

                if not url:
                    continue

                # Extract domain and path patterns
                domain = UrlUtils.extract_domain(url)
                path_segments = UrlUtils.extract_path_segments(url)

                # Learn domain -> collection associations
                if domain and collection_name:
                    self.url_category_patterns[domain].append(collection_name)
                    self.domain_collection_mapping[domain][collection_name] += 1
                    learning_stats["domains_learned"] += 1

                # Learn content -> tag associations
                if content or title:
                    text_content = f"{title} {content}"
                    self._learn_content_tag_associations(text_content, tags)
                    learning_stats["tag_patterns_learned"] += 1

                # Learn URL patterns
                self._learn_url_patterns(url, collection_name, tags)

                learning_stats["bookmarks_processed"] += 1

            except Exception as e:
                error_msg = f"Failed to learn from bookmark {bookmark.get('id', 'unknown')}: {e}"
                learning_stats["errors"].append(error_msg)
                logger.warning(error_msg)
                continue

        # Calculate pattern strengths
        self._calculate_pattern_strengths()

        # Save learned patterns
        self._save_learned_patterns()

        # Record learning session
        session_info = {
            "timestamp": datetime.now().isoformat(),
            "bookmarks_processed": learning_stats["bookmarks_processed"],
            "patterns_learned": len(self.url_category_patterns),
            "stats": learning_stats,
        }
        self.learning_sessions.append(session_info)

        logger.info(f"Learning completed: {learning_stats}")
        return learning_stats

    def _learn_content_tag_associations(
        self, text_content: str, tags: List[Any]
    ) -> None:
        """Learn which content patterns lead to specific tags"""

        try:
            # Extract keywords from content
            keywords = TextUtils.extract_keywords(text_content.lower())

            # Extract tag names
            tag_names = []
            for tag in tags:
                if isinstance(tag, dict):
                    tag_name = tag.get("name", "")
                else:
                    tag_name = str(tag)

                if tag_name:
                    tag_names.append(tag_name)

            # Associate keywords with tags
            for keyword in keywords[:10]:  # Limit to top 10 keywords
                for tag_name in tag_names:
                    self.content_tag_patterns[keyword].append(tag_name)
                    self.tag_content_associations[tag_name][keyword] += 1

        except Exception as e:
            logger.warning(f"Failed to learn content-tag associations: {e}")

    def _learn_url_patterns(
        self, url: str, collection_name: str, tags: List[Any]
    ) -> None:
        """Learn URL patterns for categorization"""

        try:
            # Extract URL components
            domain = UrlUtils.extract_domain(url)
            path_segments = UrlUtils.extract_path_segments(url)
            url_keywords = UrlUtils.extract_url_keywords(url)

            # Learn domain patterns
            if domain and collection_name:
                pattern_key = f"domain:{domain}"
                self.pattern_strength[f"{pattern_key}->{collection_name}"] += 1

            # Learn path patterns
            for segment in path_segments:
                if len(segment) > 2:  # Ignore very short segments
                    pattern_key = f"path:{segment}"
                    if collection_name:
                        self.pattern_strength[f"{pattern_key}->{collection_name}"] += 1

            # Learn URL keyword patterns
            for keyword in url_keywords:
                pattern_key = f"url_keyword:{keyword}"
                if collection_name:
                    self.pattern_strength[f"{pattern_key}->{collection_name}"] += 1

        except Exception as e:
            logger.warning(f"Failed to learn URL patterns for {url}: {e}")

    def _calculate_pattern_strengths(self) -> None:
        """Calculate strength scores for learned patterns"""

        # Calculate domain -> collection strength
        for domain, collections in self.domain_collection_mapping.items():
            total_count = sum(collections.values())
            for collection, count in collections.items():
                strength = count / total_count
                self.pattern_strength[f"domain:{domain}->{collection}"] = strength

        # Calculate tag -> content strength
        for tag, keywords in self.tag_content_associations.items():
            total_count = sum(keywords.values())
            for keyword, count in keywords.items():
                strength = count / total_count
                self.pattern_strength[f"tag:{tag}<-{keyword}"] = strength

    def predict_category_for_url(self, url: str) -> List[Tuple[str, float]]:
        """Predict category for URL based on learned patterns"""

        predictions = []

        try:
            domain = UrlUtils.extract_domain(url)
            path_segments = UrlUtils.extract_path_segments(url)
            url_keywords = UrlUtils.extract_url_keywords(url)

            # Check domain patterns
            if domain in self.domain_collection_mapping:
                for collection, count in self.domain_collection_mapping[domain].items():
                    total_domain_count = sum(
                        self.domain_collection_mapping[domain].values()
                    )
                    confidence = count / total_domain_count
                    predictions.append((collection, confidence))

            # Check path patterns
            for segment in path_segments:
                pattern_key = f"path:{segment}"
                for pattern, strength in self.pattern_strength.items():
                    if pattern.startswith(pattern_key + "->"):
                        collection = pattern.split("->")[1]
                        predictions.append(
                            (collection, strength * 0.7)
                        )  # Lower weight for path

            # Check URL keyword patterns
            for keyword in url_keywords:
                pattern_key = f"url_keyword:{keyword}"
                for pattern, strength in self.pattern_strength.items():
                    if pattern.startswith(pattern_key + "->"):
                        collection = pattern.split("->")[1]
                        predictions.append(
                            (collection, strength * 0.5)
                        )  # Lower weight for keywords

            # Aggregate predictions
            aggregated = defaultdict(float)
            for collection, confidence in predictions:
                aggregated[collection] += confidence

            # Sort by confidence
            sorted_predictions = sorted(
                aggregated.items(), key=lambda x: x[1], reverse=True
            )

            return sorted_predictions[:5]  # Return top 5 predictions

        except Exception as e:
            logger.error(f"Failed to predict category for {url}: {e}")
            return []

    def predict_tags_for_content(
        self, content: str, limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Predict tags for content based on learned patterns"""

        try:
            keywords = TextUtils.extract_keywords(content.lower())
            tag_scores = defaultdict(float)

            # Score tags based on content keywords
            for keyword in keywords:
                if keyword in self.content_tag_patterns:
                    tag_counts = Counter(self.content_tag_patterns[keyword])
                    total_count = sum(tag_counts.values())

                    for tag, count in tag_counts.items():
                        confidence = count / total_count
                        tag_scores[tag] += confidence

            # Normalize scores
            if tag_scores:
                max_score = max(tag_scores.values())
                normalized_scores = [
                    (tag, score / max_score) for tag, score in tag_scores.items()
                ]

                # Sort by score and return top predictions
                sorted_predictions = sorted(
                    normalized_scores, key=lambda x: x[1], reverse=True
                )
                return sorted_predictions[:limit]

            return []

        except Exception as e:
            logger.error(f"Failed to predict tags for content: {e}")
            return []

    def track_user_feedback(
        self, feedback_type: str, original: str, modified: str, context: Dict[str, Any]
    ) -> None:
        """Track user feedback to improve predictions"""

        try:
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": feedback_type,
                "original": original,
                "modified": modified,
                "context": context,
            }

            # Store feedback for analysis
            feedback_key = f"feedback:{feedback_type}"
            if feedback_key not in self.user_preferences:
                self.user_preferences[feedback_key] = []

            self.user_preferences[feedback_key].append(feedback_entry)

            # Adjust pattern strengths based on feedback
            if feedback_type == "category_correction":
                self._adjust_category_patterns(original, modified, context)
            elif feedback_type == "tag_correction":
                self._adjust_tag_patterns(original, modified, context)

            logger.debug(f"Recorded user feedback: {feedback_type}")

        except Exception as e:
            logger.error(f"Failed to track user feedback: {e}")

    def _adjust_category_patterns(
        self, original: str, modified: str, context: Dict[str, Any]
    ) -> None:
        """Adjust category prediction patterns based on user feedback"""

        try:
            url = context.get("url", "")
            if not url:
                return

            domain = UrlUtils.extract_domain(url)

            # Decrease strength for incorrect prediction
            if original and domain:
                wrong_pattern = f"domain:{domain}->{original}"
                if wrong_pattern in self.pattern_strength:
                    self.pattern_strength[wrong_pattern] *= 0.9  # Reduce by 10%

            # Increase strength for correct category
            if modified and domain:
                correct_pattern = f"domain:{domain}->{modified}"
                self.pattern_strength[correct_pattern] = (
                    self.pattern_strength.get(correct_pattern, 0) + 0.1
                )

                # Also update domain collection mapping
                self.domain_collection_mapping[domain][modified] += 1

        except Exception as e:
            logger.warning(f"Failed to adjust category patterns: {e}")

    def _adjust_tag_patterns(
        self, original: str, modified: str, context: Dict[str, Any]
    ) -> None:
        """Adjust tag prediction patterns based on user feedback"""

        try:
            content = context.get("content", "")
            if not content:
                return

            keywords = TextUtils.extract_keywords(content.lower())

            # Adjust patterns for modified tags
            if modified:
                for keyword in keywords[:5]:  # Top 5 keywords
                    self.content_tag_patterns[keyword].append(modified)
                    self.tag_content_associations[modified][keyword] += 1

        except Exception as e:
            logger.warning(f"Failed to adjust tag patterns: {e}")

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""

        return {
            "total_sessions": len(self.learning_sessions),
            "url_patterns": len(self.url_category_patterns),
            "content_patterns": len(self.content_tag_patterns),
            "domain_mappings": len(self.domain_collection_mapping),
            "pattern_strengths": len(self.pattern_strength),
            "user_feedback_entries": sum(
                len(feedback) for feedback in self.user_preferences.values()
            ),
            "last_learning_session": (
                self.learning_sessions[-1] if self.learning_sessions else None
            ),
            "top_domains": dict(
                Counter(
                    {
                        domain: sum(collections.values())
                        for domain, collections in self.domain_collection_mapping.items()
                    }
                ).most_common(10)
            ),
            "top_patterns": dict(Counter(self.pattern_strength).most_common(10)),
        }

    def export_learned_patterns(self) -> Dict[str, Any]:
        """Export learned patterns for backup or sharing"""

        return {
            "url_category_patterns": dict(self.url_category_patterns),
            "content_tag_patterns": dict(self.content_tag_patterns),
            "domain_collection_mapping": {
                domain: dict(collections)
                for domain, collections in self.domain_collection_mapping.items()
            },
            "tag_content_associations": {
                tag: dict(keywords)
                for tag, keywords in self.tag_content_associations.items()
            },
            "pattern_strength": dict(self.pattern_strength),
            "user_preferences": self.user_preferences,
            "learning_sessions": self.learning_sessions,
            "export_timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

    def import_learned_patterns(self, patterns: Dict[str, Any]) -> bool:
        """Import previously learned patterns"""

        try:
            if "url_category_patterns" in patterns:
                self.url_category_patterns.update(patterns["url_category_patterns"])

            if "content_tag_patterns" in patterns:
                self.content_tag_patterns.update(patterns["content_tag_patterns"])

            if "domain_collection_mapping" in patterns:
                for domain, collections in patterns[
                    "domain_collection_mapping"
                ].items():
                    self.domain_collection_mapping[domain].update(collections)

            if "pattern_strength" in patterns:
                self.pattern_strength.update(patterns["pattern_strength"])

            if "user_preferences" in patterns:
                self.user_preferences.update(patterns["user_preferences"])

            if "learning_sessions" in patterns:
                self.learning_sessions.extend(patterns["learning_sessions"])

            logger.info(
                f"Imported learned patterns from version {patterns.get('version', 'unknown')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to import learned patterns: {e}")
            return False

    def _save_learned_patterns(self) -> None:
        """Save learned patterns to disk"""

        try:
            patterns_file = self.data_dir / "learned_patterns.json"
            patterns = self.export_learned_patterns()

            with open(patterns_file, "w", encoding="utf-8") as f:
                json.dump(patterns, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved learned patterns to {patterns_file}")

        except Exception as e:
            logger.error(f"Failed to save learned patterns: {e}")

    def _load_learned_patterns(self) -> None:
        """Load previously learned patterns from disk"""

        try:
            patterns_file = self.data_dir / "learned_patterns.json"

            if patterns_file.exists():
                with open(patterns_file, "r", encoding="utf-8") as f:
                    patterns = json.load(f)

                self.import_learned_patterns(patterns)
                logger.info(f"Loaded learned patterns from {patterns_file}")
            else:
                logger.info("No existing learned patterns found")

        except Exception as e:
            logger.warning(f"Failed to load learned patterns: {e}")
