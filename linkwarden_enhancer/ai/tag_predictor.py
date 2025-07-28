"""Tag prediction system using machine learning for intelligent tagging"""

import numpy as np
import pickle
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, multilabel_confusion_matrix
    from sklearn.preprocessing import MultiLabelBinarizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..utils.logging_utils import get_logger
from ..utils.text_utils import TextUtils

logger = get_logger(__name__)


@dataclass
class TagPrediction:
    """Tag prediction result"""
    tag: str
    confidence: float
    source: str  # 'ml_model', 'pattern', 'frequency'


@dataclass
class TrainingResult:
    """Result of model training"""
    success: bool
    samples_trained: int
    unique_tags: int
    model_accuracy: float
    training_time: float
    errors: List[str]


class TagPredictor:
    """Machine learning-based tag prediction system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize tag predictor"""
        self.config = config
        self.tag_config = config.get('tag_prediction', {})
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        # Model configuration
        self.models_dir = Path(config.get('directories', {}).get('models_dir', 'models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Prediction settings
        self.min_confidence = self.tag_config.get('min_confidence', 0.3)
        self.max_predictions = self.tag_config.get('max_predictions', 10)
        self.min_tag_frequency = self.tag_config.get('min_tag_frequency', 3)
        
        # Text processing settings
        self.max_features = self.tag_config.get('max_features', 5000)
        self.ngram_range = tuple(self.tag_config.get('ngram_range', [1, 2]))
        self.min_df = self.tag_config.get('min_df', 2)
        self.max_df = self.tag_config.get('max_df', 0.8)
        
        # Models
        self.vectorizer = None
        self.label_binarizer = None
        self.classifier = None
        
        # Training data
        self.tag_frequencies = Counter()
        self.tag_patterns = defaultdict(list)
        self.is_trained = False
        
        # Statistics
        self.training_stats = {}
        
        logger.info("Tag predictor initialized")
    
    def train_from_bookmarks(self, bookmarks: List[Dict[str, Any]]) -> TrainingResult:
        """Train tag prediction model from bookmark data"""
        
        import time
        start_time = time.time()
        
        try:
            logger.info(f"Training tag predictor from {len(bookmarks)} bookmarks")
            
            # Extract training data
            texts, tag_sets = self._extract_training_data(bookmarks)
            
            if len(texts) < 10:
                error_msg = "Insufficient training data (need at least 10 bookmarks with tags)"
                logger.error(error_msg)
                return TrainingResult(
                    success=False,
                    samples_trained=0,
                    unique_tags=0,
                    model_accuracy=0.0,
                    training_time=time.time() - start_time,
                    errors=[error_msg]
                )
            
            # Filter tags by frequency
            filtered_texts, filtered_tag_sets = self._filter_by_frequency(texts, tag_sets)
            
            if len(filtered_texts) < 5:
                error_msg = "Insufficient training data after frequency filtering"
                logger.error(error_msg)
                return TrainingResult(
                    success=False,
                    samples_trained=0,
                    unique_tags=0,
                    model_accuracy=0.0,
                    training_time=time.time() - start_time,
                    errors=[error_msg]
                )
            
            # Prepare features and labels
            X, y = self._prepare_training_data(filtered_texts, filtered_tag_sets)
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.classifier = MultiOutputClassifier(MultinomialNB(alpha=0.1))
            self.classifier.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.classifier.predict(X_test)
            accuracy = self._calculate_accuracy(y_test, y_pred)
            
            # Update statistics
            unique_tags = len(self.label_binarizer.classes_)
            training_time = time.time() - start_time
            
            self.training_stats = {
                'samples_trained': len(filtered_texts),
                'unique_tags': unique_tags,
                'model_accuracy': accuracy,
                'training_time': training_time,
                'tag_frequencies': dict(self.tag_frequencies.most_common(20))
            }
            
            self.is_trained = True
            
            logger.info(f"Tag predictor training completed: {accuracy:.3f} accuracy, {unique_tags} unique tags")
            
            return TrainingResult(
                success=True,
                samples_trained=len(filtered_texts),
                unique_tags=unique_tags,
                model_accuracy=accuracy,
                training_time=training_time,
                errors=[]
            )
            
        except Exception as e:
            error_msg = f"Training failed: {e}"
            logger.error(error_msg)
            return TrainingResult(
                success=False,
                samples_trained=0,
                unique_tags=0,
                model_accuracy=0.0,
                training_time=time.time() - start_time,
                errors=[error_msg]
            )
    
    def predict_tags(self, 
                    title: str, 
                    content: str = "", 
                    url: str = "",
                    existing_tags: List[str] = None) -> List[TagPrediction]:
        """Predict tags for given content"""
        
        if existing_tags is None:
            existing_tags = []
        
        try:
            if not self.is_trained:
                logger.warning("Model not trained, cannot predict tags")
                return []
            
            # Combine text for prediction
            combined_text = f"{title} {content}".strip()
            
            if not combined_text:
                return []
            
            predictions = []
            existing_tags_lower = {tag.lower() for tag in existing_tags}
            
            # ML model predictions
            ml_predictions = self._predict_with_model(combined_text)
            for tag, confidence in ml_predictions:
                if tag.lower() not in existing_tags_lower and confidence >= self.min_confidence:
                    predictions.append(TagPrediction(
                        tag=tag,
                        confidence=confidence,
                        source='ml_model'
                    ))
            
            # Pattern-based predictions
            pattern_predictions = self._predict_with_patterns(combined_text, url)
            for tag, confidence in pattern_predictions:
                if tag.lower() not in existing_tags_lower:
                    # Check if already predicted by ML model
                    if not any(p.tag.lower() == tag.lower() for p in predictions):
                        predictions.append(TagPrediction(
                            tag=tag,
                            confidence=confidence,
                            source='pattern'
                        ))
            
            # Frequency-based predictions
            frequency_predictions = self._predict_with_frequency(combined_text)
            for tag, confidence in frequency_predictions:
                if tag.lower() not in existing_tags_lower:
                    # Check if already predicted
                    if not any(p.tag.lower() == tag.lower() for p in predictions):
                        predictions.append(TagPrediction(
                            tag=tag,
                            confidence=confidence,
                            source='frequency'
                        ))
            
            # Sort by confidence and limit results
            predictions.sort(key=lambda x: x.confidence, reverse=True)
            
            logger.debug(f"Predicted {len(predictions)} tags for content")
            return predictions[:self.max_predictions]
            
        except Exception as e:
            logger.error(f"Tag prediction failed: {e}")
            return []
    
    def _extract_training_data(self, bookmarks: List[Dict[str, Any]]) -> Tuple[List[str], List[List[str]]]:
        """Extract training texts and tag sets from bookmarks"""
        
        texts = []
        tag_sets = []
        
        for bookmark in bookmarks:
            try:
                # Extract text content
                title = bookmark.get('name', '') or ''
                description = bookmark.get('description', '') or ''
                content = bookmark.get('content', {})
                
                if isinstance(content, dict):
                    text_content = content.get('text_content', '') or ''
                else:
                    text_content = str(content) if content else ''
                
                combined_text = f"{title} {description} {text_content}".strip()
                
                # Extract tags
                tags = bookmark.get('tags', [])
                tag_names = []
                
                for tag in tags:
                    if isinstance(tag, dict):
                        tag_name = tag.get('name', '')
                    else:
                        tag_name = str(tag)
                    
                    if tag_name and tag_name.strip():
                        tag_name = tag_name.strip()
                        tag_names.append(tag_name)
                        self.tag_frequencies[tag_name] += 1
                
                # Only include bookmarks with both text and tags
                if combined_text and tag_names:
                    texts.append(combined_text)
                    tag_sets.append(tag_names)
                    
                    # Store patterns for pattern-based prediction
                    for tag_name in tag_names:
                        self.tag_patterns[tag_name].append(combined_text)
                
            except Exception as e:
                logger.warning(f"Failed to extract training data from bookmark: {e}")
                continue
        
        logger.info(f"Extracted {len(texts)} training samples with {len(self.tag_frequencies)} unique tags")
        return texts, tag_sets
    
    def _filter_by_frequency(self, texts: List[str], tag_sets: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """Filter training data by tag frequency"""
        
        # Get frequent tags
        frequent_tags = {tag for tag, count in self.tag_frequencies.items() 
                        if count >= self.min_tag_frequency}
        
        logger.info(f"Filtering to {len(frequent_tags)} frequent tags (min frequency: {self.min_tag_frequency})")
        
        filtered_texts = []
        filtered_tag_sets = []
        
        for text, tags in zip(texts, tag_sets):
            # Keep only frequent tags
            filtered_tags = [tag for tag in tags if tag in frequent_tags]
            
            # Only include samples with at least one frequent tag
            if filtered_tags:
                filtered_texts.append(text)
                filtered_tag_sets.append(filtered_tags)
        
        return filtered_texts, filtered_tag_sets
    
    def _prepare_training_data(self, texts: List[str], tag_sets: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform texts
        X = self.vectorizer.fit_transform(texts)
        
        # Create label binarizer for multi-label classification
        self.label_binarizer = MultiLabelBinarizer()
        y = self.label_binarizer.fit_transform(tag_sets)
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} labels")
        
        return X, y
    
    def _predict_with_model(self, text: str) -> List[Tuple[str, float]]:
        """Predict tags using trained ML model"""
        
        if not self.vectorizer or not self.classifier or not self.label_binarizer:
            return []
        
        try:
            # Transform text
            X = self.vectorizer.transform([text])
            
            # Get prediction probabilities
            probabilities = self.classifier.predict_proba(X)[0]
            
            # Extract predictions with confidence scores
            predictions = []
            
            for i, class_probs in enumerate(probabilities):
                if len(class_probs) > 1:  # Binary classification for each tag
                    confidence = class_probs[1]  # Probability of positive class
                    if confidence >= self.min_confidence:
                        tag = self.label_binarizer.classes_[i]
                        predictions.append((tag, confidence))
            
            return predictions
            
        except Exception as e:
            logger.error(f"ML model prediction failed: {e}")
            return []
    
    def _predict_with_patterns(self, text: str, url: str = "") -> List[Tuple[str, float]]:
        """Predict tags using pattern matching"""
        
        predictions = []
        text_lower = text.lower()
        url_lower = url.lower()
        
        try:
            # Technology patterns
            tech_patterns = {
                'Python': ['python', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
                'JavaScript': ['javascript', 'js', 'node', 'react', 'vue', 'angular'],
                'AI': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'neural', 'deep learning'],
                'Docker': ['docker', 'container', 'containerization'],
                'Kubernetes': ['kubernetes', 'k8s', 'orchestration'],
                'GitHub': ['github.com', 'repository', 'repo', 'git'],
                'Tutorial': ['tutorial', 'how to', 'guide', 'learn', 'step by step'],
                'Documentation': ['docs', 'documentation', 'api', 'reference'],
                'Gaming': ['game', 'gaming', 'genshin', 'steam', 'twitch']
            }
            
            for tag, patterns in tech_patterns.items():
                matches = sum(1 for pattern in patterns if pattern in text_lower or pattern in url_lower)
                if matches > 0:
                    confidence = min(0.8, matches * 0.3)  # Scale confidence by matches
                    predictions.append((tag, confidence))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Pattern prediction failed: {e}")
            return []
    
    def _predict_with_frequency(self, text: str) -> List[Tuple[str, float]]:
        """Predict tags based on frequency and text similarity"""
        
        predictions = []
        
        try:
            # Get most frequent tags
            top_tags = self.tag_frequencies.most_common(50)
            
            for tag, frequency in top_tags:
                # Simple text matching
                if tag.lower() in text.lower():
                    # Base confidence on frequency and text match
                    base_confidence = min(0.7, frequency / max(self.tag_frequencies.values()))
                    confidence = base_confidence * 0.8  # Reduce confidence for frequency-based
                    
                    if confidence >= self.min_confidence:
                        predictions.append((tag, confidence))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Frequency prediction failed: {e}")
            return []
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate multi-label classification accuracy"""
        
        try:
            # Calculate Jaccard similarity (intersection over union)
            intersection = np.sum(y_true * y_pred, axis=1)
            union = np.sum((y_true + y_pred) > 0, axis=1)
            
            # Avoid division by zero
            jaccard_scores = np.where(union > 0, intersection / union, 0)
            
            return float(np.mean(jaccard_scores))
            
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return 0.0
    
    def update_with_feedback(self, 
                           text: str, 
                           predicted_tags: List[str], 
                           actual_tags: List[str]) -> None:
        """Update model with user feedback (incremental learning)"""
        
        try:
            # Store feedback for future retraining
            # For now, just update tag frequencies
            for tag in actual_tags:
                self.tag_frequencies[tag] += 1
                self.tag_patterns[tag].append(text)
            
            logger.debug(f"Updated model with feedback: {len(actual_tags)} tags")
            
        except Exception as e:
            logger.error(f"Failed to update with feedback: {e}")
    
    def save_model(self, file_path: Optional[str] = None) -> bool:
        """Save trained model to disk"""
        
        try:
            if file_path is None:
                file_path = self.models_dir / 'tag_predictor.pkl'
            else:
                file_path = Path(file_path)
            
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'vectorizer': self.vectorizer,
                'label_binarizer': self.label_binarizer,
                'classifier': self.classifier,
                'tag_frequencies': dict(self.tag_frequencies),
                'tag_patterns': dict(self.tag_patterns),
                'is_trained': self.is_trained,
                'training_stats': self.training_stats,
                'config': {
                    'min_confidence': self.min_confidence,
                    'max_predictions': self.max_predictions,
                    'min_tag_frequency': self.min_tag_frequency
                }
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Saved tag predictor model to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, file_path: Optional[str] = None) -> bool:
        """Load trained model from disk"""
        
        try:
            if file_path is None:
                file_path = self.models_dir / 'tag_predictor.pkl'
            else:
                file_path = Path(file_path)
            
            if not file_path.exists():
                logger.info("No saved model found")
                return False
            
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data.get('vectorizer')
            self.label_binarizer = model_data.get('label_binarizer')
            self.classifier = model_data.get('classifier')
            self.tag_frequencies = Counter(model_data.get('tag_frequencies', {}))
            self.tag_patterns = defaultdict(list, model_data.get('tag_patterns', {}))
            self.is_trained = model_data.get('is_trained', False)
            self.training_stats = model_data.get('training_stats', {})
            
            # Update config if saved
            saved_config = model_data.get('config', {})
            for key, value in saved_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info(f"Loaded tag predictor model from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get tag predictor statistics"""
        
        return {
            'sklearn_available': SKLEARN_AVAILABLE,
            'is_trained': self.is_trained,
            'unique_tags': len(self.tag_frequencies),
            'total_tag_occurrences': sum(self.tag_frequencies.values()),
            'most_common_tags': dict(self.tag_frequencies.most_common(10)),
            'training_stats': self.training_stats,
            'model_info': {
                'has_vectorizer': self.vectorizer is not None,
                'has_classifier': self.classifier is not None,
                'has_label_binarizer': self.label_binarizer is not None,
                'vectorizer_features': self.vectorizer.get_feature_names_out().shape[0] if self.vectorizer else 0,
                'classifier_classes': len(self.label_binarizer.classes_) if self.label_binarizer else 0
            },
            'settings': {
                'min_confidence': self.min_confidence,
                'max_predictions': self.max_predictions,
                'min_tag_frequency': self.min_tag_frequency,
                'max_features': self.max_features
            }
        }