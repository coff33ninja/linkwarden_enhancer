"""Content analysis system using scikit-learn and NLP"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from ..utils.logging_utils import get_logger
from ..utils.text_utils import TextUtils

logger = get_logger(__name__)


@dataclass
class Topic:
    """Represents a discovered topic"""
    name: str
    keywords: List[str]
    weight: float
    coherence_score: Optional[float] = None


@dataclass
class SentimentScore:
    """Sentiment analysis result"""
    compound: float
    positive: float
    negative: float
    neutral: float
    label: str  # 'positive', 'negative', 'neutral'


@dataclass
class ContentAnalysisResult:
    """Result of content analysis"""
    topics: List[Topic]
    sentiment: SentimentScore
    keywords: List[str]
    content_type: str
    language_hints: List[str]
    readability_score: float
    word_count: int
    unique_words: int


class ContentAnalyzer:
    """Advanced content analysis using machine learning and NLP"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize content analyzer"""
        self.config = config
        self.ai_config = config.get('ai', {})
        
        # Check dependencies
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for content analysis. Install with: pip install scikit-learn")
        
        if not NLTK_AVAILABLE:
            logger.warning("NLTK not available. Sentiment analysis will be limited.")
        
        # Initialize components
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.sentiment_analyzer = None
        self.is_trained = False
        
        # Initialize NLTK components
        self._initialize_nltk()
        
        # Analysis cache
        self.analysis_cache = {}
        
        logger.info("Content analyzer initialized")
    
    def _initialize_nltk(self) -> None:
        """Initialize NLTK components"""
        if not NLTK_AVAILABLE:
            return
        
        try:
            # Download required NLTK data
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("NLTK sentiment analyzer initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize NLTK: {e}")
    
    def train_models(self, documents: List[str]) -> bool:
        """Train analysis models on document corpus"""
        
        if not documents:
            logger.warning("No documents provided for training")
            return False
        
        try:
            logger.info(f"Training content analysis models on {len(documents)} documents")
            
            # Clean and preprocess documents
            cleaned_docs = []
            for doc in documents:
                if doc and len(doc.strip()) > 10:  # Minimum content length
                    cleaned = TextUtils.clean_text(doc)
                    if cleaned:
                        cleaned_docs.append(cleaned)
            
            if len(cleaned_docs) < 5:
                logger.warning("Insufficient documents for training (need at least 5)")
                return False
            
            # Initialize and train TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(cleaned_docs)
            logger.info(f"TF-IDF vectorizer trained with {tfidf_matrix.shape[1]} features")
            
            # Train LDA topic model
            n_topics = min(10, max(3, len(cleaned_docs) // 10))  # Adaptive topic count
            self.lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10,
                learning_method='batch'
            )
            
            self.lda_model.fit(tfidf_matrix)
            logger.info(f"LDA model trained with {n_topics} topics")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to train content analysis models: {e}")
            return False
    
    def analyze_content(self, 
                       title: str, 
                       content: str, 
                       url: str = "") -> ContentAnalysisResult:
        """Perform comprehensive content analysis"""
        
        try:
            # Combine text for analysis
            full_text = f"{title} {content}".strip()
            
            if not full_text:
                return self._create_empty_result()
            
            # Check cache
            cache_key = hash(full_text)
            if cache_key in self.analysis_cache:
                return self.analysis_cache[cache_key]
            
            # Extract basic metrics
            word_count = len(full_text.split())
            unique_words = len(set(full_text.lower().split()))
            
            # Extract keywords
            keywords = TextUtils.extract_keywords(full_text, max_keywords=15)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(full_text)
            
            # Extract topics (if models are trained)
            topics = self._extract_topics(full_text) if self.is_trained else []
            
            # Detect content type
            content_type = self._classify_content_type(title, content, url)
            
            # Detect language hints
            language_hints = TextUtils.detect_language_hints(full_text)
            
            # Calculate readability score (simple implementation)
            readability_score = self._calculate_readability(full_text)
            
            # Create result
            result = ContentAnalysisResult(
                topics=topics,
                sentiment=sentiment,
                keywords=keywords,
                content_type=content_type,
                language_hints=language_hints,
                readability_score=readability_score,
                word_count=word_count,
                unique_words=unique_words
            )
            
            # Cache result
            self.analysis_cache[cache_key] = result
            
            logger.debug(f"Content analysis completed for {len(full_text)} characters")
            return result
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return self._create_empty_result()
    
    def _analyze_sentiment(self, text: str) -> SentimentScore:
        """Analyze sentiment of text"""
        
        if not self.sentiment_analyzer:
            # Fallback simple sentiment analysis
            positive_words = ['good', 'great', 'excellent', 'amazing', 'awesome', 'love', 'best']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sucks']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return SentimentScore(0.5, 0.7, 0.2, 0.1, 'positive')
            elif neg_count > pos_count:
                return SentimentScore(-0.5, 0.2, 0.7, 0.1, 'negative')
            else:
                return SentimentScore(0.0, 0.3, 0.3, 0.4, 'neutral')
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            
            # Determine label
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return SentimentScore(
                compound=compound,
                positive=scores['pos'],
                negative=scores['neg'],
                neutral=scores['neu'],
                label=label
            )
            
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return SentimentScore(0.0, 0.33, 0.33, 0.34, 'neutral')
    
    def _extract_topics(self, text: str) -> List[Topic]:
        """Extract topics using LDA"""
        
        if not self.is_trained or not self.tfidf_vectorizer or not self.lda_model:
            return []
        
        try:
            # Transform text to TF-IDF
            tfidf_vector = self.tfidf_vectorizer.transform([text])
            
            # Get topic distribution
            topic_distribution = self.lda_model.transform(tfidf_vector)[0]
            
            # Get feature names
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            topics = []
            
            # Extract top topics
            for topic_idx, weight in enumerate(topic_distribution):
                if weight > 0.1:  # Only include topics with significant weight
                    
                    # Get top words for this topic
                    topic_words = self.lda_model.components_[topic_idx]
                    top_word_indices = topic_words.argsort()[-10:][::-1]
                    top_words = [feature_names[i] for i in top_word_indices]
                    
                    # Create topic name from top words
                    topic_name = f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"
                    
                    topic = Topic(
                        name=topic_name,
                        keywords=top_words,
                        weight=float(weight)
                    )
                    topics.append(topic)
            
            # Sort by weight
            topics.sort(key=lambda t: t.weight, reverse=True)
            return topics[:5]  # Return top 5 topics
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return []
    
    def _classify_content_type(self, title: str, content: str, url: str) -> str:
        """Classify the type of content"""
        
        text = f"{title} {content} {url}".lower()
        
        # Content type patterns
        patterns = {
            'tutorial': ['tutorial', 'how to', 'guide', 'step by step', 'learn', 'course'],
            'documentation': ['docs', 'documentation', 'api', 'reference', 'manual'],
            'news': ['news', 'article', 'report', 'breaking', 'update', 'announcement'],
            'tool': ['tool', 'app', 'software', 'utility', 'platform', 'service'],
            'blog': ['blog', 'post', 'thoughts', 'opinion', 'personal'],
            'research': ['paper', 'study', 'research', 'analysis', 'findings'],
            'video': ['video', 'youtube', 'watch', 'stream', 'episode'],
            'code': ['github', 'repository', 'code', 'source', 'library', 'framework'],
            'forum': ['forum', 'discussion', 'community', 'reddit', 'stack overflow'],
            'shopping': ['buy', 'shop', 'store', 'price', 'product', 'purchase']
        }
        
        # Score each content type
        scores = {}
        for content_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[content_type] = score
        
        # Return highest scoring type or 'general'
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        else:
            return 'general'
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score (0-1, higher is more readable)"""
        
        try:
            sentences = text.split('.')
            words = text.split()
            
            if not sentences or not words:
                return 0.5
            
            # Average sentence length
            avg_sentence_length = len(words) / len(sentences)
            
            # Average word length
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simple readability formula (inverse of complexity)
            # Lower sentence length and word length = higher readability
            readability = 1.0 - min(1.0, (avg_sentence_length / 20 + avg_word_length / 10) / 2)
            
            return max(0.0, min(1.0, readability))
            
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            return 0.5
    
    def _create_empty_result(self) -> ContentAnalysisResult:
        """Create empty analysis result"""
        return ContentAnalysisResult(
            topics=[],
            sentiment=SentimentScore(0.0, 0.33, 0.33, 0.34, 'neutral'),
            keywords=[],
            content_type='general',
            language_hints=[],
            readability_score=0.5,
            word_count=0,
            unique_words=0
        )
    
    def get_similar_content(self, 
                           target_text: str, 
                           candidate_texts: List[str], 
                           threshold: float = 0.3) -> List[Tuple[int, float]]:
        """Find similar content using TF-IDF cosine similarity"""
        
        if not self.is_trained or not candidate_texts:
            return []
        
        try:
            # Add target text to candidates for vectorization
            all_texts = [target_text] + candidate_texts
            
            # Vectorize all texts
            tfidf_matrix = self.tfidf_vectorizer.transform(all_texts)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Filter by threshold and return with indices
            similar_items = []
            for idx, similarity in enumerate(similarities):
                if similarity >= threshold:
                    similar_items.append((idx, float(similarity)))
            
            # Sort by similarity (highest first)
            similar_items.sort(key=lambda x: x[1], reverse=True)
            
            return similar_items
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return []
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get content analyzer statistics"""
        
        return {
            'is_trained': self.is_trained,
            'sklearn_available': SKLEARN_AVAILABLE,
            'nltk_available': NLTK_AVAILABLE,
            'has_sentiment_analyzer': self.sentiment_analyzer is not None,
            'has_tfidf_vectorizer': self.tfidf_vectorizer is not None,
            'has_lda_model': self.lda_model is not None,
            'cache_size': len(self.analysis_cache),
            'vectorizer_features': self.tfidf_vectorizer.get_feature_names_out().shape[0] if self.tfidf_vectorizer else 0,
            'lda_topics': self.lda_model.n_components if self.lda_model else 0
        }