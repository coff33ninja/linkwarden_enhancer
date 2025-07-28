"""Text processing utilities"""

import re
from typing import List
from collections import Counter

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class TextUtils:
    """Utilities for text processing and analysis"""
    
    # Common stop words
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs'
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove special characters but keep alphanumeric and basic punctuation
            text = re.sub(r'[^\w\s\-\.]', ' ', text)
            
            # Strip and return
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Failed to clean text: {e}")
            return text
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
        """Extract keywords from text"""
        if not text:
            return []
        
        try:
            # Clean and lowercase
            cleaned_text = TextUtils.clean_text(text).lower()
            
            # Split into words
            words = cleaned_text.split()
            
            # Filter words
            keywords = []
            for word in words:
                if (len(word) >= min_length and 
                    word not in TextUtils.STOP_WORDS and
                    not word.isdigit() and
                    re.match(r'^[a-zA-Z][a-zA-Z0-9\-]*$', word)):
                    keywords.append(word)
            
            # Count frequency and return most common
            word_counts = Counter(keywords)
            return [word for word, count in word_counts.most_common(max_keywords)]
            
        except Exception as e:
            logger.warning(f"Failed to extract keywords: {e}")
            return []
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        try:
            # Extract keywords from both texts
            keywords1 = set(TextUtils.extract_keywords(text1))
            keywords2 = set(TextUtils.extract_keywords(text2))
            
            if not keywords1 or not keywords2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(keywords1.intersection(keywords2))
            union = len(keywords1.union(keywords2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate text similarity: {e}")
            return 0.0
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 200, suffix: str = "...") -> str:
        """Truncate text to specified length"""
        if not text or len(text) <= max_length:
            return text
        
        try:
            # Find last space before max_length
            truncated = text[:max_length]
            last_space = truncated.rfind(' ')
            
            if last_space > max_length * 0.8:  # If space is reasonably close to end
                truncated = truncated[:last_space]
            
            return truncated + suffix
            
        except Exception as e:
            logger.warning(f"Failed to truncate text: {e}")
            return text[:max_length] + suffix
    
    @staticmethod
    def normalize_tag_name(tag_name: str) -> str:
        """Normalize tag name for consistency"""
        if not tag_name:
            return ""
        
        try:
            # Clean and strip
            normalized = tag_name.strip()
            
            # Apply title case for multi-word tags
            if " " in normalized:
                normalized = normalized.title()
            else:
                # Capitalize first letter for single words
                normalized = normalized.capitalize()
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Failed to normalize tag name '{tag_name}': {e}")
            return tag_name
    
    @staticmethod
    def extract_sentences(text: str, max_sentences: int = 3) -> List[str]:
        """Extract first few sentences from text"""
        if not text:
            return []
        
        try:
            # Simple sentence splitting on periods, exclamation marks, question marks
            sentences = re.split(r'[.!?]+', text)
            
            # Clean and filter sentences
            clean_sentences = []
            for sentence in sentences[:max_sentences]:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Minimum sentence length
                    clean_sentences.append(sentence)
            
            return clean_sentences
            
        except Exception as e:
            logger.warning(f"Failed to extract sentences: {e}")
            return [text[:200]]  # Fallback to truncated text
    
    @staticmethod
    def detect_language_hints(text: str) -> List[str]:
        """Detect programming language hints in text"""
        language_patterns = {
            'python': [r'\bpython\b', r'\.py\b', r'\bdjango\b', r'\bflask\b', r'\bpip\b'],
            'javascript': [r'\bjavascript\b', r'\bjs\b', r'\bnode\b', r'\bnpm\b', r'\breact\b', r'\bvue\b'],
            'java': [r'\bjava\b', r'\.jar\b', r'\bmaven\b', r'\bspring\b'],
            'csharp': [r'\bc#\b', r'\bcsharp\b', r'\.net\b', r'\bvisual studio\b'],
            'cpp': [r'\bc\+\+\b', r'\bcpp\b', r'\.cpp\b', r'\.h\b'],
            'go': [r'\bgolang\b', r'\bgo\b', r'\.go\b'],
            'rust': [r'\brust\b', r'\.rs\b', r'\bcargo\b'],
            'php': [r'\bphp\b', r'\.php\b', r'\blaravel\b', r'\bcomposer\b'],
            'ruby': [r'\bruby\b', r'\.rb\b', r'\brails\b', r'\bgem\b'],
        }
        
        detected_languages = []
        text_lower = text.lower()
        
        try:
            for language, patterns in language_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        detected_languages.append(language)
                        break
            
            return list(set(detected_languages))  # Remove duplicates
            
        except Exception as e:
            logger.warning(f"Failed to detect language hints: {e}")
            return []