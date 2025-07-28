"""Tests for enhanced similarity calculator"""

import pytest
from unittest.mock import Mock, patch
from enhancement.similarity_calculator import SimilarityCalculator, ComprehensiveSimilarityScore


class TestSimilarityCalculator:
    """Test enhanced similarity calculation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        config = {
            'similarity': {
                'url_threshold': 0.8,
                'title_threshold': 0.7,
                'content_threshold': 0.6,
                'semantic_threshold': 0.7,
                'overall_threshold': 0.75
            }
        }
        
        # Mock the SimilarityEngine to avoid model loading
        with patch('enhancement.similarity_calculator.SimilarityEngine'):
            self.calculator = SimilarityCalculator(config)
    
    def test_url_similarity_calculation(self):
        """Test URL similarity calculation"""
        # Mock the URL normalizer
        self.calculator.url_normalizer.get_url_similarity_score = Mock(return_value=0.9)
        
        similarity = self.calculator.calculate_url_similarity(
            "https://example.com/page",
            "https://example.com/page/"
        )
        
        assert similarity == 0.9
        self.calculator.url_normalizer.get_url_similarity_score.assert_called_once()
    
    def test_title_similarity_calculation(self):
        """Test title similarity calculation"""
        test_cases = [
            # Exact match
            ("Python Programming Guide", "Python Programming Guide", 1.0),
            
            # Similar titles
            ("Python Programming Guide", "Python Programming Tutorial", 0.5),
            
            # Different titles
            ("Python Programming", "JavaScript Tutorial", 0.0),
            
            # Empty titles
            ("", "Python Guide", 0.0),
            ("Python Guide", "", 0.0),
        ]
        
        for title1, title2, expected_min in test_cases:
            similarity = self.calculator.calculate_title_similarity(title1, title2)
            
            if expected_min == 1.0:
                assert similarity == expected_min, f"Expected exact match for '{title1}' vs '{title2}'"
            elif expected_min == 0.0:
                assert similarity <= 0.2, f"Expected low similarity for '{title1}' vs '{title2}'"
            else:
                assert similarity >= expected_min, f"Expected similarity >= {expected_min} for '{title1}' vs '{title2}'"
    
    def test_content_similarity_calculation(self):
        """Test content similarity calculation"""
        content1 = "This is a comprehensive guide to Python programming with examples and tutorials."
        content2 = "A complete Python programming guide with examples and step-by-step tutorials."
        content3 = "JavaScript is a popular programming language for web development."
        
        # Similar content
        similarity1 = self.calculator.calculate_content_similarity(content1, content2)
        assert similarity1 > 0.5, "Similar content should have high similarity"
        
        # Different content
        similarity2 = self.calculator.calculate_content_similarity(content1, content3)
        assert similarity2 < 0.5, "Different content should have low similarity"
        
        # Empty content
        similarity3 = self.calculator.calculate_content_similarity("", content1)
        assert similarity3 == 0.0, "Empty content should have zero similarity"
    
    def test_comprehensive_similarity_score(self):
        """Test comprehensive similarity score calculation"""
        bookmark1 = {
            'id': 1,
            'url': 'https://example.com/python-guide',
            'name': 'Python Programming Guide',
            'description': 'A comprehensive guide to Python programming',
            'content': {'text_content': 'Python is a powerful programming language...'}
        }
        
        bookmark2 = {
            'id': 2,
            'url': 'https://example.com/python-tutorial',
            'name': 'Python Programming Tutorial',
            'description': 'Learn Python programming with examples',
            'content': {'text_content': 'Python is a versatile programming language...'}
        }
        
        # Mock semantic similarity
        self.calculator.calculate_semantic_similarity = Mock(return_value=0.8)
        
        score = self.calculator.calculate_comprehensive_similarity(bookmark1, bookmark2)
        
        assert isinstance(score, ComprehensiveSimilarityScore)
        assert 0.0 <= score.url_similarity <= 1.0
        assert 0.0 <= score.title_similarity <= 1.0
        assert 0.0 <= score.content_similarity <= 1.0
        assert score.semantic_similarity == 0.8
        assert 0.0 <= score.overall_similarity <= 1.0
        assert 0.0 <= score.confidence <= 1.0
    
    def test_comprehensive_similarity_score_post_init(self):
        """Test ComprehensiveSimilarityScore post-initialization calculation"""
        score = ComprehensiveSimilarityScore(
            url_similarity=0.8,
            title_similarity=0.7,
            content_similarity=0.6,
            semantic_similarity=0.9,
            overall_similarity=0.0,  # Should be calculated
            confidence=0.0  # Should be calculated
        )
        
        # Check that overall similarity was calculated
        expected_overall = (0.8 * 0.3 + 0.7 * 0.25 + 0.6 * 0.2 + 0.9 * 0.25)
        assert abs(score.overall_similarity - expected_overall) < 0.01
        
        # Check that confidence was calculated (2 high scores out of 4: 0.8, 0.9 > 0.7)
        expected_confidence = min(1.0, (2 + 1) / 5.0)  # 2 high scores + 1, divided by 5
        assert abs(score.confidence - expected_confidence) < 0.1
    
    def test_clean_title(self):
        """Test title cleaning functionality"""
        test_cases = [
            # Remove site suffixes
            ("Python Guide - Example.com", "Python Guide"),
            ("Tutorial | Programming Site", "Tutorial"),
            ("Article – News Website", "Article"),
            
            # Remove special characters
            ("Python & JavaScript!", "Python JavaScript"),
            ("Guide: Programming 101", "Guide Programming 101"),
            
            # Normalize whitespace
            ("Python    Programming   Guide", "Python Programming Guide"),
        ]
        
        for original, expected in test_cases:
            cleaned = self.calculator._clean_title(original)
            assert cleaned == expected, f"Failed to clean '{original}' to '{expected}', got '{cleaned}'"
    
    def test_clean_content(self):
        """Test content cleaning functionality"""
        test_cases = [
            # Remove HTML tags
            ("<p>This is <strong>important</strong> content.</p>", "This is important content."),
            
            # Normalize whitespace
            ("This   is    content   with   spaces.", "This is content with spaces."),
            
            # Truncate long content
            ("a" * 1500, "a" * 1000),
        ]
        
        for original, expected in test_cases:
            cleaned = self.calculator._clean_content(original)
            assert cleaned == expected, f"Failed to clean content properly"
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        text = "Python programming is a powerful tool for data science and web development"
        keywords = self.calculator._extract_keywords(text)
        
        assert isinstance(keywords, set)
        assert "python" in keywords
        assert "programming" in keywords
        assert "data" in keywords
        assert "science" in keywords
        
        # Stop words should be removed
        assert "is" not in keywords
        assert "a" not in keywords
        assert "for" not in keywords
    
    def test_levenshtein_similarity(self):
        """Test Levenshtein similarity calculation"""
        test_cases = [
            ("hello", "hello", 1.0),
            ("hello", "hallo", 0.8),  # 1 character difference out of 5
            ("", "hello", 0.0),
            ("hello", "", 0.0),
            ("abc", "xyz", 0.0),  # Completely different
        ]
        
        for s1, s2, expected in test_cases:
            similarity = self.calculator._levenshtein_similarity(s1, s2)
            assert abs(similarity - expected) < 0.1, f"Levenshtein similarity for '{s1}' vs '{s2}' should be ~{expected}"
    
    def test_lcs_similarity(self):
        """Test Longest Common Subsequence similarity"""
        test_cases = [
            ("hello", "hello", 1.0),
            ("hello", "helo", 0.8),  # 4 common characters out of 5
            ("", "hello", 0.0),
            ("hello", "", 0.0),
            ("abc", "xyz", 0.0),  # No common subsequence
        ]
        
        for s1, s2, expected in test_cases:
            similarity = self.calculator._lcs_similarity(s1, s2)
            assert abs(similarity - expected) < 0.1, f"LCS similarity for '{s1}' vs '{s2}' should be ~{expected}"
    
    def test_ngram_similarity(self):
        """Test n-gram similarity calculation"""
        test_cases = [
            ("hello", "hello", 1.0),
            ("hello", "hallo", 0.2),  # Some common 3-grams
            ("", "hello", 0.0),
            ("hello", "", 0.0),
            ("abc", "xyz", 0.0),  # No common n-grams
        ]
        
        for s1, s2, expected in test_cases:
            similarity = self.calculator._ngram_similarity(s1, s2, n=3)
            assert abs(similarity - expected) < 0.2, f"N-gram similarity for '{s1}' vs '{s2}' should be ~{expected}"
    
    def test_determine_match_type(self):
        """Test match type determination"""
        # URL match
        score1 = ComprehensiveSimilarityScore(0.95, 0.5, 0.5, 0.5, 0.0, 0.0)
        assert self.calculator._determine_match_type(score1) == "url_match"
        
        # Title match
        score2 = ComprehensiveSimilarityScore(0.5, 0.95, 0.5, 0.5, 0.0, 0.0)
        assert self.calculator._determine_match_type(score2) == "title_match"
        
        # Semantic match
        score3 = ComprehensiveSimilarityScore(0.5, 0.5, 0.5, 0.85, 0.0, 0.0)
        assert self.calculator._determine_match_type(score3) == "semantic_match"
        
        # Content match
        score4 = ComprehensiveSimilarityScore(0.5, 0.5, 0.85, 0.5, 0.0, 0.0)
        assert self.calculator._determine_match_type(score4) == "content_match"
        
        # Mixed match
        score5 = ComprehensiveSimilarityScore(0.6, 0.6, 0.6, 0.6, 0.0, 0.0)
        assert self.calculator._determine_match_type(score5) == "mixed_match"
    
    def test_extract_bookmark_content(self):
        """Test bookmark content extraction"""
        bookmark = {
            'description': 'This is a description',
            'content': {
                'text_content': 'This is the main content'
            }
        }
        
        content = self.calculator._extract_bookmark_content(bookmark)
        assert 'This is a description' in content
        assert 'This is the main content' in content
        
        # Test with string content
        bookmark2 = {
            'description': 'Description',
            'content': 'String content'
        }
        
        content2 = self.calculator._extract_bookmark_content(bookmark2)
        assert 'Description' in content2
        assert 'String content' in content2


if __name__ == "__main__":
    pytest.main([__file__])