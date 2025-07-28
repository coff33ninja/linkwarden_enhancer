"""Tests for duplicate detection and resolution system"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from enhancement.duplicate_detector import (
    DuplicateDetector, ResolutionStrategy, DuplicateResolution, 
    QualityScore, InteractiveChoice
)
from ai.similarity_engine import DuplicateGroup


class TestDuplicateDetector:
    """Test duplicate detection and resolution functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        config = {
            'duplicate_detection': {
                'exact_threshold': 0.95,
                'near_threshold': 0.85,
                'similar_threshold': 0.7,
                'default_strategy': 'quality_based',
                'auto_resolve_threshold': 0.9,
                'require_confirmation': True
            }
        }
        
        # Mock the similarity calculator to avoid model loading
        with patch('enhancement.duplicate_detector.SimilarityCalculator'):
            self.detector = DuplicateDetector(config)
    
    def test_initialization(self):
        """Test duplicate detector initialization"""
        assert self.detector.exact_duplicate_threshold == 0.95
        assert self.detector.near_duplicate_threshold == 0.85
        assert self.detector.similar_threshold == 0.7
        assert self.detector.default_strategy == ResolutionStrategy.QUALITY_BASED
        assert self.detector.auto_resolve_threshold == 0.9
        assert self.detector.require_user_confirmation == True
    
    def test_url_duplicate_detection(self):
        """Test URL-based duplicate detection"""
        bookmarks = [
            {
                'id': 1,
                'url': 'https://example.com/page',
                'name': 'Example Page',
                'description': 'A test page'
            },
            {
                'id': 2,
                'url': 'https://example.com/page/',  # Trailing slash
                'name': 'Example Page',
                'description': 'Same test page'
            },
            {
                'id': 3,
                'url': 'https://different.com/page',
                'name': 'Different Page',
                'description': 'A different page'
            }
        ]
        
        # Mock URL normalizer to return same normalized URL for first two
        self.detector.url_normalizer.normalize_url = Mock()
        self.detector.url_normalizer.normalize_url.side_effect = [
            Mock(normalized_url='https://example.com/page'),
            Mock(normalized_url='https://example.com/page'),
            Mock(normalized_url='https://different.com/page')
        ]
        
        # Mock similarity calculator
        self.detector.similarity_calculator.calculate_comprehensive_similarity = Mock()
        self.detector.similarity_calculator.calculate_comprehensive_similarity.return_value = Mock(
            overall_similarity=0.9
        )
        
        groups = self.detector.detect_url_duplicates(bookmarks)
        
        assert len(groups) == 1
        assert len(groups[0].bookmarks) == 2
        assert 1 in groups[0].bookmarks
        assert 2 in groups[0].bookmarks
        assert groups[0].group_type == "url_duplicate"
    
    def test_quality_score_calculation(self):
        """Test bookmark quality score calculation"""
        bookmarks = [
            {
                'id': 1,
                'name': 'High Quality Bookmark Title',
                'url': 'https://example.com/article',
                'description': 'This is a comprehensive description of the bookmark content with good length and detail.',
                'tags': ['programming', 'tutorial', 'python'],
                'collection': 'Development',
                'created_at': '2024-01-01T00:00:00Z',
                'content': {'text_content': 'Rich content here...'}
            },
            {
                'id': 2,
                'name': 'untitled',  # Generic title
                'url': 'http://example.com',  # HTTP, not HTTPS
                'description': '',  # No description
                'tags': [],  # No tags
            },
            {
                'id': 3,
                'name': '',  # No title
                'url': '',  # No URL
                'description': 'https://example.com',  # Description is just URL
            }
        ]
        
        quality_scores = self.detector.calculate_quality_scores(bookmarks)
        
        assert len(quality_scores) == 3
        
        # High quality bookmark should have highest score
        high_quality = next(q for q in quality_scores if q.bookmark_id == 1)
        low_quality = next(q for q in quality_scores if q.bookmark_id == 2)
        no_title = next(q for q in quality_scores if q.bookmark_id == 3)
        
        assert high_quality.overall_quality > low_quality.overall_quality
        assert high_quality.overall_quality > no_title.overall_quality
        
        # Check individual components
        assert high_quality.title_quality > 0.5
        assert high_quality.description_quality > 0.5
        assert high_quality.metadata_completeness > 0.5
        
        assert low_quality.title_quality < 0.5  # Generic title
        assert low_quality.description_quality == 0.0  # No description
    
    def test_merge_resolution_strategy(self):
        """Test merge resolution strategy"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Python Tutorial',
                'url': 'https://example.com/python',
                'description': 'Learn Python programming',
                'tags': ['python', 'programming'],
                'collection': 'Development'
            },
            {
                'id': 2,
                'name': 'Python Programming Guide',
                'url': 'https://example.com/python-guide',
                'description': 'Comprehensive Python programming guide with examples',
                'tags': ['python', 'tutorial', 'guide'],
                'collection': 'Tutorials'
            }
        ]
        
        group = DuplicateGroup(
            group_id=1,
            bookmarks=[1, 2],
            similarity_scores=[0.85, 0.85],
            group_type="near_duplicate",
            representative_bookmark=1
        )
        
        # Mock quality score calculation
        self.detector.calculate_quality_scores = Mock()
        self.detector.calculate_quality_scores.return_value = [
            QualityScore(1, 0.6, 0.5, 0.8, 0.7, 0.65),
            QualityScore(2, 0.8, 0.9, 0.8, 0.8, 0.825)  # Higher quality
        ]
        
        resolutions = self.detector.resolve_duplicates([group], bookmarks, ResolutionStrategy.MERGE)
        
        assert len(resolutions) == 1
        resolution = resolutions[0]
        
        assert resolution.strategy == ResolutionStrategy.MERGE
        assert resolution.action == "merge"
        assert resolution.primary_bookmark_id == 2  # Higher quality bookmark
        assert resolution.secondary_bookmark_ids == [1]
        assert resolution.merged_data is not None
        
        # Check merged data
        merged = resolution.merged_data
        assert 'python' in merged['tags']
        assert 'programming' in merged['tags']
        assert 'tutorial' in merged['tags']
        assert 'guide' in merged['tags']
        
        # Should use longer description
        assert merged['description'] == 'Comprehensive Python programming guide with examples'
    
    def test_quality_based_resolution_strategy(self):
        """Test quality-based resolution strategy"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Low Quality',
                'url': 'http://example.com',
                'description': '',
                'tags': []
            },
            {
                'id': 2,
                'name': 'High Quality Bookmark Title',
                'url': 'https://example.com/article',
                'description': 'Detailed description with good content',
                'tags': ['tag1', 'tag2'],
                'collection': 'Collection'
            }
        ]
        
        group = DuplicateGroup(
            group_id=1,
            bookmarks=[1, 2],
            similarity_scores=[0.8, 0.8],
            group_type="similar",
            representative_bookmark=1
        )
        
        # Mock quality scores
        self.detector.calculate_quality_scores = Mock()
        self.detector.calculate_quality_scores.return_value = [
            QualityScore(1, 0.2, 0.0, 0.3, 0.1, 0.15),  # Low quality
            QualityScore(2, 0.8, 0.7, 0.9, 0.8, 0.8)    # High quality
        ]
        
        resolutions = self.detector.resolve_duplicates([group], bookmarks, ResolutionStrategy.QUALITY_BASED)
        
        assert len(resolutions) == 1
        resolution = resolutions[0]
        
        assert resolution.strategy == ResolutionStrategy.QUALITY_BASED
        assert resolution.action == "keep"
        assert resolution.primary_bookmark_id == 2  # Higher quality
        assert resolution.secondary_bookmark_ids == [1]
    
    def test_recency_based_resolution_strategy(self):
        """Test recency-based resolution strategy"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Older Bookmark',
                'url': 'https://example.com',
                'created_at': '2024-01-01T00:00:00Z'
            },
            {
                'id': 2,
                'name': 'Newer Bookmark',
                'url': 'https://example.com',
                'created_at': '2024-02-01T00:00:00Z'
            }
        ]
        
        group = DuplicateGroup(
            group_id=1,
            bookmarks=[1, 2],
            similarity_scores=[0.9, 0.9],
            group_type="exact",
            representative_bookmark=1
        )
        
        resolutions = self.detector.resolve_duplicates([group], bookmarks, ResolutionStrategy.RECENCY_BASED)
        
        assert len(resolutions) == 1
        resolution = resolutions[0]
        
        assert resolution.strategy == ResolutionStrategy.RECENCY_BASED
        assert resolution.action == "keep"
        assert resolution.primary_bookmark_id == 2  # More recent
        assert resolution.secondary_bookmark_ids == [1]
    
    def test_interactive_choices_generation(self):
        """Test generation of interactive choices for ambiguous cases"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'url': 'https://example.com/1',
                'description': 'First bookmark'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'url': 'https://example.com/2',
                'description': 'Second bookmark'
            }
        ]
        
        # Low similarity group (ambiguous)
        group = DuplicateGroup(
            group_id=1,
            bookmarks=[1, 2],
            similarity_scores=[0.75, 0.75],  # Below auto_resolve_threshold (0.9)
            group_type="similar",
            representative_bookmark=1
        )
        
        choices = self.detector.get_interactive_choices([group], bookmarks)
        
        assert len(choices) == 1
        choice = choices[0]
        
        assert choice.group_id == 1
        assert len(choice.bookmarks) == 2
        assert choice.similarity_scores == [0.75, 0.75]
        assert choice.recommended_action in ["merge_all", "keep_best_quality", "keep_most_recent", "manual_review"]
        assert "merge_all" in choice.options
        assert "keep_best_quality" in choice.options
        assert "manual_review" in choice.options
    
    def test_title_quality_assessment(self):
        """Test title quality assessment"""
        test_cases = [
            # (bookmark, expected_quality_range)
            ({'name': 'High Quality Descriptive Title'}, (0.5, 1.0)),
            ({'name': 'untitled'}, (0.0, 0.3)),  # Generic title
            ({'name': ''}, (0.0, 0.0)),  # No title
            ({'name': 'A'}, (0.0, 0.3)),  # Too short
            ({'name': 'Good Title'}, (0.3, 1.0)),  # Decent title
        ]
        
        for bookmark, (min_quality, max_quality) in test_cases:
            quality = self.detector._assess_title_quality(bookmark)
            assert min_quality <= quality <= max_quality, f"Title quality for '{bookmark.get('name', '')}' should be between {min_quality} and {max_quality}, got {quality}"
    
    def test_description_quality_assessment(self):
        """Test description quality assessment"""
        test_cases = [
            ({'description': 'This is a comprehensive description with good length and meaningful content that describes the bookmark well.'}, (0.7, 1.0)),
            ({'description': 'Short desc'}, (0.2, 0.5)),
            ({'description': ''}, (0.0, 0.0)),
            ({'description': 'https://example.com'}, (0.0, 0.3)),  # Just URL
        ]
        
        for bookmark, (min_quality, max_quality) in test_cases:
            quality = self.detector._assess_description_quality(bookmark)
            assert min_quality <= quality <= max_quality, f"Description quality should be between {min_quality} and {max_quality}, got {quality}"
    
    def test_url_quality_assessment(self):
        """Test URL quality assessment"""
        test_cases = [
            ({'url': 'https://example.com/article/title'}, (0.8, 1.0)),  # Good HTTPS URL with path
            ({'url': 'http://example.com'}, (0.5, 0.8)),  # HTTP, no path
            ({'url': ''}, (0.0, 0.0)),  # No URL
            ({'url': 'not-a-url'}, (0.0, 0.3)),  # Invalid URL
        ]
        
        for bookmark, (min_quality, max_quality) in test_cases:
            quality = self.detector._assess_url_quality(bookmark)
            assert min_quality <= quality <= max_quality, f"URL quality should be between {min_quality} and {max_quality}, got {quality}"
    
    def test_generic_title_detection(self):
        """Test generic title detection"""
        generic_titles = [
            'untitled', 'Untitled', 'UNTITLED',
            'page', 'Page', 'New Page',
            'document', 'Document',
            'home', 'Home',
            'index', 'Index',
            'default', 'Default',
            'new tab', 'New Tab',
            'blank page', 'Blank Page',
            'loading', 'Loading...',
            'error', 'Error',
            '404', '404 Not Found'
        ]
        
        for title in generic_titles:
            assert self.detector._is_generic_title(title), f"'{title}' should be detected as generic"
        
        non_generic_titles = [
            'Python Programming Guide',
            'How to Learn JavaScript',
            'Machine Learning Tutorial',
            'Web Development Best Practices'
        ]
        
        for title in non_generic_titles:
            assert not self.detector._is_generic_title(title), f"'{title}' should not be detected as generic"
    
    def test_bookmark_data_merging(self):
        """Test bookmark data merging functionality"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Short Title',
                'url': 'https://example.com',
                'description': 'Short description',
                'tags': ['tag1', 'tag2'],
                'collection': 'Collection1'
            },
            {
                'id': 2,
                'name': 'Much Longer and More Descriptive Title',
                'url': 'https://example.com/article',
                'description': 'This is a much longer and more comprehensive description with lots of detail',
                'tags': ['tag2', 'tag3', 'tag4'],
                'collection': 'Collection2'
            }
        ]
        
        merged = self.detector._merge_bookmark_data(bookmarks, primary_id=1)
        
        # Should use longer title
        assert merged['name'] == 'Much Longer and More Descriptive Title'
        
        # Should use longer description
        assert merged['description'] == 'This is a much longer and more comprehensive description with lots of detail'
        
        # Should merge all tags
        merged_tags = set(merged['tags'])
        expected_tags = {'tag1', 'tag2', 'tag3', 'tag4'}
        assert merged_tags == expected_tags
        
        # Should use primary bookmark's collection
        assert merged['collection'] == 'Collection1'


if __name__ == "__main__":
    pytest.main([__file__])