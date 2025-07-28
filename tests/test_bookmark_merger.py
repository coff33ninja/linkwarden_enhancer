"""Tests for bookmark merging functionality"""

import pytest
from enhancement.bookmark_merger import BookmarkMerger, MergeResult, MergeConflict
from enhancement.duplicate_detector import QualityScore


class TestBookmarkMerger:
    """Test bookmark merging functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        config = {
            'merge': {
                'prefer_longer_text': True,
                'preserve_user_data': True,
                'tags_strategy': 'union',
                'title_strategy': 'quality_based',
                'description_strategy': 'best_quality'
            }
        }
        self.merger = BookmarkMerger(config)
    
    def test_single_bookmark_merge(self):
        """Test merging a single bookmark (should return as-is)"""
        bookmark = {
            'id': 1,
            'name': 'Test Bookmark',
            'url': 'https://example.com',
            'description': 'Test description',
            'tags': ['test']
        }
        
        result = self.merger.merge_bookmarks([bookmark])
        
        assert result.merged_bookmark == bookmark
        assert result.source_bookmarks == [1]
        assert result.merge_strategy == "single_bookmark"
        assert result.conflicts_resolved == []
        assert result.quality_improvement == 0.0
    
    def test_basic_bookmark_merge(self):
        """Test basic merging of two bookmarks"""
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
                'description': 'This is a much longer and more comprehensive description',
                'tags': ['tag2', 'tag3'],
                'collection': 'Collection2'
            }
        ]
        
        result = self.merger.merge_bookmarks(bookmarks)
        
        assert result.merge_strategy == "intelligent_merge"
        assert len(result.source_bookmarks) == 2
        
        merged = result.merged_bookmark
        
        # Should use longer, more descriptive title
        assert merged['name'] == 'Much Longer and More Descriptive Title'
        
        # Should use longer description
        assert merged['description'] == 'This is a much longer and more comprehensive description'
        
        # Should merge tags (union strategy)
        merged_tags = set(merged['tags'])
        expected_tags = {'tag1', 'tag2', 'tag3'}
        assert merged_tags == expected_tags
    
    def test_title_merging_strategies(self):
        """Test different title merging strategies"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Short',
                'url': 'https://example.com'
            },
            {
                'id': 2,
                'name': 'Much Longer Title Here',
                'url': 'https://example.com'
            },
            {
                'id': 3,
                'name': 'High Quality Descriptive Title',
                'url': 'https://example.com'
            }
        ]
        
        # Test longest strategy
        merger_longest = BookmarkMerger({'merge': {'title_strategy': 'longest'}})
        result_longest = merger_longest.merge_bookmarks(bookmarks)
        assert result_longest.merged_bookmark['name'] == 'High Quality Descriptive Title'
        
        # Test most descriptive strategy
        merger_descriptive = BookmarkMerger({'merge': {'title_strategy': 'most_descriptive'}})
        result_descriptive = merger_descriptive.merge_bookmarks(bookmarks)
        # Should prefer the descriptive title
        assert 'Descriptive' in result_descriptive.merged_bookmark['name']
    
    def test_description_merging_strategies(self):
        """Test different description merging strategies"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'description': 'Short desc',
                'url': 'https://example.com'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'description': 'This is a much longer description with more detail and information',
                'url': 'https://example.com'
            },
            {
                'id': 3,
                'name': 'Bookmark 3',
                'description': 'https://example.com',  # URL-only (auto-generated)
                'url': 'https://example.com'
            }
        ]
        
        # Test longest strategy
        merger_longest = BookmarkMerger({'merge': {'description_strategy': 'longest'}})
        result_longest = merger_longest.merge_bookmarks(bookmarks)
        assert result_longest.merged_bookmark['description'] == 'This is a much longer description with more detail and information'
        
        # Test user_preferred strategy (should avoid URL-only descriptions)
        merger_user = BookmarkMerger({'merge': {'description_strategy': 'user_preferred'}})
        result_user = merger_user.merge_bookmarks(bookmarks)
        assert result_user.merged_bookmark['description'] != 'https://example.com'
    
    def test_tag_merging_strategies(self):
        """Test different tag merging strategies"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'tags': ['python', 'programming', 'tutorial'],
                'url': 'https://example.com'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'tags': ['python', 'guide', 'learning'],
                'url': 'https://example.com'
            },
            {
                'id': 3,
                'name': 'Bookmark 3',
                'tags': ['javascript', 'web'],
                'url': 'https://example.com'
            }
        ]
        
        # Test union strategy (default)
        merger_union = BookmarkMerger({'merge': {'tags_strategy': 'union'}})
        result_union = merger_union.merge_bookmarks(bookmarks)
        union_tags = set(result_union.merged_bookmark['tags'])
        expected_union = {'python', 'programming', 'tutorial', 'guide', 'learning', 'javascript', 'web'}
        assert union_tags == expected_union
        
        # Test intersection strategy
        merger_intersection = BookmarkMerger({'merge': {'tags_strategy': 'intersection'}})
        result_intersection = merger_intersection.merge_bookmarks(bookmarks)
        intersection_tags = set(result_intersection.merged_bookmark['tags'])
        # No tags are common to all three bookmarks
        assert len(intersection_tags) == 0
    
    def test_url_merging_https_preference(self):
        """Test URL merging with HTTPS preference"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'url': 'http://example.com',
                'description': 'HTTP version'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'url': 'https://example.com',
                'description': 'HTTPS version'
            }
        ]
        
        result = self.merger.merge_bookmarks(bookmarks)
        
        # Should prefer HTTPS URL
        assert result.merged_bookmark['url'] == 'https://example.com'
        assert 'url' in result.conflicts_resolved
    
    def test_collection_merging(self):
        """Test collection merging (most common)"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'collection': 'Development',
                'url': 'https://example.com'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'collection': 'Development',
                'url': 'https://example.com'
            },
            {
                'id': 3,
                'name': 'Bookmark 3',
                'collection': 'Tutorials',
                'url': 'https://example.com'
            }
        ]
        
        result = self.merger.merge_bookmarks(bookmarks)
        
        # Should use most common collection
        assert result.merged_bookmark['collection'] == 'Development'
    
    def test_timestamp_merging(self):
        """Test timestamp merging (earliest created, latest updated)"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'created_at': '2024-01-01T00:00:00Z',
                'updated_at': '2024-01-05T00:00:00Z',
                'url': 'https://example.com'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'created_at': '2024-01-03T00:00:00Z',
                'updated_at': '2024-01-10T00:00:00Z',
                'url': 'https://example.com'
            }
        ]
        
        result = self.merger.merge_bookmarks(bookmarks)
        merged = result.merged_bookmark
        
        # Should use earliest created time
        assert merged['created_at'] == '2024-01-01T00:00:00Z'
        
        # Should use latest updated time
        assert merged['updated_at'] == '2024-01-10T00:00:00Z'
    
    def test_content_merging(self):
        """Test content field merging"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'content': {
                    'text_content': 'Short content',
                    'metadata': {'author': 'Author 1'}
                },
                'url': 'https://example.com'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'content': {
                    'text_content': 'This is much longer content with more detail and information',
                    'summary': 'Content summary'
                },
                'url': 'https://example.com'
            }
        ]
        
        result = self.merger.merge_bookmarks(bookmarks)
        merged_content = result.merged_bookmark['content']
        
        # Should use longer text content
        assert merged_content['text_content'] == 'This is much longer content with more detail and information'
        
        # Should preserve other fields
        assert 'metadata' in merged_content
        assert 'summary' in merged_content
    
    def test_primary_bookmark_selection(self):
        """Test primary bookmark selection with quality scores"""
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
                'name': 'High Quality Bookmark',
                'url': 'https://example.com/article',
                'description': 'Detailed description',
                'tags': ['tag1', 'tag2']
            }
        ]
        
        quality_scores = [
            QualityScore(1, 0.2, 0.0, 0.3, 0.1, 0.15),  # Low quality
            QualityScore(2, 0.8, 0.7, 0.9, 0.8, 0.8)    # High quality
        ]
        
        result = self.merger.merge_bookmarks(bookmarks, quality_scores=quality_scores)
        
        # Should use high quality bookmark as base
        assert result.metadata_preserved['primary_id'] == 2
    
    def test_quality_improvement_calculation(self):
        """Test quality improvement calculation"""
        original = {
            'id': 1,
            'name': 'Short',
            'description': '',
            'tags': [],
            'content': {}
        }
        
        merged = {
            'id': 1,
            'name': 'Much Longer Descriptive Title',
            'description': 'Comprehensive description with details',
            'tags': ['tag1', 'tag2', 'tag3'],
            'content': {'text_content': 'Rich content'}
        }
        
        improvement = self.merger._calculate_quality_improvement(original, merged, None)
        
        # Should show positive improvement
        assert improvement > 0.0
    
    def test_generic_title_detection(self):
        """Test detection of generic/auto-generated descriptions"""
        test_cases = [
            ('https://example.com', True),  # URL only
            ('Short', True),  # Too short
            ('page 1', True),  # Generic pattern
            ('This is a good description with meaningful content', False),  # Good description
            ('Loading...', True),  # Generic pattern
        ]
        
        for description, expected in test_cases:
            result = self.merger._looks_auto_generated(description)
            assert result == expected, f"Description '{description}' should be {'auto-generated' if expected else 'user-written'}"
    
    def test_empty_bookmarks_handling(self):
        """Test handling of empty bookmark list"""
        with pytest.raises(ValueError, match="No bookmarks provided"):
            self.merger.merge_bookmarks([])
    
    def test_string_tags_handling(self):
        """Test handling of string tags (should convert to list)"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'tags': 'single-tag',  # String instead of list
                'url': 'https://example.com'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'tags': ['tag1', 'tag2'],  # Normal list
                'url': 'https://example.com'
            }
        ]
        
        result = self.merger.merge_bookmarks(bookmarks)
        merged_tags = set(result.merged_bookmark['tags'])
        
        # Should handle string tag properly
        assert 'single-tag' in merged_tags
        assert 'tag1' in merged_tags
        assert 'tag2' in merged_tags
    
    def test_merge_result_metadata(self):
        """Test merge result metadata preservation"""
        bookmarks = [
            {
                'id': 1,
                'name': 'Bookmark 1',
                'url': 'https://example.com'
            },
            {
                'id': 2,
                'name': 'Bookmark 2',
                'url': 'https://example.com'
            }
        ]
        
        result = self.merger.merge_bookmarks(bookmarks)
        metadata = result.metadata_preserved
        
        assert 'original_ids' in metadata
        assert metadata['original_ids'] == [1, 2]
        assert 'merge_timestamp' in metadata
        assert 'merge_strategy' in metadata
        assert metadata['merge_strategy'] == 'intelligent_merge'


if __name__ == "__main__":
    pytest.main([__file__])