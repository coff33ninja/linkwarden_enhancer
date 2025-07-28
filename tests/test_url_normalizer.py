"""Tests for URL normalization engine"""

import pytest
import asyncio
from enhancement.url_normalizer import URLNormalizer, NormalizedURL


class TestURLNormalizer:
    """Test URL normalization functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.normalizer = URLNormalizer()
    
    def test_basic_normalization(self):
        """Test basic URL normalization"""
        test_cases = [
            # Remove trailing slash
            ("https://example.com/", "https://example.com"),
            ("https://example.com/path/", "https://example.com/path"),
            
            # Normalize scheme
            ("HTTP://EXAMPLE.COM", "http://example.com"),
            
            # Normalize domain case
            ("https://GITHUB.COM/user/repo", "https://github.com/user/repo"),
            
            # Remove www for known domains
            ("https://www.github.com/user/repo", "https://github.com/user/repo"),
            ("https://www.stackoverflow.com/questions/123", "https://stackoverflow.com/questions/123"),
        ]
        
        for original, expected in test_cases:
            result = self.normalizer.normalize_url(original)
            assert result.normalized_url == expected, f"Failed for {original}"
    
    def test_tracking_parameter_removal(self):
        """Test removal of tracking parameters"""
        test_cases = [
            # UTM parameters
            ("https://example.com?utm_source=google&utm_medium=cpc", "https://example.com"),
            ("https://example.com/path?param=value&utm_campaign=test", "https://example.com/path?param=value"),
            
            # Facebook click ID
            ("https://example.com?fbclid=abc123", "https://example.com"),
            
            # Google click ID
            ("https://example.com?gclid=xyz789&param=keep", "https://example.com?param=keep"),
            
            # Multiple tracking params
            ("https://example.com?utm_source=fb&fbclid=123&ref=twitter&keep=this", 
             "https://example.com?keep=this"),
        ]
        
        for original, expected in test_cases:
            result = self.normalizer.normalize_url(original)
            assert result.normalized_url == expected, f"Failed for {original}"
            assert len(result.parameters_removed) > 0, f"Should have removed params for {original}"
    
    def test_github_pattern_normalization(self):
        """Test GitHub-specific URL patterns"""
        test_cases = [
            # Main branch normalization
            ("https://github.com/user/repo/tree/main", "https://github.com/user/repo"),
            ("https://github.com/user/repo/tree/master", "https://github.com/user/repo"),
            
            # Repository root
            ("https://github.com/user/repo/", "https://github.com/user/repo"),
        ]
        
        for original, expected in test_cases:
            result = self.normalizer.normalize_url(original)
            assert result.normalized_url == expected, f"Failed for {original}"
    
    def test_youtube_pattern_normalization(self):
        """Test YouTube-specific URL patterns"""
        test_cases = [
            # Short URL to full URL
            ("https://youtu.be/dQw4w9WgXcQ", "https://youtube.com/watch?v=dQw4w9WgXcQ"),
        ]
        
        for original, expected in test_cases:
            result = self.normalizer.normalize_url(original)
            assert result.normalized_url == expected, f"Failed for {original}"
        
        # Test parameter removal (order may vary due to sorting)
        result = self.normalizer.normalize_url("https://youtube.com/watch?v=dQw4w9WgXcQ&t=30s&utm_source=share")
        assert "utm_source" not in result.normalized_url
        assert "v=dQw4w9WgXcQ" in result.normalized_url
        assert "t=30s" in result.normalized_url
    
    def test_domain_aliases(self):
        """Test domain alias recognition"""
        result = self.normalizer.normalize_url("https://www.github.com/user/repo")
        
        expected_aliases = {'www.github.com', 'github.com'}
        assert result.domain_aliases == expected_aliases
    
    def test_url_equivalence(self):
        """Test URL equivalence checking"""
        equivalent_pairs = [
            # Same URL
            ("https://example.com", "https://example.com"),
            
            # Trailing slash
            ("https://example.com/", "https://example.com"),
            
            # www vs non-www for GitHub
            ("https://www.github.com/user/repo", "https://github.com/user/repo"),
            
            # Different tracking parameters
            ("https://example.com?utm_source=google", "https://example.com?fbclid=123"),
            
            # GitHub branch variations
            ("https://github.com/user/repo", "https://github.com/user/repo/tree/main"),
        ]
        
        for url1, url2 in equivalent_pairs:
            assert self.normalizer.are_urls_equivalent(url1, url2), f"Should be equivalent: {url1} vs {url2}"
    
    def test_url_similarity_score(self):
        """Test URL similarity scoring"""
        test_cases = [
            # Identical URLs
            ("https://example.com", "https://example.com", 1.0),
            
            # Same domain, different paths
            ("https://example.com/path1", "https://example.com/path2", 0.5),
            
            # Different domains
            ("https://example.com", "https://different.com", 0.0),
            
            # Similar paths
            ("https://example.com/a/b/c", "https://example.com/a/b/d", 0.7),
        ]
        
        for url1, url2, expected_min in test_cases:
            score = self.normalizer.get_url_similarity_score(url1, url2)
            if expected_min == 1.0:
                assert score == expected_min, f"Expected exact match for {url1} vs {url2}"
            elif expected_min == 0.0:
                assert score <= 0.2, f"Expected low similarity for {url1} vs {url2}"
            else:
                assert score >= expected_min, f"Expected similarity >= {expected_min} for {url1} vs {url2}"
    
    def test_batch_normalization(self):
        """Test batch URL normalization"""
        urls = [
            "https://example.com/",
            "https://www.github.com/user/repo",
            "https://youtube.com/watch?v=abc&utm_source=test"
        ]
        
        results = asyncio.run(self.normalizer.batch_normalize_urls(urls))
        
        assert len(results) == len(urls)
        assert all(isinstance(result, NormalizedURL) for result in results)
        
        # Check specific normalizations
        assert results[0].normalized_url == "https://example.com"
        assert results[1].normalized_url == "https://github.com/user/repo"
        assert "utm_source" not in results[2].normalized_url
    
    def test_malformed_urls(self):
        """Test handling of malformed URLs"""
        malformed_urls = [
            "",
            "not-a-url",
            "ftp://example.com",  # Different scheme
            "https://",  # Incomplete
        ]
        
        for url in malformed_urls:
            result = self.normalizer.normalize_url(url)
            # Should not crash and return original URL if normalization fails
            assert result.original_url == url
    
    def test_parameter_preservation(self):
        """Test that important parameters are preserved"""
        test_cases = [
            # Keep search parameters
            ("https://example.com/search?q=test&utm_source=google", "q=test"),
            
            # Keep YouTube timestamp  
            ("https://youtube.com/watch?v=abc&t=30s&utm_campaign=share", ["v=abc", "t=30s"]),
            
            # Keep pagination
            ("https://example.com/page?page=2&utm_medium=email", "page=2"),
        ]
        
        for original, expected_params in test_cases:
            result = self.normalizer.normalize_url(original)
            
            # Check that tracking params are removed
            assert "utm_source" not in result.normalized_url
            assert "utm_campaign" not in result.normalized_url  
            assert "utm_medium" not in result.normalized_url
            
            # Check that important params are preserved
            if isinstance(expected_params, list):
                for param in expected_params:
                    assert param in result.normalized_url, f"Missing {param} in {result.normalized_url}"
            else:
                assert expected_params in result.normalized_url, f"Missing {expected_params} in {result.normalized_url}"


if __name__ == "__main__":
    pytest.main([__file__])