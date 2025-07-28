"""
Tests for the original script analyzer
"""

import unittest
import tempfile
import shutil
from pathlib import Path

from linkwarden_enhancer.reference.original_script_analyzer import (
    OriginalScriptAnalyzer, ScriptAnalysisResult
)


class TestOriginalScriptAnalyzer(unittest.TestCase):
    """Test the original script analyzer functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.analyzer = OriginalScriptAnalyzer()
        
        # Create a sample script for testing
        self.sample_script_content = '''
"""
Sample bookmark management script for testing analysis
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional

class BookmarkManager:
    """Manages bookmark operations and organization"""
    
    def __init__(self):
        self.bookmarks = []
        self.collections = defaultdict(list)
        self.tags = set()
    
    def normalize_tag(self, tag: str) -> str:
        """Normalize tag by cleaning and standardizing format"""
        if not tag:
            return ""
        
        # Clean whitespace and convert to lowercase
        normalized = tag.strip().lower()
        
        # Remove special characters
        normalized = re.sub(r'[^a-zA-Z0-9\s-]', '', normalized)
        
        # Replace spaces with hyphens
        normalized = re.sub(r'\s+', '-', normalized)
        
        return normalized
    
    def organize_bookmarks_by_domain(self, bookmarks: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize bookmarks by their domain"""
        organized = defaultdict(list)
        
        for bookmark in bookmarks:
            url = bookmark.get('url', '')
            if url:
                # Extract domain
                domain = self._extract_domain(url)
                organized[domain].append(bookmark)
        
        return dict(organized)
    
    def suggest_tags_for_bookmark(self, bookmark: Dict[str, Any]) -> List[str]:
        """Suggest tags for a bookmark based on content analysis"""
        suggestions = []
        
        title = bookmark.get('title', '').lower()
        url = bookmark.get('url', '').lower()
        content = bookmark.get('content', '').lower()
        
        # Domain-based suggestions
        if 'github.com' in url:
            suggestions.extend(['development', 'code', 'repository'])
        elif 'stackoverflow.com' in url:
            suggestions.extend(['programming', 'help', 'qa'])
        elif 'youtube.com' in url:
            suggestions.extend(['video', 'tutorial', 'entertainment'])
        
        # Content-based suggestions
        if any(word in title for word in ['tutorial', 'guide', 'how-to']):
            suggestions.append('tutorial')
        
        if any(word in title for word in ['python', 'javascript', 'java']):
            suggestions.append('programming')
        
        return list(set(suggestions))
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        import urllib.parse
        
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc.lower()
        except:
            return 'unknown'
    
    def validate_bookmark(self, bookmark: Dict[str, Any]) -> bool:
        """Validate bookmark data structure"""
        required_fields = ['url', 'title']
        
        for field in required_fields:
            if field not in bookmark or not bookmark[field]:
                return False
        
        # Validate URL format
        url = bookmark['url']
        if not (url.startswith('http://') or url.startswith('https://')):
            return False
        
        return True
    
    def group_bookmarks_by_tags(self, bookmarks: List[Dict]) -> Dict[str, List[Dict]]:
        """Group bookmarks by their tags"""
        grouped = defaultdict(list)
        
        for bookmark in bookmarks:
            tags = bookmark.get('tags', [])
            if not tags:
                grouped['untagged'].append(bookmark)
            else:
                for tag in tags:
                    normalized_tag = self.normalize_tag(tag)
                    grouped[normalized_tag].append(bookmark)
        
        return dict(grouped)

def process_bookmark_file(file_path: str) -> Dict[str, Any]:
    """Process a bookmark file and return statistics"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        bookmarks = data.get('bookmarks', [])
        
        stats = {
            'total_bookmarks': len(bookmarks),
            'domains': set(),
            'tags': set()
        }
        
        for bookmark in bookmarks:
            # Extract domain
            url = bookmark.get('url', '')
            if url:
                domain = extract_domain_simple(url)
                stats['domains'].add(domain)
            
            # Extract tags
            tags = bookmark.get('tags', [])
            for tag in tags:
                stats['tags'].add(tag.lower())
        
        stats['domains'] = list(stats['domains'])
        stats['tags'] = list(stats['tags'])
        
        return stats
    
    except Exception as e:
        return {'error': str(e)}

def extract_domain_simple(url: str) -> str:
    """Simple domain extraction function"""
    if '://' in url:
        url = url.split('://', 1)[1]
    
    if '/' in url:
        url = url.split('/', 1)[0]
    
    return url.lower()

def clean_tag_list(tags: List[str]) -> List[str]:
    """Clean and normalize a list of tags"""
    cleaned = []
    
    for tag in tags:
        if tag and isinstance(tag, str):
            # Basic cleaning
            clean_tag = tag.strip().lower()
            clean_tag = re.sub(r'[^a-zA-Z0-9\s-]', '', clean_tag)
            
            if clean_tag and clean_tag not in cleaned:
                cleaned.append(clean_tag)
    
    return sorted(cleaned)

# Global configuration
DEFAULT_TAGS = ['general', 'bookmark', 'web']
MAX_SUGGESTIONS = 5
'''
        
        self.sample_script_path = Path(self.test_dir) / "sample_script.py"
        with open(self.sample_script_path, 'w', encoding='utf-8') as f:
            f.write(self.sample_script_content)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_analyze_script_basic(self):
        """Test basic script analysis functionality"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Verify basic structure
        self.assertIsInstance(result, ScriptAnalysisResult)
        self.assertEqual(result.script_path, str(self.sample_script_path))
        self.assertGreater(result.total_lines, 0)
        self.assertGreater(result.total_functions, 0)
        self.assertGreater(result.total_classes, 0)
    
    def test_function_extraction(self):
        """Test function extraction and analysis"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Should find multiple functions
        self.assertGreater(len(result.functions), 5)
        
        # Check for specific functions
        function_names = [f.name for f in result.functions]
        self.assertIn('normalize_tag', function_names)
        self.assertIn('suggest_tags_for_bookmark', function_names)
        self.assertIn('validate_bookmark', function_names)
        
        # Check function analysis details
        normalize_func = next(f for f in result.functions if f.name == 'normalize_tag')
        self.assertEqual(normalize_func.parameters, ['self', 'tag'])
        self.assertGreater(normalize_func.complexity_score, 1)
        self.assertIn('tag_normalization', normalize_func.patterns_detected)
    
    def test_class_extraction(self):
        """Test class extraction and analysis"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Should find the BookmarkManager class
        self.assertEqual(len(result.classes), 1)
        
        bookmark_manager = result.classes[0]
        self.assertEqual(bookmark_manager.name, 'BookmarkManager')
        self.assertGreater(len(bookmark_manager.methods), 5)
        self.assertIn('bookmarks', bookmark_manager.attributes)
        self.assertIn('collections', bookmark_manager.attributes)
    
    def test_pattern_detection(self):
        """Test pattern detection in the script"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Should detect tag normalization patterns
        self.assertGreater(len(result.tag_normalization_patterns), 0)
        
        # Should detect collection organization patterns
        self.assertGreater(len(result.collection_organization_patterns), 0)
        
        # Should detect suggestion mechanism patterns
        self.assertGreater(len(result.suggestion_mechanism_patterns), 0)
        
        # Check pattern details
        tag_patterns = result.tag_normalization_patterns
        self.assertTrue(any('normalize_tag' in pattern.pattern_name for pattern in tag_patterns))
        
        suggestion_patterns = result.suggestion_mechanism_patterns
        self.assertTrue(any('suggest_tags' in pattern.pattern_name for pattern in suggestion_patterns))
    
    def test_algorithm_insights(self):
        """Test algorithm insights extraction"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        insights = result.algorithm_insights
        
        # Should have algorithm type distribution
        self.assertIn('algorithm_types', insights)
        self.assertIn('complexity_distribution', insights)
        self.assertIn('common_patterns', insights)
        
        # Should detect some patterns
        self.assertGreater(len(insights['common_patterns']), 0)
    
    def test_modernization_recommendations(self):
        """Test modernization recommendations generation"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Should generate recommendations
        self.assertGreater(len(result.modernization_recommendations), 0)
        
        # Should include common modernization suggestions
        recommendations_text = ' '.join(result.modernization_recommendations).lower()
        self.assertTrue(any(keyword in recommendations_text 
                          for keyword in ['type hints', 'async', 'logging', 'test']))
    
    def test_reusable_components_identification(self):
        """Test identification of reusable components"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Should identify some reusable components
        self.assertGreater(len(result.reusable_components), 0)
        
        # Should identify utility functions
        components_text = ' '.join(result.reusable_components).lower()
        self.assertTrue(any(keyword in components_text 
                          for keyword in ['utility', 'validation', 'processing']))
    
    def test_reference_documentation_generation(self):
        """Test reference documentation generation"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Should generate reference documentation
        self.assertIsNotNone(result.reference_documentation)
        self.assertGreater(len(result.reference_documentation), 100)
        
        # Should include function and class information
        doc_text = result.reference_documentation.lower()
        self.assertIn('bookmarkmanager', doc_text)
        self.assertIn('normalize_tag', doc_text)
    
    def test_api_documentation_generation(self):
        """Test API documentation generation"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Should generate API documentation
        self.assertGreater(len(result.api_documentation), 0)
        
        # Should document functions and classes
        self.assertTrue(any(key.startswith('function_') for key in result.api_documentation.keys()))
        self.assertTrue(any(key.startswith('class_') for key in result.api_documentation.keys()))
    
    def test_export_json_report(self):
        """Test exporting analysis report as JSON"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        output_path = Path(self.test_dir) / "analysis_report.json"
        self.analyzer.export_analysis_report(result, str(output_path), "json")
        
        # Verify file was created
        self.assertTrue(output_path.exists())
        
        # Verify JSON content
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        self.assertIn('script_path', report_data)
        self.assertIn('functions', report_data)
        self.assertIn('classes', report_data)
        self.assertIn('tag_normalization_patterns', report_data)
    
    def test_export_markdown_report(self):
        """Test exporting analysis report as Markdown"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        output_path = Path(self.test_dir) / "analysis_report.md"
        self.analyzer.export_analysis_report(result, str(output_path), "markdown")
        
        # Verify file was created
        self.assertTrue(output_path.exists())
        
        # Verify Markdown content
        with open(output_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        self.assertIn('# Original Script Analysis Report', markdown_content)
        self.assertIn('## Identified Patterns', markdown_content)
        self.assertIn('## Modernization Recommendations', markdown_content)
    
    def test_nonexistent_script(self):
        """Test handling of nonexistent script file"""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze_script("nonexistent_script.py")
    
    def test_invalid_python_syntax(self):
        """Test handling of invalid Python syntax"""
        invalid_script_path = Path(self.test_dir) / "invalid_script.py"
        with open(invalid_script_path, 'w') as f:
            f.write("def invalid_function(\n    # Missing closing parenthesis")
        
        with self.assertRaises(SyntaxError):
            self.analyzer.analyze_script(str(invalid_script_path))
    
    def test_complexity_calculation(self):
        """Test complexity calculation for functions"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Find a function with known complexity
        suggest_func = next(f for f in result.functions if f.name == 'suggest_tags_for_bookmark')
        
        # Should have reasonable complexity score (has multiple if statements)
        self.assertGreater(suggest_func.complexity_score, 3)
        self.assertLess(suggest_func.complexity_score, 20)
    
    def test_pattern_complexity_assessment(self):
        """Test pattern complexity assessment"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Check that patterns have complexity assessments
        for pattern in result.tag_normalization_patterns:
            self.assertIn(pattern.complexity, ['simple', 'moderate', 'complex'])
        
        for pattern in result.suggestion_mechanism_patterns:
            self.assertIn(pattern.complexity, ['simple', 'moderate', 'complex'])
    
    def test_reusability_assessment(self):
        """Test reusability assessment of patterns"""
        result = self.analyzer.analyze_script(str(self.sample_script_path))
        
        # Check that patterns have reusability assessments
        for pattern in result.tag_normalization_patterns:
            self.assertIn(pattern.reusability, ['high', 'medium', 'low'])
        
        for pattern in result.collection_organization_patterns:
            self.assertIn(pattern.reusability, ['high', 'medium', 'low'])


if __name__ == '__main__':
    unittest.main()