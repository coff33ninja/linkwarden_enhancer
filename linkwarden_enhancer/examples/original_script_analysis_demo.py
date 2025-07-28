#!/usr/bin/env python3
"""
Demo script showing the original script analyzer functionality.
This demonstrates the functionality implemented in task 8.1.
"""

import tempfile
import shutil
from pathlib import Path

from linkwarden_enhancer.reference.original_script_analyzer import OriginalScriptAnalyzer


def create_sample_original_script() -> str:
    """Create a sample original script for analysis"""
    
    sample_script = '''
"""
Original Bookmark Management Script
This is a sample script that demonstrates various bookmark management patterns
that would be found in an original implementation.
"""

import re
import json
import urllib.parse
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional, Set

class BookmarkNormalizer:
    """Handles normalization of bookmark data"""
    
    def __init__(self):
        self.tag_cache = {}
        self.domain_cache = {}
    
    def normalize_tag(self, tag: str) -> str:
        """Normalize tag by cleaning and standardizing format"""
        if not tag or not isinstance(tag, str):
            return ""
        
        # Check cache first
        if tag in self.tag_cache:
            return self.tag_cache[tag]
        
        # Clean whitespace and convert to lowercase
        normalized = tag.strip().lower()
        
        # Remove special characters except hyphens and underscores
        normalized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', normalized)
        
        # Replace multiple spaces with single hyphen
        normalized = re.sub(r'\s+', '-', normalized)
        
        # Remove leading/trailing hyphens
        normalized = normalized.strip('-')
        
        # Cache the result
        self.tag_cache[tag] = normalized
        return normalized
    
    def clean_title(self, title: str) -> str:
        """Clean and normalize bookmark title"""
        if not title:
            return "Untitled"
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', title.strip())
        
        # Remove common suffixes
        suffixes = [' - YouTube', ' | GitHub', ' - Stack Overflow']
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
        
        return cleaned
    
    def extract_domain(self, url: str) -> str:
        """Extract and normalize domain from URL"""
        if not url:
            return "unknown"
        
        if url in self.domain_cache:
            return self.domain_cache[url]
        
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove www prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            
            self.domain_cache[url] = domain
            return domain
        except:
            return "invalid"

class BookmarkOrganizer:
    """Handles organization and categorization of bookmarks"""
    
    def __init__(self):
        self.category_rules = self._load_category_rules()
        self.collection_hierarchy = {}
    
    def organize_by_domain(self, bookmarks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Organize bookmarks by their domain"""
        organized = defaultdict(list)
        normalizer = BookmarkNormalizer()
        
        for bookmark in bookmarks:
            url = bookmark.get('url', '')
            domain = normalizer.extract_domain(url)
            organized[domain].append(bookmark)
        
        # Sort each domain's bookmarks by title
        for domain in organized:
            organized[domain].sort(key=lambda x: x.get('title', '').lower())
        
        return dict(organized)
    
    def categorize_bookmark(self, bookmark: Dict[str, Any]) -> str:
        """Categorize a bookmark based on its properties"""
        url = bookmark.get('url', '').lower()
        title = bookmark.get('title', '').lower()
        tags = [tag.lower() for tag in bookmark.get('tags', [])]
        
        # Check domain-based rules
        for category, domains in self.category_rules.items():
            if any(domain in url for domain in domains):
                return category
        
        # Check title-based rules
        if any(word in title for word in ['tutorial', 'guide', 'how-to']):
            return 'tutorials'
        elif any(word in title for word in ['news', 'article', 'blog']):
            return 'articles'
        elif any(word in title for word in ['tool', 'utility', 'app']):
            return 'tools'
        
        # Check tag-based rules
        if any(tag in ['development', 'programming', 'code'] for tag in tags):
            return 'development'
        elif any(tag in ['design', 'ui', 'ux'] for tag in tags):
            return 'design'
        elif any(tag in ['research', 'paper', 'academic'] for tag in tags):
            return 'research'
        
        return 'general'
    
    def group_by_tags(self, bookmarks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group bookmarks by their tags"""
        grouped = defaultdict(list)
        normalizer = BookmarkNormalizer()
        
        for bookmark in bookmarks:
            tags = bookmark.get('tags', [])
            
            if not tags:
                grouped['untagged'].append(bookmark)
            else:
                for tag in tags:
                    normalized_tag = normalizer.normalize_tag(tag)
                    if normalized_tag:
                        grouped[normalized_tag].append(bookmark)
        
        return dict(grouped)
    
    def _load_category_rules(self) -> Dict[str, List[str]]:
        """Load categorization rules"""
        return {
            'development': ['github.com', 'stackoverflow.com', 'gitlab.com', 'bitbucket.org'],
            'social': ['twitter.com', 'facebook.com', 'linkedin.com', 'reddit.com'],
            'media': ['youtube.com', 'vimeo.com', 'twitch.tv', 'netflix.com'],
            'news': ['bbc.com', 'cnn.com', 'reuters.com', 'techcrunch.com'],
            'shopping': ['amazon.com', 'ebay.com', 'etsy.com', 'shopify.com']
        }

class BookmarkSuggestionEngine:
    """Provides intelligent suggestions for bookmark management"""
    
    def __init__(self):
        self.suggestion_cache = {}
        self.tag_frequency = Counter()
        self.domain_patterns = defaultdict(set)
    
    def suggest_tags(self, bookmark: Dict[str, Any], existing_bookmarks: List[Dict] = None) -> List[str]:
        """Suggest tags for a bookmark based on content and patterns"""
        url = bookmark.get('url', '').lower()
        title = bookmark.get('title', '').lower()
        content = bookmark.get('content', '').lower()
        
        suggestions = set()
        
        # Domain-based suggestions
        domain_suggestions = self._get_domain_suggestions(url)
        suggestions.update(domain_suggestions)
        
        # Content-based suggestions
        content_suggestions = self._analyze_content_for_tags(title, content)
        suggestions.update(content_suggestions)
        
        # Pattern-based suggestions from existing bookmarks
        if existing_bookmarks:
            pattern_suggestions = self._get_pattern_suggestions(bookmark, existing_bookmarks)
            suggestions.update(pattern_suggestions)
        
        # Filter and rank suggestions
        ranked_suggestions = self._rank_suggestions(list(suggestions), bookmark)
        
        return ranked_suggestions[:5]  # Return top 5 suggestions
    
    def suggest_collection(self, bookmark: Dict[str, Any]) -> str:
        """Suggest a collection for the bookmark"""
        organizer = BookmarkOrganizer()
        category = organizer.categorize_bookmark(bookmark)
        
        # Map categories to collection names
        collection_mapping = {
            'development': 'Development Resources',
            'design': 'Design & UI/UX',
            'research': 'Research & Papers',
            'tutorials': 'Learning & Tutorials',
            'tools': 'Tools & Utilities',
            'articles': 'Articles & Blogs',
            'social': 'Social Media',
            'media': 'Media & Entertainment',
            'news': 'News & Current Events',
            'shopping': 'Shopping & Commerce',
            'general': 'General Bookmarks'
        }
        
        return collection_mapping.get(category, 'General Bookmarks')
    
    def find_similar_bookmarks(self, bookmark: Dict[str, Any], 
                              existing_bookmarks: List[Dict]) -> List[Dict]:
        """Find bookmarks similar to the given bookmark"""
        if not existing_bookmarks:
            return []
        
        target_url = bookmark.get('url', '').lower()
        target_title = bookmark.get('title', '').lower()
        target_tags = set(tag.lower() for tag in bookmark.get('tags', []))
        
        similarities = []
        
        for existing in existing_bookmarks:
            similarity_score = self._calculate_similarity(
                bookmark, existing, target_url, target_title, target_tags
            )
            
            if similarity_score > 0.3:  # Threshold for similarity
                similarities.append((existing, similarity_score))
        
        # Sort by similarity score and return top matches
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [bookmark for bookmark, score in similarities[:3]]
    
    def _get_domain_suggestions(self, url: str) -> List[str]:
        """Get tag suggestions based on domain"""
        domain_tags = {
            'github.com': ['development', 'code', 'repository', 'open-source'],
            'stackoverflow.com': ['programming', 'qa', 'help', 'development'],
            'youtube.com': ['video', 'tutorial', 'entertainment'],
            'medium.com': ['article', 'blog', 'writing'],
            'reddit.com': ['discussion', 'community', 'social'],
            'twitter.com': ['social', 'news', 'updates'],
            'linkedin.com': ['professional', 'networking', 'career']
        }
        
        for domain, tags in domain_tags.items():
            if domain in url:
                return tags
        
        return []
    
    def _analyze_content_for_tags(self, title: str, content: str) -> List[str]:
        """Analyze content to suggest relevant tags"""
        text = f"{title} {content}".lower()
        suggestions = []
        
        # Technology keywords
        tech_keywords = {
            'python': 'python', 'javascript': 'javascript', 'java': 'java',
            'react': 'react', 'vue': 'vue', 'angular': 'angular',
            'docker': 'docker', 'kubernetes': 'kubernetes',
            'machine learning': 'ml', 'artificial intelligence': 'ai',
            'data science': 'data-science'
        }
        
        for keyword, tag in tech_keywords.items():
            if keyword in text:
                suggestions.append(tag)
        
        # Content type keywords
        if any(word in text for word in ['tutorial', 'guide', 'how-to']):
            suggestions.append('tutorial')
        
        if any(word in text for word in ['news', 'breaking', 'update']):
            suggestions.append('news')
        
        if any(word in text for word in ['review', 'comparison', 'vs']):
            suggestions.append('review')
        
        return suggestions
    
    def _get_pattern_suggestions(self, bookmark: Dict[str, Any], 
                               existing_bookmarks: List[Dict]) -> List[str]:
        """Get suggestions based on patterns in existing bookmarks"""
        normalizer = BookmarkNormalizer()
        target_domain = normalizer.extract_domain(bookmark.get('url', ''))
        
        # Find bookmarks from the same domain
        same_domain_bookmarks = [
            b for b in existing_bookmarks 
            if normalizer.extract_domain(b.get('url', '')) == target_domain
        ]
        
        # Collect tags from same-domain bookmarks
        common_tags = []
        for b in same_domain_bookmarks:
            common_tags.extend(b.get('tags', []))
        
        # Return most common tags
        tag_counts = Counter(common_tags)
        return [tag for tag, count in tag_counts.most_common(3)]
    
    def _rank_suggestions(self, suggestions: List[str], bookmark: Dict[str, Any]) -> List[str]:
        """Rank suggestions by relevance"""
        # Simple ranking based on frequency and relevance
        # In a real implementation, this would be more sophisticated
        
        # Prioritize domain-specific suggestions
        url = bookmark.get('url', '').lower()
        title = bookmark.get('title', '').lower()
        
        ranked = []
        
        # High priority suggestions (domain-specific)
        high_priority = []
        if 'github.com' in url:
            high_priority.extend(['development', 'code', 'repository'])
        elif 'youtube.com' in url:
            high_priority.extend(['video', 'tutorial'])
        
        # Add high priority suggestions first
        for suggestion in suggestions:
            if suggestion in high_priority:
                ranked.append(suggestion)
        
        # Add remaining suggestions
        for suggestion in suggestions:
            if suggestion not in ranked:
                ranked.append(suggestion)
        
        return ranked
    
    def _calculate_similarity(self, bookmark1: Dict, bookmark2: Dict,
                            url1: str, title1: str, tags1: set) -> float:
        """Calculate similarity between two bookmarks"""
        normalizer = BookmarkNormalizer()
        
        url2 = bookmark2.get('url', '').lower()
        title2 = bookmark2.get('title', '').lower()
        tags2 = set(tag.lower() for tag in bookmark2.get('tags', []))
        
        similarity = 0.0
        
        # Domain similarity
        domain1 = normalizer.extract_domain(url1)
        domain2 = normalizer.extract_domain(url2)
        if domain1 == domain2:
            similarity += 0.3
        
        # Tag similarity (Jaccard similarity)
        if tags1 or tags2:
            intersection = len(tags1.intersection(tags2))
            union = len(tags1.union(tags2))
            if union > 0:
                similarity += 0.4 * (intersection / union)
        
        # Title similarity (simple word overlap)
        words1 = set(title1.split())
        words2 = set(title2.split())
        if words1 or words2:
            word_intersection = len(words1.intersection(words2))
            word_union = len(words1.union(words2))
            if word_union > 0:
                similarity += 0.3 * (word_intersection / word_union)
        
        return similarity

def validate_bookmark_data(bookmark: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean bookmark data"""
    errors = []
    warnings = []
    
    # Required fields validation
    if not bookmark.get('url'):
        errors.append("URL is required")
    elif not (bookmark['url'].startswith('http://') or bookmark['url'].startswith('https://')):
        errors.append("URL must start with http:// or https://")
    
    if not bookmark.get('title'):
        warnings.append("Title is missing")
    
    # Data type validation
    if 'tags' in bookmark and not isinstance(bookmark['tags'], list):
        errors.append("Tags must be a list")
    
    # Clean data
    cleaned_bookmark = bookmark.copy()
    
    if 'title' in cleaned_bookmark and cleaned_bookmark['title']:
        cleaned_bookmark['title'] = cleaned_bookmark['title'].strip()
    
    if 'tags' in cleaned_bookmark and isinstance(cleaned_bookmark['tags'], list):
        normalizer = BookmarkNormalizer()
        cleaned_bookmark['tags'] = [
            normalizer.normalize_tag(tag) for tag in cleaned_bookmark['tags']
            if tag and isinstance(tag, str)
        ]
        # Remove empty tags
        cleaned_bookmark['tags'] = [tag for tag in cleaned_bookmark['tags'] if tag]
    
    return {
        'bookmark': cleaned_bookmark,
        'errors': errors,
        'warnings': warnings,
        'is_valid': len(errors) == 0
    }

# Global utility functions
def extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text content"""
    if not text:
        return []
    
    # Simple keyword extraction (in reality, would use NLP)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
    
    keywords = [word for word in words if word not in stop_words]
    
    # Count frequency and return most common
    keyword_counts = Counter(keywords)
    return [word for word, count in keyword_counts.most_common(max_keywords)]

def merge_duplicate_bookmarks(bookmarks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge duplicate bookmarks based on URL"""
    seen_urls = {}
    merged = []
    
    for bookmark in bookmarks:
        url = bookmark.get('url', '').lower().strip()
        
        if url in seen_urls:
            # Merge with existing bookmark
            existing = seen_urls[url]
            
            # Merge tags
            existing_tags = set(existing.get('tags', []))
            new_tags = set(bookmark.get('tags', []))
            merged_tags = list(existing_tags.union(new_tags))
            existing['tags'] = merged_tags
            
            # Keep the longer title
            if len(bookmark.get('title', '')) > len(existing.get('title', '')):
                existing['title'] = bookmark['title']
        
        else:
            seen_urls[url] = bookmark
            merged.append(bookmark)
    
    return merged
'''
    
    return sample_script


def demo_original_script_analysis():
    """Demonstrate the original script analyzer functionality"""
    
    # Create temporary directory for demo
    demo_dir = tempfile.mkdtemp()
    print(f"Demo directory: {demo_dir}")
    
    try:
        # Create sample original script
        sample_script_content = create_sample_original_script()
        script_path = Path(demo_dir) / "original_bookmark_script.py"
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(sample_script_content)
        
        print(f"Created sample script: {script_path}")
        print(f"Script size: {len(sample_script_content)} characters")
        
        print("\n=== Original Script Analysis Demo ===\n")
        
        # Initialize analyzer
        analyzer = OriginalScriptAnalyzer()
        
        # Analyze the script
        print("1. Analyzing Original Script")
        print("-" * 40)
        
        result = analyzer.analyze_script(str(script_path))
        
        print(f"Analysis completed successfully!")
        print(f"Script: {result.script_path}")
        print(f"Total lines: {result.total_lines}")
        print(f"Functions found: {result.total_functions}")
        print(f"Classes found: {result.total_classes}")
        
        # Show function analysis
        print("\n2. Function Analysis")
        print("-" * 40)
        
        print(f"Functions analyzed: {len(result.functions)}")
        for func in result.functions[:5]:  # Show first 5 functions
            print(f"  • {func.name}")
            print(f"    Purpose: {func.purpose}")
            print(f"    Algorithm Type: {func.algorithm_type}")
            print(f"    Complexity Score: {func.complexity_score}")
            print(f"    Parameters: {', '.join(func.parameters)}")
            if func.patterns_detected:
                print(f"    Patterns: {', '.join(func.patterns_detected)}")
            print()
        
        if len(result.functions) > 5:
            print(f"  ... and {len(result.functions) - 5} more functions")
        
        # Show class analysis
        print("\n3. Class Analysis")
        print("-" * 40)
        
        for cls in result.classes:
            print(f"  • {cls.name}")
            print(f"    Purpose: {cls.purpose}")
            print(f"    Methods: {len(cls.methods)}")
            print(f"    Attributes: {', '.join(cls.attributes)}")
            if cls.patterns_detected:
                print(f"    Patterns: {', '.join(cls.patterns_detected)}")
            print()
        
        # Show pattern analysis
        print("\n4. Pattern Analysis")
        print("-" * 40)
        
        print(f"Tag Normalization Patterns: {len(result.tag_normalization_patterns)}")
        for pattern in result.tag_normalization_patterns:
            print(f"  • {pattern.pattern_name}")
            print(f"    Description: {pattern.description}")
            print(f"    Complexity: {pattern.complexity}")
            print(f"    Reusability: {pattern.reusability}")
            print(f"    Location: {pattern.code_location}")
            print()
        
        print(f"Collection Organization Patterns: {len(result.collection_organization_patterns)}")
        for pattern in result.collection_organization_patterns:
            print(f"  • {pattern.pattern_name}")
            print(f"    Description: {pattern.description}")
            print(f"    Complexity: {pattern.complexity}")
            print(f"    Reusability: {pattern.reusability}")
            print()
        
        print(f"Suggestion Mechanism Patterns: {len(result.suggestion_mechanism_patterns)}")
        for pattern in result.suggestion_mechanism_patterns:
            print(f"  • {pattern.pattern_name}")
            print(f"    Description: {pattern.description}")
            print(f"    Complexity: {pattern.complexity}")
            print(f"    Reusability: {pattern.reusability}")
            print()
        
        # Show algorithm insights
        print("\n5. Algorithm Insights")
        print("-" * 40)
        
        insights = result.algorithm_insights
        print("Algorithm Type Distribution:")
        for algo_type, count in insights['algorithm_types'].items():
            print(f"  {algo_type}: {count}")
        
        print("\nComplexity Distribution:")
        for complexity, count in insights['complexity_distribution'].items():
            print(f"  {complexity}: {count}")
        
        print(f"\nCommon Patterns: {', '.join(insights['common_patterns'])}")
        
        if insights['design_principles']:
            print(f"Design Principles: {', '.join(insights['design_principles'])}")
        
        # Show modernization recommendations
        print("\n6. Modernization Recommendations")
        print("-" * 40)
        
        for i, recommendation in enumerate(result.modernization_recommendations[:5], 1):
            print(f"{i}. {recommendation}")
        
        if len(result.modernization_recommendations) > 5:
            print(f"   ... and {len(result.modernization_recommendations) - 5} more recommendations")
        
        # Show reusable components
        print("\n7. Reusable Components")
        print("-" * 40)
        
        for component in result.reusable_components:
            print(f"  • {component}")
        
        # Show improvement opportunities
        print("\n8. Improvement Opportunities")
        print("-" * 40)
        
        for opportunity in result.improvement_opportunities[:3]:
            print(f"  • {opportunity}")
        
        # Export reports
        print("\n9. Exporting Analysis Reports")
        print("-" * 40)
        
        # Export JSON report
        json_report_path = Path(demo_dir) / "analysis_report.json"
        analyzer.export_analysis_report(result, str(json_report_path), "json")
        print(f"JSON report exported: {json_report_path}")
        
        # Export Markdown report
        md_report_path = Path(demo_dir) / "analysis_report.md"
        analyzer.export_analysis_report(result, str(md_report_path), "markdown")
        print(f"Markdown report exported: {md_report_path}")
        
        # Show sample of reference documentation
        print("\n10. Reference Documentation Sample")
        print("-" * 40)
        
        doc_lines = result.reference_documentation.split('\n')
        for line in doc_lines[:10]:
            print(line)
        
        if len(doc_lines) > 10:
            print(f"... ({len(doc_lines) - 10} more lines)")
        
        print(f"\nFull documentation available in: {md_report_path}")
        
        print("\n=== Analysis Demo Completed Successfully ===")
        print(f"\nGenerated files:")
        print(f"  - Original script: {script_path}")
        print(f"  - JSON report: {json_report_path}")
        print(f"  - Markdown report: {md_report_path}")
        
    finally:
        # Clean up (comment out to keep files for inspection)
        # shutil.rmtree(demo_dir, ignore_errors=True)
        print(f"\nDemo files preserved in: {demo_dir}")


if __name__ == '__main__':
    demo_original_script_analysis()