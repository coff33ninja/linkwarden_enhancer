#!/usr/bin/env python3
"""
Test script for Tag Quality Control and Merging functionality
Tests the TagQualityController implementation in auto_tagger.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhancement.auto_tagger import TagQualityController, AutoTaggingConfig, TagSuggestion


def test_confidence_threshold_filtering():
    """Test that tags below confidence threshold are filtered out"""
    print("Testing confidence threshold filtering...")
    
    config = AutoTaggingConfig()
    config.min_confidence_threshold = 0.5
    controller = TagQualityController(config)
    
    suggestions = [
        TagSuggestion(tag="python", confidence=0.8, source="ml_model"),
        TagSuggestion(tag="lowconf", confidence=0.3, source="url_analysis"),  # Should be filtered
        TagSuggestion(tag="javascript", confidence=0.6, source="content_analysis"),
        TagSuggestion(tag="verylowconf", confidence=0.1, source="dictionary"),  # Should be filtered
    ]
    
    result = controller.process_tag_suggestions(suggestions, [])
    
    assert "python" in result, "High confidence tag should be included"
    assert "javascript" in result, "Medium confidence tag should be included"
    assert "lowconf" not in result, "Low confidence tag should be filtered"
    assert "verylowconf" not in result, "Very low confidence tag should be filtered"
    
    print("✅ Confidence threshold filtering works correctly")


def test_tag_deduplication():
    """Test that duplicate tags are removed, keeping highest confidence"""
    print("Testing tag deduplication...")
    
    config = AutoTaggingConfig()
    controller = TagQualityController(config)
    
    suggestions = [
        TagSuggestion(tag="python", confidence=0.7, source="ml_model"),
        TagSuggestion(tag="Python", confidence=0.8, source="url_analysis"),  # Same tag, higher confidence
        TagSuggestion(tag="PYTHON", confidence=0.6, source="content_analysis"),  # Same tag, lower confidence
        TagSuggestion(tag="javascript", confidence=0.9, source="specialized_analyzer"),
    ]
    
    result = controller.process_tag_suggestions(suggestions, [])
    
    # Should only have one "python" tag with highest confidence
    python_count = sum(1 for tag in result if tag.lower() == "python")
    assert python_count == 1, f"Should have exactly 1 python tag, got {python_count}"
    assert "javascript" in result, "Other tags should be preserved"
    
    print("✅ Tag deduplication works correctly")


def test_similarity_merging():
    """Test that similar tags are merged to canonical forms"""
    print("Testing similarity merging...")
    
    config = AutoTaggingConfig()
    controller = TagQualityController(config)
    
    suggestions = [
        TagSuggestion(tag="js", confidence=0.7, source="url_analysis"),
        TagSuggestion(tag="javascript", confidence=0.8, source="ml_model"),
        TagSuggestion(tag="py", confidence=0.6, source="content_analysis"),
        TagSuggestion(tag="python", confidence=0.9, source="specialized_analyzer"),
        TagSuggestion(tag="reactjs", confidence=0.7, source="url_analysis"),
        TagSuggestion(tag="react.js", confidence=0.5, source="dictionary"),
        TagSuggestion(tag="nodejs", confidence=0.8, source="ml_model"),
    ]
    
    result = controller.process_tag_suggestions(suggestions, [])
    
    # Check that similar tags are merged to canonical forms
    assert "javascript" in result, "js should be merged to javascript"
    assert "js" not in result, "js should be merged away"
    
    assert "python" in result, "py should be merged to python"
    assert "py" not in result, "py should be merged away"
    
    assert "react" in result, "reactjs/react.js should be merged to react"
    assert "reactjs" not in result, "reactjs should be merged away"
    assert "react.js" not in result, "react.js should be merged away"
    
    assert "node" in result, "nodejs should be merged to node"
    assert "nodejs" not in result, "nodejs should be merged away"
    
    print("✅ Similarity merging works correctly")


def test_generic_tag_filtering():
    """Test that generic/overly broad tags are filtered out"""
    print("Testing generic tag filtering...")
    
    config = AutoTaggingConfig()
    config.enable_generic_filtering = True
    controller = TagQualityController(config)
    
    suggestions = [
        TagSuggestion(tag="python", confidence=0.8, source="ml_model"),
        TagSuggestion(tag="web", confidence=0.7, source="url_analysis"),  # Generic - should be filtered
        TagSuggestion(tag="website", confidence=0.6, source="content_analysis"),  # Generic - should be filtered
        TagSuggestion(tag="stuff", confidence=0.9, source="dictionary"),  # Generic - should be filtered
        TagSuggestion(tag="programming", confidence=0.8, source="specialized_analyzer"),
        TagSuggestion(tag="general", confidence=0.7, source="url_analysis"),  # Generic - should be filtered
    ]
    
    result = controller.process_tag_suggestions(suggestions, [])
    
    assert "python" in result, "Specific tags should be preserved"
    assert "programming" in result, "Specific tags should be preserved"
    
    assert "web" not in result, "Generic tag 'web' should be filtered"
    assert "website" not in result, "Generic tag 'website' should be filtered"
    assert "stuff" not in result, "Generic tag 'stuff' should be filtered"
    assert "general" not in result, "Generic tag 'general' should be filtered"
    
    print("✅ Generic tag filtering works correctly")


def test_tag_count_limiting():
    """Test that tag count is limited to configured maximum"""
    print("Testing tag count limiting...")
    
    config = AutoTaggingConfig()
    config.max_new_tags = 3  # Limit to 3 new tags
    controller = TagQualityController(config)
    
    # Create many high-confidence suggestions
    suggestions = [
        TagSuggestion(tag=f"tag{i}", confidence=0.9 - i*0.05, source="ml_model")
        for i in range(10)  # 10 tags, but should be limited to 3
    ]
    
    result = controller.process_tag_suggestions(suggestions, [])
    
    assert len(result) <= 3, f"Should have at most 3 tags, got {len(result)}"
    
    # Should keep the highest confidence tags
    assert "tag0" in result, "Highest confidence tag should be kept"
    assert "tag1" in result, "Second highest confidence tag should be kept"
    assert "tag2" in result, "Third highest confidence tag should be kept"
    
    print("✅ Tag count limiting works correctly")


def test_existing_tags_exclusion():
    """Test that existing tags are not duplicated"""
    print("Testing existing tags exclusion...")
    
    config = AutoTaggingConfig()
    controller = TagQualityController(config)
    
    existing_tags = ["python", "javascript", "programming"]
    
    suggestions = [
        TagSuggestion(tag="python", confidence=0.9, source="ml_model"),  # Already exists
        TagSuggestion(tag="JavaScript", confidence=0.8, source="url_analysis"),  # Already exists (case insensitive)
        TagSuggestion(tag="react", confidence=0.7, source="content_analysis"),  # New tag
        TagSuggestion(tag="Programming", confidence=0.6, source="dictionary"),  # Already exists (case insensitive)
    ]
    
    result = controller.process_tag_suggestions(suggestions, existing_tags)
    
    assert "python" not in result, "Existing tag should not be duplicated"
    assert "javascript" not in result, "Existing tag should not be duplicated (case insensitive)"
    assert "programming" not in result, "Existing tag should not be duplicated (case insensitive)"
    assert "react" in result, "New tag should be included"
    
    print("✅ Existing tags exclusion works correctly")


def test_comprehensive_pipeline():
    """Test the complete quality control pipeline"""
    print("Testing comprehensive quality control pipeline...")
    
    config = AutoTaggingConfig()
    config.min_confidence_threshold = 0.4
    config.max_new_tags = 5
    config.enable_generic_filtering = True
    config.enable_tag_deduplication = True
    config.enable_similarity_merging = True
    
    controller = TagQualityController(config)
    
    existing_tags = ["existing-tag"]
    
    suggestions = [
        # High confidence, should be kept
        TagSuggestion(tag="python", confidence=0.9, source="ml_model"),
        TagSuggestion(tag="javascript", confidence=0.8, source="specialized_analyzer"),
        
        # Duplicate of existing, should be filtered
        TagSuggestion(tag="existing-tag", confidence=0.9, source="url_analysis"),
        
        # Low confidence, should be filtered
        TagSuggestion(tag="lowconf", confidence=0.2, source="dictionary"),
        
        # Generic tag, should be filtered
        TagSuggestion(tag="web", confidence=0.7, source="content_analysis"),
        
        # Similar tags that should be merged
        TagSuggestion(tag="js", confidence=0.6, source="url_analysis"),  # Should merge with javascript
        TagSuggestion(tag="py", confidence=0.5, source="content_analysis"),  # Should merge with python
        
        # Duplicate tags (different cases)
        TagSuggestion(tag="React", confidence=0.7, source="ml_model"),
        TagSuggestion(tag="react", confidence=0.8, source="specialized_analyzer"),  # Higher confidence
        
        # Additional tags to test count limiting
        TagSuggestion(tag="nodejs", confidence=0.6, source="url_analysis"),  # Should become "node"
        TagSuggestion(tag="typescript", confidence=0.5, source="dictionary"),
        TagSuggestion(tag="docker", confidence=0.4, source="content_analysis"),
    ]
    
    result = controller.process_tag_suggestions(suggestions, existing_tags)
    
    # Verify results
    assert len(result) <= 5, f"Should have at most 5 tags, got {len(result)}: {result}"
    
    # High confidence tags should be present (after merging)
    assert "python" in result, "High confidence python should be kept"
    assert "javascript" in result, "High confidence javascript should be kept"
    
    # Existing tag should not be duplicated
    assert "existing-tag" not in result, "Existing tag should not be duplicated"
    
    # Low confidence should be filtered
    assert "lowconf" not in result, "Low confidence tag should be filtered"
    
    # Generic tag should be filtered
    assert "web" not in result, "Generic tag should be filtered"
    
    # Similar tags should be merged
    assert "js" not in result, "js should be merged to javascript"
    assert "py" not in result, "py should be merged to python"
    
    # Duplicates should be resolved to highest confidence
    assert "react" in result, "react should be kept with highest confidence"
    
    # Similarity merging should work
    if "node" in result:
        assert "nodejs" not in result, "nodejs should be merged to node"
    
    print(f"✅ Comprehensive pipeline works correctly. Final tags: {result}")


def main():
    """Run all tests"""
    print("🧪 Testing Tag Quality Control and Merging functionality\n")
    
    try:
        test_confidence_threshold_filtering()
        test_tag_deduplication()
        test_similarity_merging()
        test_generic_tag_filtering()
        test_tag_count_limiting()
        test_existing_tags_exclusion()
        test_comprehensive_pipeline()
        
        print("\n🎉 All tests passed! Tag Quality Control implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()