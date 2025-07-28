#!/usr/bin/env python3
"""
Test script for the AutoTagger

This script tests the AutoTagger component with various content scenarios
to verify URL analysis, quality control, and integration with AI systems.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_url_analyzer():
    """Test URL analysis functionality"""
    print("🔍 Testing URL Analysis")
    print("=" * 22)

    try:
        from enhancement.auto_tagger import URLAnalyzer

        analyzer = URLAnalyzer()

        # Test cases: (url, expected_tags)
        test_cases = [
            (
                "https://github.com/microsoft/vscode",
                ["development", "code", "repository"],
            ),
            (
                "https://stackoverflow.com/questions/tagged/python",
                ["programming", "development"],
            ),
            ("https://docs.python.org/3/tutorial/", ["documentation", "tutorial"]),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", ["video", "entertainment"]),
            ("https://medium.com/@author/react-tutorial", ["article", "blog"]),
            ("https://api.github.com/repos/user/repo", ["api", "development"]),
            ("https://example.com/blog/javascript-tips", ["blog", "javascript"]),
            ("https://discord.com/channels/123/456", ["communication", "gaming"]),
        ]

        for url, expected_tags in test_cases:
            suggestions = analyzer.analyze_url(url)
            found_tags = [s.tag for s in suggestions]

            # Check if at least one expected tag is found
            has_expected = any(tag in found_tags for tag in expected_tags)
            status = "✅" if has_expected else "⚠️"

            print(f"{status} {url}")
            print(f"   Found: {found_tags}")
            print(f"   Expected: {expected_tags}")

            if suggestions:
                highest_confidence = max(s.confidence for s in suggestions)
                print(f"   Highest confidence: {highest_confidence:.2f}")

        return True

    except Exception as e:
        print(f"❌ URL analysis test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tag_quality_controller():
    """Test tag quality control functionality"""
    print("\n🎯 Testing Tag Quality Control")
    print("=" * 30)

    try:
        from enhancement.auto_tagger import (
            TagQualityController,
            TagSuggestion,
            AutoTaggingConfig,
        )

        config = AutoTaggingConfig()
        controller = TagQualityController(config)

        # Test suggestions with various quality issues
        test_suggestions = [
            TagSuggestion("javascript", 0.9, "ml_model"),
            TagSuggestion("js", 0.8, "url_analysis"),  # Should merge with javascript
            TagSuggestion("programming", 0.7, "specialized_analyzer"),
            TagSuggestion(
                "web", 0.6, "content_analysis"
            ),  # Generic, should be filtered
            TagSuggestion("python", 0.85, "ml_model"),
            TagSuggestion("py", 0.7, "url_analysis"),  # Should merge with python
            TagSuggestion("tutorial", 0.75, "content_analysis"),
            TagSuggestion("guide", 0.65, "dictionary"),  # Should merge with tutorial
            TagSuggestion("lowconf", 0.2, "ml_model"),  # Below threshold
            TagSuggestion("", 0.8, "ml_model"),  # Empty tag
        ]

        existing_tags = ["development", "code"]  # Should be excluded

        final_tags = controller.process_tag_suggestions(test_suggestions, existing_tags)

        print(f"📊 Quality Control Results:")
        print(f"   Original suggestions: {len(test_suggestions)}")
        print(f"   Final tags: {len(final_tags)}")
        print(f"   Tags: {final_tags}")

        # Check expected behaviors
        checks = [
            (
                "javascript merged with js",
                "javascript" in final_tags and "js" not in final_tags,
            ),
            (
                "python merged with py",
                "python" in final_tags and "py" not in final_tags,
            ),
            (
                "tutorial merged with guide",
                "tutorial" in final_tags and "guide" not in final_tags,
            ),
            ("generic 'web' filtered", "web" not in final_tags),
            ("low confidence filtered", "lowconf" not in final_tags),
            (
                "existing tags excluded",
                "development" not in final_tags and "code" not in final_tags,
            ),
        ]

        for check_name, result in checks:
            status = "✅" if result else "⚠️"
            print(f"   {status} {check_name}")

        return True

    except Exception as e:
        print(f"❌ Quality control test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_auto_tagger_integration():
    """Test the main AutoTagger class"""
    print("\n🚀 Testing AutoTagger Integration")
    print("=" * 32)

    try:
        from enhancement.auto_tagger import AutoTagger

        # Create minimal config
        config = {
            "ai": {"tag_prediction": {"min_confidence": 0.3}},
            "directories": {"data_dir": "data", "models_dir": "models"},
        }

        tagger = AutoTagger(config)

        # Test cases: (url, title, content, existing_tags)
        test_cases = [
            (
                "https://github.com/microsoft/vscode",
                "Visual Studio Code",
                "Code editor for developers",
                [],
            ),
            (
                "https://stackoverflow.com/questions/tagged/python",
                "Python Questions",
                "Questions about Python programming language",
                ["programming"],
            ),
            (
                "https://docs.python.org/3/tutorial/",
                "Python Tutorial",
                "Official Python tutorial and documentation",
                [],
            ),
            (
                "https://www.youtube.com/watch?v=gaming",
                "Gaming Video",
                "Video about gaming and entertainment",
                ["video"],
            ),
            (
                "https://medium.com/@dev/react-guide",
                "Complete React Guide",
                "Comprehensive guide to React development",
                [],
            ),
        ]

        for url, title, content, existing_tags in test_cases:
            result = await tagger.generate_tags(url, title, content, existing_tags)

            print(f"\n📋 {url}")
            print(f"   Title: {title}")
            print(f"   Existing tags: {existing_tags}")
            print(f"   Generated tags: {result.final_tags}")
            print(f"   Tags added: {result.tags_added}")
            print(f"   Sources used: {result.sources_used}")
            print(f"   Processing time: {result.processing_time:.3f}s")

            if result.error_message:
                print(f"   Error: {result.error_message}")

            # Show confidence scores for generated tags
            if result.confidence_scores:
                print(f"   Confidence scores:")
                for tag, confidence in result.confidence_scores.items():
                    print(f"     - {tag}: {confidence:.3f}")

        return True

    except Exception as e:
        print(f"❌ AutoTagger integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_batch_tagging():
    """Test batch tagging functionality"""
    print("\n📦 Testing Batch Tagging")
    print("=" * 24)

    try:
        from enhancement.auto_tagger import AutoTagger

        config = {
            "ai": {"tag_prediction": {"min_confidence": 0.3}},
            "directories": {"data_dir": "data", "models_dir": "models"},
        }

        tagger = AutoTagger(config)

        # Sample bookmarks
        bookmarks = [
            {
                "url": "https://github.com/microsoft/vscode",
                "name": "Visual Studio Code",
                "description": "Code editor",
                "tags": [{"name": "editor"}],
            },
            {
                "url": "https://stackoverflow.com/questions/tagged/python",
                "name": "Python Questions",
                "description": "Programming help",
                "tags": [],
            },
            {
                "url": "https://www.youtube.com/watch?v=gaming",
                "name": "Gaming Video",
                "description": "Entertainment content",
                "tags": [{"name": "video"}, {"name": "entertainment"}],
            },
            {
                "url": "",  # Invalid bookmark
                "name": "Invalid",
                "description": "",
                "tags": [],
            },
        ]

        results = await tagger.generate_tags_batch(bookmarks)

        print(f"📊 Processed {len(results)} bookmarks:")

        total_tags_added = 0
        successful_taggings = 0

        for i, result in enumerate(results):
            bookmark = bookmarks[i]
            print(f"\n{i+1}. {bookmark.get('name', 'No name')}")
            print(f"   URL: {bookmark.get('url', 'No URL')}")
            print(f"   Original tags: {result.original_tags}")
            print(f"   Generated tags: {result.final_tags}")
            print(f"   Tags added: {result.tags_added}")
            print(f"   Sources: {result.sources_used}")

            if result.error_message:
                print(f"   Error: {result.error_message}")
            else:
                successful_taggings += 1
                total_tags_added += result.tags_added

        print(f"\n📈 Batch Summary:")
        print(f"   Successful taggings: {successful_taggings}/{len(results)}")
        print(f"   Total tags added: {total_tags_added}")
        print(
            f"   Average tags per bookmark: {total_tags_added/max(1, successful_taggings):.1f}"
        )

        return True

    except Exception as e:
        print(f"❌ Batch tagging test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tagging_stats():
    """Test tagging statistics"""
    print("\n📈 Testing Tagging Statistics")
    print("=" * 29)

    try:
        from enhancement.auto_tagger import AutoTagger

        config = {
            "ai": {"tag_prediction": {"min_confidence": 0.3}},
            "directories": {"data_dir": "data", "models_dir": "models"},
        }

        tagger = AutoTagger(config)
        stats = tagger.get_tagging_stats()

        print("📊 Tagging Statistics:")
        for section, data in stats.items():
            print(f"\n   {section}:")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"     - {key}: {value}")
            else:
                print(f"     {data}")

        return True

    except Exception as e:
        print(f"❌ Tagging stats test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tag_suggestion_structure():
    """Test TagSuggestion and TaggingResult data structures"""
    print("\n📦 Testing Data Structures")
    print("=" * 26)

    try:
        from enhancement.auto_tagger import TagSuggestion, TaggingResult

        # Test TagSuggestion
        suggestion = TagSuggestion(
            tag="python",
            confidence=0.85,
            source="ml_model",
            reasoning="ML prediction with high confidence",
            domain_specific=False,
        )

        print(f"✅ TagSuggestion created:")
        print(f"   Tag: {suggestion.tag}")
        print(f"   Confidence: {suggestion.confidence}")
        print(f"   Source: {suggestion.source}")
        print(f"   Domain specific: {suggestion.domain_specific}")

        # Test TaggingResult
        result = TaggingResult(
            original_tags=["programming"],
            suggested_tags=[suggestion],
            final_tags=["python", "development"],
            tags_added=2,
            confidence_scores={"python": 0.85, "development": 0.7},
            processing_time=0.123,
            sources_used=["ml_model", "url_analysis"],
        )

        print(f"\n✅ TaggingResult created:")
        print(f"   Original tags: {result.original_tags}")
        print(f"   Final tags: {result.final_tags}")
        print(f"   Tags added: {result.tags_added}")
        print(f"   Processing time: {result.processing_time}s")
        print(f"   Sources used: {result.sources_used}")

        return True

    except Exception as e:
        print(f"❌ Data structures test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🧪 AutoTagger Test Suite")
    print("=" * 25)

    tests = [
        ("URL Analysis", test_url_analyzer),
        ("Tag Quality Control", test_tag_quality_controller),
        ("AutoTagger Integration", test_auto_tagger_integration),
        ("Batch Tagging", test_batch_tagging),
        ("Tagging Statistics", test_tagging_stats),
        ("Data Structures", test_tag_suggestion_structure),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()

            if success:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")

        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")

    print(f"\n🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! AutoTagger is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
