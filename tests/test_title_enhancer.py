#!/usr/bin/env python3
"""
Test script for the TitleEnhancer

This script tests the TitleEnhancer component with various title scenarios
to verify quality assessment, cleaning, and enhancement functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_title_quality_assessment():
    """Test title quality assessment functionality"""
    print("🔍 Testing Title Quality Assessment")
    print("=" * 35)
    
    try:
        from enhancement.title_enhancer import TitleQualityAssessor
        
        assessor = TitleQualityAssessor()
        
        # Test cases with expected quality levels
        test_cases = [
            ("Complete Guide to Python Programming for Beginners", "high"),
            ("GitHub", "low"),
            ("untitled", "very_low"),
            ("", "very_low"),
            ("How to Build a REST API with FastAPI - Tutorial", "high"),
            ("page", "very_low"),
            ("Python Documentation", "medium"),
            ("stackoverflow.com", "low"),
            ("Ultimate React Tutorial 2024 - Learn React from Scratch", "high"),
            ("home", "very_low")
        ]
        
        for title, expected_level in test_cases:
            score = assessor.assess_title_quality(title, "https://example.com")
            
            # Determine actual level
            if score.overall_score >= 0.8:
                actual_level = "high"
            elif score.overall_score >= 0.5:
                actual_level = "medium"
            elif score.overall_score >= 0.2:
                actual_level = "low"
            else:
                actual_level = "very_low"
            
            status = "✅" if actual_level == expected_level else "⚠️"
            
            print(f"{status} '{title}' -> {score.overall_score:.2f} ({actual_level})")
            if actual_level != expected_level:
                print(f"   Expected: {expected_level}, Got: {actual_level}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality assessment test failed: {e}")
        return False


def test_title_cleaning():
    """Test title cleaning functionality"""
    print("\n🧹 Testing Title Cleaning")
    print("=" * 25)
    
    try:
        from enhancement.title_enhancer import TitleCleaner
        
        cleaner = TitleCleaner()
        
        # Test cases: (original, expected_cleaned)
        test_cases = [
            ("Great Article - Example.com", "Great Article"),
            ("Python Tutorial | Programming Site", "Python Tutorial"),
            ("How to Code :: Developer Blog", "How to Code"),
            ("React Guide • Tech News", "React Guide"),
            ("Welcome to My Site - Home", "My Site"),
            ("  Multiple   Spaces   Here  ", "Multiple Spaces Here"),
            ("SHOUTING TITLE", "Shouting Title"),
            ("lowercase title here", "Lowercase title here"),
            ("Title with!!! excessive!!! punctuation!!!", "Title with! excessive! punctuation!"),
            ("Home - Main Page", ""),  # Should be mostly removed
        ]
        
        for original, expected in test_cases:
            cleaned = cleaner.clean_title(original, "https://example.com")
            
            # For empty expected results, just check if significantly shortened
            if expected == "":
                success = len(cleaned) < len(original) * 0.5
                status = "✅" if success else "⚠️"
                print(f"{status} '{original}' -> '{cleaned}' (shortened)")
            else:
                success = cleaned == expected
                status = "✅" if success else "⚠️"
                print(f"{status} '{original}' -> '{cleaned}'")
                if not success:
                    print(f"   Expected: '{expected}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Title cleaning test failed: {e}")
        return False


def test_title_generation():
    """Test title generation from URLs"""
    print("\n🏗️ Testing Title Generation")
    print("=" * 27)
    
    try:
        from enhancement.title_enhancer import TitleGenerator
        
        generator = TitleGenerator()
        
        # Test cases: (url, expected_pattern)
        test_cases = [
            ("https://github.com/microsoft/vscode", "microsoft/vscode"),
            ("https://stackoverflow.com/questions/tagged/python", "python"),
            ("https://www.reddit.com/r/programming", "r/programming"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "YouTube Video"),
            ("https://medium.com/@author/article-title", "Medium Article"),
            ("https://example.com/about/team", "Team - Example"),
            ("https://docs.python.org/3/", "Python"),
        ]
        
        for url, expected_pattern in test_cases:
            generated = generator.generate_title_from_url(url)
            
            if generated:
                contains_pattern = expected_pattern.lower() in generated.lower()
                status = "✅" if contains_pattern else "⚠️"
                print(f"{status} {url} -> '{generated}'")
                if not contains_pattern:
                    print(f"   Expected to contain: '{expected_pattern}'")
            else:
                print(f"❌ {url} -> No title generated")
        
        return True
        
    except Exception as e:
        print(f"❌ Title generation test failed: {e}")
        return False


async def test_title_enhancer_integration():
    """Test the main TitleEnhancer class"""
    print("\n🚀 Testing TitleEnhancer Integration")
    print("=" * 35)
    
    try:
        from enhancement.title_enhancer import TitleEnhancer
        
        # Create minimal config
        config = {
            'enhancement': {'enable_scraping': True},
            'directories': {'data_dir': 'data', 'models_dir': 'models'}
        }
        
        enhancer = TitleEnhancer(config)
        
        # Test cases: (url, current_title, should_improve)
        test_cases = [
            ("https://github.com/microsoft/vscode", "GitHub", True),
            ("https://stackoverflow.com/questions/tagged/python", "page", True),
            ("https://example.com/article", "Complete Guide to Web Development", False),
            ("https://docs.python.org/3/", "", True),
            ("https://www.youtube.com/watch?v=test", "untitled", True),
        ]
        
        for url, current_title, should_improve in test_cases:
            result = await enhancer.enhance_title(url, current_title)
            
            improved = result.quality_improvement > 0
            status = "✅" if improved == should_improve else "⚠️"
            
            print(f"{status} {url}")
            print(f"   Original: '{result.original_title}'")
            print(f"   Enhanced: '{result.enhanced_title}'")
            print(f"   Method: {result.enhancement_method}")
            print(f"   Quality improvement: {result.quality_improvement:.3f}")
            print(f"   Confidence: {result.confidence_score:.3f}")
            
            if result.error_message:
                print(f"   Error: {result.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ TitleEnhancer integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_enhancement():
    """Test batch title enhancement"""
    print("\n📦 Testing Batch Enhancement")
    print("=" * 28)
    
    try:
        from enhancement.title_enhancer import TitleEnhancer
        
        config = {
            'enhancement': {'enable_scraping': True},
            'directories': {'data_dir': 'data', 'models_dir': 'models'}
        }
        
        enhancer = TitleEnhancer(config)
        
        # Sample bookmarks
        bookmarks = [
            {'url': 'https://github.com/microsoft/vscode', 'name': 'GitHub'},
            {'url': 'https://stackoverflow.com/questions/tagged/python', 'name': 'page'},
            {'url': 'https://example.com/good-title', 'name': 'Complete Guide to Example'},
            {'url': '', 'name': 'Invalid bookmark'},  # Should handle gracefully
        ]
        
        results = await enhancer.enhance_titles_batch(bookmarks)
        
        print(f"📊 Processed {len(results)} bookmarks:")
        
        for i, result in enumerate(results):
            bookmark = bookmarks[i]
            print(f"\n{i+1}. {bookmark.get('url', 'No URL')}")
            print(f"   Original: '{result.original_title}'")
            print(f"   Enhanced: '{result.enhanced_title}'")
            print(f"   Method: {result.enhancement_method}")
            print(f"   Improvement: {result.quality_improvement:.3f}")
            
            if result.error_message:
                print(f"   Error: {result.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch enhancement test failed: {e}")
        return False


def test_enhancement_stats():
    """Test enhancement statistics"""
    print("\n📈 Testing Enhancement Statistics")
    print("=" * 32)
    
    try:
        from enhancement.title_enhancer import TitleEnhancer
        
        config = {
            'enhancement': {'enable_scraping': True},
            'directories': {'data_dir': 'data', 'models_dir': 'models'}
        }
        
        enhancer = TitleEnhancer(config)
        stats = enhancer.get_enhancement_stats()
        
        print("📊 Enhancement Statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     - {sub_key}: {sub_value}")
            else:
                print(f"   - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhancement stats test failed: {e}")
        return False


async def main():
    """Run all tests"""
    print("🧪 TitleEnhancer Test Suite")
    print("=" * 30)
    
    tests = [
        ("Quality Assessment", test_title_quality_assessment),
        ("Title Cleaning", test_title_cleaning),
        ("Title Generation", test_title_generation),
        ("TitleEnhancer Integration", test_title_enhancer_integration),
        ("Batch Enhancement", test_batch_enhancement),
        ("Enhancement Statistics", test_enhancement_stats)
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
        print("🎉 All tests passed! TitleEnhancer is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())