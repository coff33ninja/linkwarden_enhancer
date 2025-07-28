#!/usr/bin/env python3
"""
Simple test for the Enhancement Pipeline structure

This script tests the pipeline structure without requiring all dependencies.
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_pipeline_import():
    """Test that the pipeline can be imported"""
    print("🧪 Testing Pipeline Import")
    print("=" * 30)
    
    try:
        from enhancement.pipeline import (
            EnhancementPipeline, 
            EnhancementConfig, 
            EnhancementOptions, 
            EnhancementMode,
            EnhancementStats,
            EnhancementResult
        )
        
        print("✅ Successfully imported pipeline classes")
        
        # Test enum values
        print(f"📋 Available modes: {[mode.value for mode in EnhancementMode]}")
        
        # Test configuration creation
        config = EnhancementConfig()
        print(f"⚙️  Default config mode: {config.mode.value}")
        print(f"⚙️  Default batch size: {config.batch_size}")
        
        # Test options creation
        options = EnhancementOptions()
        print(f"🎯 Default options - enhance titles: {options.enhance_titles}")
        print(f"🎯 Default options - generate tags: {options.generate_tags}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\n🚀 Testing Pipeline Initialization")
    print("=" * 35)
    
    try:
        from enhancement.pipeline import EnhancementPipeline
        
        # Create minimal config
        config = {
            'enhancement': {'enable_scraping': True},
            'directories': {'data_dir': 'data', 'models_dir': 'models'},
            'progress': {},
            'safety': {}
        }
        
        # Initialize pipeline
        pipeline = EnhancementPipeline(config)
        print("✅ Pipeline initialized successfully")
        
        # Test stats method
        stats = pipeline.get_pipeline_stats()
        print(f"📊 Pipeline stats keys: {list(stats.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_pipeline_validation():
    """Test bookmark validation"""
    print("\n🔍 Testing Bookmark Validation")
    print("=" * 32)
    
    try:
        from enhancement.pipeline import EnhancementPipeline
        
        config = {
            'enhancement': {'enable_scraping': True},
            'directories': {'data_dir': 'data', 'models_dir': 'models'},
            'progress': {},
            'safety': {}
        }
        
        pipeline = EnhancementPipeline(config)
        
        # Test valid bookmark
        valid_bookmark = {
            'id': 1,
            'url': 'https://example.com',
            'name': 'Example',
            'tags': []
        }
        
        is_valid = pipeline._is_valid_bookmark(valid_bookmark)
        print(f"✅ Valid bookmark test: {is_valid}")
        
        # Test invalid bookmark
        invalid_bookmark = {
            'id': 2,
            'url': 'not-a-url',
            'name': 'Invalid'
        }
        
        is_invalid = pipeline._is_valid_bookmark(invalid_bookmark)
        print(f"❌ Invalid bookmark test: {not is_invalid}")
        
        # Test validation method
        test_bookmarks = [valid_bookmark, invalid_bookmark]
        validated = await pipeline._validate_input_bookmarks(test_bookmarks)
        print(f"📋 Validated {len(validated)}/2 bookmarks")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


async def test_quality_assessment():
    """Test quality assessment methods"""
    print("\n📏 Testing Quality Assessment")
    print("=" * 30)
    
    try:
        from enhancement.pipeline import EnhancementPipeline
        
        config = {
            'enhancement': {'enable_scraping': True},
            'directories': {'data_dir': 'data', 'models_dir': 'models'},
            'progress': {},
            'safety': {}
        }
        
        pipeline = EnhancementPipeline(config)
        
        # Test title quality assessment
        good_title = "Complete Guide to Python Programming"
        poor_title = "untitled"
        empty_title = ""
        
        good_score = pipeline._assess_title_quality(good_title)
        poor_score = pipeline._assess_title_quality(poor_title)
        empty_score = pipeline._assess_title_quality(empty_title)
        
        print(f"📊 Good title score: {good_score:.2f}")
        print(f"📊 Poor title score: {poor_score:.2f}")
        print(f"📊 Empty title score: {empty_score:.2f}")
        
        # Test generic title detection
        generic_titles = ["untitled", "page", "home", "index", "document"]
        for title in generic_titles:
            is_generic = pipeline._is_generic_title(title)
            print(f"🔍 '{title}' is generic: {is_generic}")
        
        # Test title cleaning
        messy_title = "Great Article - Example.com | News Site"
        cleaned = pipeline._clean_title(messy_title, "https://example.com")
        print(f"🧹 Cleaned title: '{messy_title}' → '{cleaned}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Quality assessment test failed: {e}")
        return False


def test_data_structures():
    """Test data structure creation"""
    print("\n📦 Testing Data Structures")
    print("=" * 27)
    
    try:
        from enhancement.pipeline import (
            EnhancementStats, 
            EnhancementResult, 
            EnhancementConfig,
            EnhancementOptions
        )
        
        # Test stats
        stats = EnhancementStats()
        stats.total_bookmarks = 100
        stats.processed_bookmarks = 95
        stats.calculate_success_rate()
        print(f"📊 Success rate calculation: {stats.success_rate}%")
        
        # Test result
        result = EnhancementResult(
            success=True,
            enhanced_bookmarks=[],
            stats=stats,
            errors=[],
            warnings=[]
        )
        print(f"✅ Result created: success={result.success}")
        
        # Test config with custom options
        options = EnhancementOptions(
            enhance_titles=True,
            generate_tags=False,
            max_tags_per_bookmark=5
        )
        
        config = EnhancementConfig(options=options)
        print(f"⚙️  Custom config: enhance_titles={config.options.enhance_titles}")
        print(f"⚙️  Custom config: generate_tags={config.options.generate_tags}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


async def main():
    """Run all simple tests"""
    print("🧪 Enhancement Pipeline Simple Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_pipeline_import),
        ("Initialization Test", test_pipeline_initialization),
        ("Validation Test", test_pipeline_validation),
        ("Quality Assessment Test", test_quality_assessment),
        ("Data Structures Test", test_data_structures)
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
        print("🎉 All tests passed! Pipeline structure is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())