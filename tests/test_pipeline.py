#!/usr/bin/env python3
"""
Test script for the Enhancement Pipeline

This script tests the new EnhancementPipeline with sample bookmark data
to verify integration with existing components.
"""

import asyncio
import json
from pathlib import Path

from enhancement.pipeline import EnhancementPipeline, EnhancementConfig, EnhancementOptions, EnhancementMode


def load_sample_config():
    """Load sample configuration for testing"""
    return {
        'enhancement': {
            'enable_scraping': True,
            'enable_caching': True,
            'max_concurrent_requests': 3,
            'default_timeout': 10,
            'scrapers_to_use': ['beautifulsoup']
        },
        'ai': {
            'tag_prediction': {
                'min_confidence': 0.3,
                'max_predictions': 8
            }
        },
        'similarity': {
            'model_name': 'all-MiniLM-L6-v2',
            'duplicate_threshold': 0.85
        },
        'directories': {
            'data_dir': 'data',
            'models_dir': 'models'
        },
        'progress': {
            'enable_progress_tracking': True
        },
        'safety': {
            'max_deletion_percentage': 10.0,
            'max_error_rate': 5.0
        }
    }


def create_sample_bookmarks():
    """Create sample bookmark data for testing"""
    return [
        {
            'id': 1,
            'url': 'https://github.com/microsoft/vscode',
            'name': 'GitHub',
            'description': '',
            'tags': []
        },
        {
            'id': 2,
            'url': 'https://stackoverflow.com/questions/tagged/python',
            'name': 'python - Stack Overflow',
            'description': 'Questions about Python programming',
            'tags': [{'name': 'programming'}]
        },
        {
            'id': 3,
            'url': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            'name': 'YouTube',
            'description': '',
            'tags': []
        },
        {
            'id': 4,
            'url': 'https://paimon.moe/wish',
            'name': 'Paimon.moe - Genshin Impact Wish Tracker',
            'description': 'Track your wishes in Genshin Impact',
            'tags': [{'name': 'gaming'}]
        },
        {
            'id': 5,
            'url': 'https://docs.python.org/3/',
            'name': 'Python Documentation',
            'description': '',
            'tags': []
        }
    ]


async def test_full_pipeline():
    """Test the full enhancement pipeline"""
    print("🚀 Testing Enhancement Pipeline - Full Mode")
    print("=" * 50)
    
    # Initialize pipeline
    config = load_sample_config()
    pipeline = EnhancementPipeline(config)
    
    # Get sample bookmarks
    bookmarks = create_sample_bookmarks()
    print(f"📚 Processing {len(bookmarks)} sample bookmarks")
    
    # Configure for full enhancement
    enhancement_config = EnhancementConfig(
        mode=EnhancementMode.FULL,
        options=EnhancementOptions(
            enhance_titles=True,
            generate_tags=True,
            create_descriptions=True,
            detect_duplicates=False,  # Skip for this test
            max_tags_per_bookmark=8
        ),
        batch_size=2,
        max_concurrent=2,
        enable_ai_summarization=False  # Disable AI for testing
    )
    
    try:
        # Process bookmarks
        result = await pipeline.process_bookmarks(bookmarks, enhancement_config)
        
        # Display results
        print(f"\n✅ Enhancement completed: {result.success}")
        print(f"📊 Statistics:")
        print(f"   - Total bookmarks: {result.stats.total_bookmarks}")
        print(f"   - Processed: {result.stats.processed_bookmarks}")
        print(f"   - Titles enhanced: {result.stats.titles_enhanced}")
        print(f"   - Tags generated: {result.stats.tags_generated}")
        print(f"   - Descriptions created: {result.stats.descriptions_created}")
        print(f"   - Processing time: {result.stats.processing_time:.2f}s")
        print(f"   - Success rate: {result.stats.success_rate:.1f}%")
        
        if result.errors:
            print(f"❌ Errors: {result.errors}")
        
        if result.warnings:
            print(f"⚠️  Warnings: {result.warnings}")
        
        # Show enhanced bookmarks
        print(f"\n📋 Enhanced Bookmarks:")
        for i, bookmark in enumerate(result.enhanced_bookmarks, 1):
            print(f"\n{i}. {bookmark.get('name', 'No title')}")
            print(f"   URL: {bookmark.get('url', 'No URL')}")
            print(f"   Description: {bookmark.get('description', 'No description')}")
            tags = [tag.get('name', str(tag)) if isinstance(tag, dict) else str(tag) 
                   for tag in bookmark.get('tags', [])]
            print(f"   Tags: {', '.join(tags) if tags else 'No tags'}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_dry_run_mode():
    """Test the pipeline in dry-run mode"""
    print("\n🔍 Testing Enhancement Pipeline - Dry Run Mode")
    print("=" * 50)
    
    # Initialize pipeline
    config = load_sample_config()
    pipeline = EnhancementPipeline(config)
    
    # Get sample bookmarks
    bookmarks = create_sample_bookmarks()
    
    # Configure for dry-run
    enhancement_config = EnhancementConfig(
        mode=EnhancementMode.DRY_RUN,
        options=EnhancementOptions(
            enhance_titles=True,
            generate_tags=True,
            create_descriptions=True
        )
    )
    
    try:
        # Process bookmarks in dry-run mode
        result = await pipeline.process_bookmarks(bookmarks, enhancement_config)
        
        print(f"✅ Dry-run completed: {result.success}")
        print(f"📊 Preview results:")
        
        for i, bookmark in enumerate(result.enhanced_bookmarks, 1):
            preview = bookmark.get('_enhancement_preview', {})
            print(f"\n{i}. {bookmark.get('name', 'No title')}")
            print(f"   Would enhance title: {preview.get('would_enhance_title', False)}")
            print(f"   Would generate tags: {preview.get('would_generate_tags', False)}")
            print(f"   Would create description: {preview.get('would_create_description', False)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Dry-run test failed: {e}")
        return None


async def test_selective_enhancement():
    """Test selective enhancement options"""
    print("\n🎯 Testing Enhancement Pipeline - Selective Mode")
    print("=" * 50)
    
    # Initialize pipeline
    config = load_sample_config()
    pipeline = EnhancementPipeline(config)
    
    # Get sample bookmarks
    bookmarks = create_sample_bookmarks()
    
    # Configure for selective enhancement (tags only)
    options = EnhancementOptions(
        enhance_titles=False,
        generate_tags=True,
        create_descriptions=False,
        detect_duplicates=False,
        max_tags_per_bookmark=5
    )
    
    try:
        # Process bookmarks selectively
        result = await pipeline.process_selective(bookmarks, options)
        
        print(f"✅ Selective enhancement completed: {result.success}")
        print(f"📊 Tags-only enhancement results:")
        print(f"   - Tags generated: {result.stats.tags_generated}")
        print(f"   - Titles enhanced: {result.stats.titles_enhanced} (should be 0)")
        print(f"   - Descriptions created: {result.stats.descriptions_created} (should be 0)")
        
        return result
        
    except Exception as e:
        print(f"❌ Selective enhancement test failed: {e}")
        return None


def test_pipeline_stats():
    """Test pipeline statistics"""
    print("\n📈 Testing Pipeline Statistics")
    print("=" * 50)
    
    try:
        config = load_sample_config()
        pipeline = EnhancementPipeline(config)
        
        stats = pipeline.get_pipeline_stats()
        
        print("🔧 Component Status:")
        for component, status in stats['components_initialized'].items():
            print(f"   - {component}: {'✅' if status else '❌'}")
        
        print("\n🤖 Model Status:")
        for model, status in stats['model_status'].items():
            print(f"   - {model}: {status}")
        
        print("\n⚙️  Configuration:")
        for key, value in stats['configuration'].items():
            print(f"   - {key}: {value}")
        
        return stats
        
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        return None


async def main():
    """Run all tests"""
    print("🧪 Enhancement Pipeline Test Suite")
    print("=" * 60)
    
    # Test 1: Pipeline statistics
    test_pipeline_stats()
    
    # Test 2: Dry-run mode
    await test_dry_run_mode()
    
    # Test 3: Selective enhancement
    await test_selective_enhancement()
    
    # Test 4: Full pipeline (may take longer due to web scraping)
    print("\n⏳ Running full pipeline test (may take a moment for web scraping)...")
    await test_full_pipeline()
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())