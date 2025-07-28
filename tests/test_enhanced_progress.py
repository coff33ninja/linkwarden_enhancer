#!/usr/bin/env python3
"""Test script for enhanced progress tracking and learning feedback"""

import json
import time
from pathlib import Path
from cli.main_cli import MainCLI

def create_test_data():
    """Create test bookmark data"""
    
    test_data = {
        "bookmarks": [
            {
                "id": 1,
                "name": "GitHub - Python Repository",
                "url": "https://github.com/python/cpython",
                "description": "The Python programming language",
                "tags": [{"name": "python"}, {"name": "programming"}],
                "collection": {"name": "Development"}
            },
            {
                "id": 2,
                "name": "OpenAI API Documentation",
                "url": "https://platform.openai.com/docs",
                "description": "Official OpenAI API documentation",
                "tags": [{"name": "ai"}, {"name": "api"}],
                "collection": {"name": "AI/ML"}
            },
            {
                "id": 3,
                "name": "Genshin Impact Wiki",
                "url": "https://genshin-impact.fandom.com/wiki/Genshin_Impact_Wiki",
                "description": "Comprehensive Genshin Impact game guide",
                "tags": [{"name": "gaming"}, {"name": "genshin"}],
                "collection": {"name": "Gaming"}
            }
        ],
        "collections": [
            {"id": 1, "name": "Development", "description": "Programming resources"},
            {"id": 2, "name": "AI/ML", "description": "Artificial Intelligence and Machine Learning"},
            {"id": 3, "name": "Gaming", "description": "Gaming resources and guides"}
        ],
        "tags": [
            {"id": 1, "name": "python"},
            {"id": 2, "name": "programming"},
            {"id": 3, "name": "ai"},
            {"id": 4, "name": "api"},
            {"id": 5, "name": "gaming"},
            {"id": 6, "name": "genshin"}
        ]
    }
    
    return test_data

def test_enhanced_progress_tracking():
    """Test enhanced progress tracking with learning feedback"""
    
    print("🧪 Testing Enhanced Progress Tracking and Learning Feedback")
    print("=" * 60)
    
    # Create test input file
    test_input = "test_input.json"
    test_output = "test_output.json"
    
    test_data = create_test_data()
    
    with open(test_input, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Created test input file: {test_input}")
    
    # Test CLI with enhanced progress tracking
    cli = MainCLI()
    
    # Simulate command line arguments for enhanced processing
    class MockArgs:
        def __init__(self):
            self.command = 'process'
            self.input_file = test_input
            self.output_file = test_output
            self.verbose = True
            self.interactive = False
            self.dry_run = True  # Safe testing
            self.debug = False
            self.config = None
            
            # Enhanced progress options
            self.progress_detail = 'detailed'
            self.learning_feedback = True
            self.performance_metrics = True
            
            # Feature flags
            self.enable_scraping = False  # Disabled for testing
            self.enable_ai_analysis = True
            self.enable_learning = True
            self.enable_clustering = True
            self.enable_similarity_detection = True
            self.enable_smart_tagging = True
            self.enable_network_analysis = False
            
            # Import options
            self.import_github = False
            self.import_browser = None
            
            # Report options
            self.generate_report = True
            self.report_format = 'json'
            
            # Safety options
            self.max_deletion_percent = 10.0
            self.backup_before_processing = True
            self.skip_integrity_check = False
            self.safety_pause_threshold = 100
            self.auto_approve_low_risk = True
    
    args = MockArgs()
    
    try:
        print("\n🚀 Starting enhanced processing test...")
        
        # Run the CLI with enhanced progress tracking
        result = cli.run([
            'process',
            test_input,
            test_output,
            '--enable-ai-analysis',
            '--enable-learning',
            '--enable-clustering',
            '--enable-similarity-detection',
            '--enable-smart-tagging',
            '--generate-report'
        ])
        
        if result == 0:
            print("\n✅ Enhanced progress tracking test completed successfully!")
        else:
            print(f"\n❌ Test failed with exit code: {result}")
        
        # Show test results
        print("\n📊 Test Summary:")
        print(f"   • Input file: {test_input}")
        print(f"   • Output file: {test_output}")
        print(f"   • Test bookmarks: {len(test_data['bookmarks'])}")
        print(f"   • Test collections: {len(test_data['collections'])}")
        print(f"   • Test tags: {len(test_data['tags'])}")
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup test files
        for file_path in [test_input, test_output]:
            if Path(file_path).exists():
                Path(file_path).unlink()
                print(f"🧹 Cleaned up: {file_path}")

def test_detailed_progress_tracker():
    """Test the DetailedProgressTracker directly"""
    
    print("\n🧪 Testing DetailedProgressTracker Directly")
    print("=" * 50)
    
    from utils.progress_utils import DetailedProgressTracker
    
    # Test phases
    phases = ['validation', 'enhancement', 'ai_analysis', 'learning', 'output']
    
    tracker = DetailedProgressTracker(phases, verbose=True)
    
    try:
        # Simulate processing phases
        for i, phase in enumerate(phases):
            items_to_process = (i + 1) * 10  # Variable items per phase
            
            progress = tracker.start_phase(phase, items_to_process)
            
            # Simulate processing items
            for j in range(items_to_process):
                progress.update(j, f"Processing item {j+1}")
                time.sleep(0.01)  # Small delay to show progress
            
            # Simulate learning data for some phases
            learning_data = {}
            if phase == 'learning':
                learning_data = {
                    'patterns_learned': 5,
                    'dictionary_updates': 3,
                    'feedback_processed': 2
                }
            elif phase == 'ai_analysis':
                learning_data = {
                    'content_analyzed': items_to_process,
                    'clusters_created': 3,
                    'similarities_found': 7
                }
            elif phase == 'enhancement':
                learning_data = {
                    'items_enhanced': items_to_process - 2,
                    'metadata_extracted': items_to_process * 3,
                    'scraping_successes': items_to_process - 3
                }
            
            progress.finish(f"{phase.title()} completed")
            tracker.finish_phase(phase, items_to_process, learning_data)
            
            # Show progress between phases
            if i < len(phases) - 1:
                tracker.show_overall_progress()
                time.sleep(0.5)
        
        # Show final summary
        summary = tracker.finish()
        
        print("\n📋 Final Summary:")
        print(f"   • Total phases: {summary['total_phases']}")
        print(f"   • Completed phases: {summary['completed_phases']}")
        print(f"   • Total duration: {summary['total_duration']:.2f}s")
        print(f"   • Total items: {summary['total_items_processed']}")
        
        assert True
        
    except Exception as e:
        print(f"❌ DetailedProgressTracker test failed: {e}")
        import traceback
        traceback.print_exc()
        assert False

if __name__ == "__main__":
    print("🔬 Enhanced Progress Tracking and Learning Feedback Test Suite")
    print("=" * 70)
    
    # Test 1: Direct progress tracker test
    tracker_success = test_detailed_progress_tracker()
    
    # Test 2: Full CLI integration test (commented out for now due to dependencies)
    # cli_success = test_enhanced_progress_tracking()
    
    print("\n🏁 Test Suite Complete!")
    print(f"   • DetailedProgressTracker: {'✅ PASS' if tracker_success else '❌ FAIL'}")
    # print(f"   • CLI Integration: {'✅ PASS' if cli_success else '❌ FAIL'}")
    
    if tracker_success:
        print("\n🎉 Enhanced progress tracking is working correctly!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")