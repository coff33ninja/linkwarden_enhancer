#!/usr/bin/env python3
"""
Enhanced Progress Tracking Demo

This script demonstrates the enhanced progress tracking and learning feedback
capabilities of the Linkwarden Enhancer CLI.
"""

import json
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import the enhancer
sys.path.insert(0, str(Path(__file__).parent.parent))

from linkwarden_enhancer.cli.main_cli import MainCLI

def create_demo_bookmarks():
    """Create demonstration bookmark data"""
    
    return {
        "bookmarks": [
            {
                "id": 1,
                "name": "Python Official Documentation",
                "url": "https://docs.python.org/3/",
                "description": "Official Python 3 documentation",
                "tags": [{"name": "python"}, {"name": "documentation"}, {"name": "programming"}],
                "collection": {"name": "Development", "id": 1}
            },
            {
                "id": 2,
                "name": "GitHub - TensorFlow",
                "url": "https://github.com/tensorflow/tensorflow",
                "description": "An Open Source Machine Learning Framework for Everyone",
                "tags": [{"name": "machine-learning"}, {"name": "tensorflow"}, {"name": "ai"}],
                "collection": {"name": "AI/ML", "id": 2}
            },
            {
                "id": 3,
                "name": "Stack Overflow",
                "url": "https://stackoverflow.com/",
                "description": "Where developers learn, share, & build careers",
                "tags": [{"name": "programming"}, {"name": "community"}, {"name": "help"}],
                "collection": {"name": "Development", "id": 1}
            },
            {
                "id": 4,
                "name": "Hugging Face",
                "url": "https://huggingface.co/",
                "description": "The AI community building the future",
                "tags": [{"name": "ai"}, {"name": "nlp"}, {"name": "models"}],
                "collection": {"name": "AI/ML", "id": 2}
            },
            {
                "id": 5,
                "name": "Genshin Impact Official Site",
                "url": "https://genshin.hoyoverse.com/",
                "description": "Official Genshin Impact game website",
                "tags": [{"name": "gaming"}, {"name": "genshin"}, {"name": "rpg"}],
                "collection": {"name": "Gaming", "id": 3}
            },
            {
                "id": 6,
                "name": "Docker Hub",
                "url": "https://hub.docker.com/",
                "description": "Container registry and development platform",
                "tags": [{"name": "docker"}, {"name": "containers"}, {"name": "devops"}],
                "collection": {"name": "DevOps", "id": 4}
            },
            {
                "id": 7,
                "name": "OpenAI Platform",
                "url": "https://platform.openai.com/",
                "description": "OpenAI API platform and documentation",
                "tags": [{"name": "openai"}, {"name": "api"}, {"name": "gpt"}],
                "collection": {"name": "AI/ML", "id": 2}
            },
            {
                "id": 8,
                "name": "Kubernetes Documentation",
                "url": "https://kubernetes.io/docs/",
                "description": "Production-Grade Container Orchestration",
                "tags": [{"name": "kubernetes"}, {"name": "orchestration"}, {"name": "devops"}],
                "collection": {"name": "DevOps", "id": 4}
            }
        ],
        "collections": [
            {"id": 1, "name": "Development", "description": "Programming and development resources"},
            {"id": 2, "name": "AI/ML", "description": "Artificial Intelligence and Machine Learning"},
            {"id": 3, "name": "Gaming", "description": "Gaming resources and entertainment"},
            {"id": 4, "name": "DevOps", "description": "DevOps tools and practices"}
        ],
        "tags": [
            {"id": 1, "name": "python"},
            {"id": 2, "name": "documentation"},
            {"id": 3, "name": "programming"},
            {"id": 4, "name": "machine-learning"},
            {"id": 5, "name": "tensorflow"},
            {"id": 6, "name": "ai"},
            {"id": 7, "name": "community"},
            {"id": 8, "name": "help"},
            {"id": 9, "name": "nlp"},
            {"id": 10, "name": "models"},
            {"id": 11, "name": "gaming"},
            {"id": 12, "name": "genshin"},
            {"id": 13, "name": "rpg"},
            {"id": 14, "name": "docker"},
            {"id": 15, "name": "containers"},
            {"id": 16, "name": "devops"},
            {"id": 17, "name": "openai"},
            {"id": 18, "name": "api"},
            {"id": 19, "name": "gpt"},
            {"id": 20, "name": "kubernetes"},
            {"id": 21, "name": "orchestration"}
        ]
    }

def demo_basic_progress():
    """Demonstrate basic progress tracking"""
    
    print("🔄 Demo 1: Basic Progress Tracking")
    print("=" * 50)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        json.dump(create_demo_bookmarks(), input_file, indent=2)
        input_path = input_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        cli = MainCLI()
        
        # Run with basic progress
        result = cli.run([
            'process',
            input_path,
            output_path,
            '--dry-run',
            '--verbose'
        ])
        
        print(f"✅ Basic demo completed with result: {result}")
        
    finally:
        # Cleanup
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)

def demo_detailed_progress():
    """Demonstrate detailed progress tracking with learning feedback"""
    
    print("\n🔄 Demo 2: Detailed Progress with Learning Feedback")
    print("=" * 60)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
        json.dump(create_demo_bookmarks(), input_file, indent=2)
        input_path = input_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as output_file:
        output_path = output_file.name
    
    try:
        cli = MainCLI()
        
        # Run with detailed progress and learning feedback
        result = cli.run([
            'process',
            input_path,
            output_path,
            '--dry-run',
            '--verbose',
            '--progress-detail', 'detailed',
            '--learning-feedback',
            '--performance-metrics',
            '--enable-ai-analysis',
            '--enable-learning',
            '--enable-clustering',
            '--enable-similarity-detection',
            '--enable-smart-tagging',
            '--generate-report',
            '--report-format', 'json'
        ])
        
        print(f"✅ Detailed demo completed with result: {result}")
        
    finally:
        # Cleanup
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)

def demo_progress_tracker_direct():
    """Demonstrate direct usage of DetailedProgressTracker"""
    
    print("\n🔄 Demo 3: Direct Progress Tracker Usage")
    print("=" * 50)
    
    from linkwarden_enhancer.utils.progress_utils import DetailedProgressTracker
    import time
    
    # Define processing phases
    phases = ['data_loading', 'validation', 'enhancement', 'ai_analysis', 'learning', 'output']
    
    tracker = DetailedProgressTracker(phases, verbose=True)
    
    try:
        # Simulate each phase
        for i, phase in enumerate(phases):
            items_count = (i + 1) * 5  # Variable items per phase
            
            progress = tracker.start_phase(phase, items_count)
            
            # Simulate processing
            for j in range(items_count):
                progress.update(j, f"Processing {phase} item {j+1}")
                time.sleep(0.02)  # Small delay to show progress
            
            # Generate phase-specific learning data
            learning_data = {}
            if phase == 'validation':
                learning_data = {
                    'schema_valid': True,
                    'total_bookmarks': 8,
                    'total_collections': 4,
                    'total_tags': 21
                }
            elif phase == 'enhancement':
                learning_data = {
                    'items_enhanced': items_count - 1,
                    'metadata_extracted': items_count * 3,
                    'scraping_successes': items_count - 2,
                    'scraping_failures': 2
                }
            elif phase == 'ai_analysis':
                learning_data = {
                    'content_analyzed': items_count,
                    'clusters_created': 3,
                    'similarities_found': 5,
                    'tags_suggested': items_count * 2
                }
            elif phase == 'learning':
                learning_data = {
                    'patterns_learned': 7,
                    'dictionary_updates': 4,
                    'feedback_processed': 3,
                    'adaptations_made': 2
                }
            
            progress.finish(f"{phase.replace('_', ' ').title()} completed")
            tracker.finish_phase(phase, items_count, learning_data)
            
            # Show intermediate progress
            if i < len(phases) - 1:
                time.sleep(0.3)
        
        # Get final summary
        summary = tracker.finish()
        
        print("\n📊 Demo Summary:")
        print(f"   • Total phases: {summary['total_phases']}")
        print(f"   • Completed phases: {summary['completed_phases']}")
        print(f"   • Total duration: {summary['total_duration']:.2f}s")
        print(f"   • Items processed: {summary['total_items_processed']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct tracker demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all progress tracking demos"""
    
    print("🚀 Enhanced Progress Tracking Demonstration")
    print("=" * 70)
    print("This demo showcases the enhanced progress tracking and learning")
    print("feedback capabilities of the Linkwarden Enhancer CLI.")
    print()
    
    try:
        # Demo 1: Basic progress (commented out due to dependencies)
        # demo_basic_progress()
        
        # Demo 2: Detailed progress (commented out due to dependencies)  
        # demo_detailed_progress()
        
        # Demo 3: Direct progress tracker usage
        success = demo_progress_tracker_direct()
        
        print("\n🏁 Demo Complete!")
        print(f"   • Direct Progress Tracker: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if success:
            print("\n🎉 Enhanced progress tracking is working perfectly!")
            print("\nKey Features Demonstrated:")
            print("   • Multi-phase progress tracking")
            print("   • Real-time learning statistics")
            print("   • Performance metrics collection")
            print("   • Comprehensive progress summaries")
            print("   • Learning feedback integration")
        else:
            print("\n⚠️ Some demos failed. Check the output for details.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()