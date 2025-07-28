#!/usr/bin/env python3
"""
Demo script showing the continuous learning and adaptive intelligence system.
This demonstrates the functionality implemented in task 7.
"""

import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from linkwarden_enhancer.intelligence.continuous_learner import ContinuousLearner
from linkwarden_enhancer.intelligence.adaptive_intelligence import AdaptiveIntelligence, FeedbackType


def demo_continuous_learning_system():
    """Demonstrate the continuous learning and adaptive intelligence system"""
    
    # Create temporary directory for demo
    demo_dir = tempfile.mkdtemp()
    print(f"Demo data directory: {demo_dir}")
    
    try:
        # Initialize systems
        learner = ContinuousLearner(data_dir=demo_dir)
        adaptive_ai = AdaptiveIntelligence(data_dir=demo_dir, user_id="demo_user")
        
        print("\n=== Continuous Learning and Adaptive Intelligence Demo ===\n")
        
        # Sample bookmark data
        sample_bookmarks = [
            {
                'id': '1',
                'url': 'https://github.com/microsoft/vscode',
                'name': 'Visual Studio Code',
                'content': {'text_content': 'TypeScript code editor for development'},
                'collection_name': 'Development Tools',
                'tags': [{'name': 'editor'}, {'name': 'typescript'}, {'name': 'development'}]
            },
            {
                'id': '2',
                'url': 'https://paimon.moe/calculator',
                'name': 'Genshin Impact Calculator',
                'content': {'text_content': 'Character build calculator for Genshin Impact'},
                'collection_name': 'Gaming',
                'tags': [{'name': 'genshin'}, {'name': 'calculator'}, {'name': 'gaming'}]
            },
            {
                'id': '3',
                'url': 'https://arxiv.org/abs/2301.12345',
                'name': 'Deep Learning Research Paper',
                'content': {'text_content': 'Research on neural networks and machine learning'},
                'collection_name': 'Research',
                'tags': [{'name': 'ai'}, {'name': 'research'}, {'name': 'paper'}]
            },
            {
                'id': '4',
                'url': 'https://docs.docker.com/get-started/',
                'name': 'Docker Documentation',
                'content': {'text_content': 'Getting started guide for Docker containerization'},
                'collection_name': 'Development Tools',
                'tags': [{'name': 'docker'}, {'name': 'documentation'}, {'name': 'containers'}]
            },
            {
                'id': '5',
                'url': 'https://unity.com/learn/tutorials',
                'name': 'Unity Game Development Tutorial',
                'content': {'text_content': 'Learn C# programming and Unity engine for games'},
                'collection_name': 'Gaming',
                'tags': [{'name': 'unity'}, {'name': 'gamedev'}, {'name': 'tutorial'}]
            }
        ]
        
        # Phase 1: Initial Learning
        print("1. Initial Learning Phase")
        print("-" * 40)
        
        session_id = learner.start_learning_session("demo_initial")
        print(f"Started learning session: {session_id}")
        
        learning_results = learner.learn_from_new_bookmarks(sample_bookmarks)
        print(f"Processed {learning_results['bookmarks_processed']} bookmarks")
        print(f"Learned {learning_results['new_patterns_learned']} new patterns")
        print(f"Updated {learning_results['patterns_updated']} existing patterns")
        
        session = learner.end_learning_session()
        print(f"Session completed in {(session.end_time - session.start_time).total_seconds():.2f} seconds")
        
        # Show initial learning statistics
        stats = learner.get_learning_statistics()
        print(f"\nLearning Statistics:")
        print(f"  Total patterns learned: {stats['learning_metrics']['patterns_learned']}")
        print(f"  Total bookmarks processed: {stats['learning_metrics']['total_bookmarks_processed']}")
        print(f"  Learning sessions: {stats['learning_metrics']['learning_sessions']}")
        
        # Phase 2: Pattern Predictions
        print("\n2. Pattern Prediction Phase")
        print("-" * 40)
        
        # Test category prediction
        test_url = 'https://github.com/facebook/react'
        test_title = 'React JavaScript Library'
        test_content = 'A JavaScript library for building user interfaces'
        
        category_predictions = learner.predict_category(test_url, test_title, test_content)
        print(f"Category predictions for '{test_title}':")
        for category, confidence in category_predictions[:3]:
            print(f"  {category}: {confidence:.3f}")
        
        # Test tag prediction
        tag_predictions = learner.predict_tags(test_url, test_title, test_content)
        print(f"\nTag predictions for '{test_title}':")
        for tag, confidence in tag_predictions[:5]:
            print(f"  {tag}: {confidence:.3f}")
        
        # Phase 3: User Feedback and Adaptation
        print("\n3. User Feedback and Adaptation Phase")
        print("-" * 40)
        
        # Simulate user accepting a category suggestion
        feedback_id = adaptive_ai.track_user_feedback(
            FeedbackType.SUGGESTION_ACCEPTED,
            context={
                'url': test_url,
                'title': test_title,
                'suggestion_type': 'category'
            },
            original_suggestion='Development Tools',
            user_action='accepted',
            confidence_before=0.8
        )
        print(f"Tracked positive feedback: {feedback_id}")
        
        # Simulate user rejecting a tag suggestion
        negative_feedback_id = adaptive_ai.track_user_feedback(
            FeedbackType.SUGGESTION_REJECTED,
            context={
                'url': 'https://example.com/random',
                'title': 'Random Article',
                'suggestion_type': 'tag'
            },
            original_suggestion='development',
            user_action='rejected',
            confidence_before=0.6
        )
        print(f"Tracked negative feedback: {negative_feedback_id}")
        
        # Phase 4: Personalized Suggestions
        print("\n4. Personalized Suggestions Phase")
        print("-" * 40)
        
        base_suggestions = [
            ('Development Tools', 0.7),
            ('Gaming', 0.4),
            ('Research', 0.5)
        ]
        
        personalized = adaptive_ai.get_personalized_suggestions(
            'category',
            context={'url': 'https://github.com/python/cpython'},
            base_suggestions=base_suggestions
        )
        
        print("Base vs Personalized Suggestions:")
        for i, ((base_cat, base_conf), (pers_cat, pers_conf)) in enumerate(zip(base_suggestions, personalized)):
            change = pers_conf - base_conf
            print(f"  {base_cat}: {base_conf:.3f} → {pers_conf:.3f} ({change:+.3f})")
        
        # Phase 5: Incremental Learning
        print("\n5. Incremental Learning Phase")
        print("-" * 40)
        
        # Add more bookmarks for incremental learning
        new_bookmarks = [
            {
                'id': '6',
                'url': 'https://stackoverflow.com/questions/python',
                'name': 'Python Programming Questions',
                'content': {'text_content': 'Programming help and solutions for Python'},
                'collection_name': 'Development Tools',
                'tags': [{'name': 'python'}, {'name': 'programming'}, {'name': 'help'}]
            },
            {
                'id': '7',
                'url': 'https://www.twitch.tv/gaming',
                'name': 'Gaming Streams on Twitch',
                'content': {'text_content': 'Live gaming streams and esports content'},
                'collection_name': 'Gaming',
                'tags': [{'name': 'streaming'}, {'name': 'esports'}, {'name': 'live'}]
            }
        ]
        
        retrain_results = learner.retrain_models_incrementally(new_bookmarks)
        print(f"Incremental retraining results:")
        print(f"  Models retrained: {retrain_results['models_retrained']}")
        print(f"  Pattern updates: {retrain_results['pattern_updates']}")
        print(f"  Accuracy improvements: {retrain_results['accuracy_improvements']}")
        
        # Phase 6: Pattern Reliability Analysis
        print("\n6. Pattern Reliability Analysis")
        print("-" * 40)
        
        reliability_analysis = learner.analyze_pattern_reliability()
        print(f"Pattern Reliability Analysis:")
        print(f"  Total patterns: {reliability_analysis['total_patterns']}")
        print(f"  Reliable patterns: {reliability_analysis['reliable_patterns']}")
        print(f"  Unreliable patterns: {reliability_analysis['unreliable_patterns']}")
        
        print(f"\nReliability Distribution:")
        for level, count in reliability_analysis['reliability_distribution'].items():
            print(f"  {level.title()}: {count}")
        
        if reliability_analysis['recommendations']:
            print(f"\nRecommendations:")
            for rec in reliability_analysis['recommendations'][:2]:
                print(f"  • {rec}")
        
        # Phase 7: Performance Metrics
        print("\n7. Performance Metrics")
        print("-" * 40)
        
        performance_metrics = learner.get_learning_performance_metrics()
        session_metrics = performance_metrics['session_metrics']
        
        print(f"Session Metrics:")
        print(f"  Total sessions: {session_metrics['total_sessions']}")
        print(f"  Avg bookmarks per session: {session_metrics['avg_bookmarks_per_session']:.1f}")
        print(f"  Avg patterns per session: {session_metrics['avg_patterns_per_session']:.1f}")
        
        print(f"\nPattern Age Distribution:")
        age_dist = performance_metrics['pattern_age_distribution']
        for age_group, count in age_dist.items():
            print(f"  {age_group.title()}: {count}")
        
        # Phase 8: Adaptive Intelligence Statistics
        print("\n8. Adaptive Intelligence Statistics")
        print("-" * 40)
        
        adaptation_stats = adaptive_ai.get_adaptation_statistics()
        print(f"Adaptation Statistics:")
        print(f"  Total preferences: {adaptation_stats['total_preferences']}")
        print(f"  Strong preferences: {adaptation_stats['strong_preferences']}")
        print(f"  Total feedback items: {adaptation_stats['total_feedback_items']}")
        
        if adaptation_stats['suggestion_performance']:
            print(f"\nSuggestion Performance:")
            for suggestion_type, performance in adaptation_stats['suggestion_performance'].items():
                print(f"  {suggestion_type.title()}:")
                print(f"    Acceptance rate: {performance['acceptance_rate']:.1%}")
                print(f"    Total interactions: {performance['total_interactions']}")
        
        # Phase 9: Parameter Optimization
        print("\n9. Parameter Optimization")
        print("-" * 40)
        
        optimization_results = learner.optimize_learning_parameters()
        print(f"Parameter Optimization Results:")
        print(f"  Parameters changed: {len(optimization_results['parameters_changed'])}")
        
        if optimization_results['parameters_changed']:
            print(f"  Changes made:")
            for change in optimization_results['parameters_changed']:
                print(f"    {change['parameter']}: {change['old_value']:.3f} → {change['new_value']:.3f}")
                print(f"      Reason: {change['reason']}")
        
        if optimization_results['recommendations']:
            print(f"  Recommendations:")
            for rec in optimization_results['recommendations']:
                print(f"    • {rec}")
        
        # Phase 10: Data Export/Import
        print("\n10. Data Export/Import")
        print("-" * 40)
        
        # Export user data
        exported_data = adaptive_ai.export_user_data()
        print(f"Exported user data:")
        print(f"  User ID: {exported_data['user_id']}")
        print(f"  Export timestamp: {exported_data['export_timestamp']}")
        print(f"  Version: {exported_data['version']}")
        print(f"  Preferences exported: {len(exported_data['user_preferences'])}")
        
        # Create new instance and import
        new_adaptive_ai = AdaptiveIntelligence(data_dir=demo_dir, user_id="imported_user")
        import_success = new_adaptive_ai.import_user_data(exported_data)
        print(f"  Import successful: {import_success}")
        
        if import_success:
            imported_stats = new_adaptive_ai.get_adaptation_statistics()
            print(f"  Imported preferences: {imported_stats['total_preferences']}")
        
        print("\n=== Demo Completed Successfully ===")
        
    finally:
        # Clean up
        shutil.rmtree(demo_dir, ignore_errors=True)
        print(f"\nCleaned up demo directory: {demo_dir}")


if __name__ == '__main__':
    demo_continuous_learning_system()