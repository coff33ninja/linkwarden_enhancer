#!/usr/bin/env python3
"""
Demo script showing how the specialized analyzers work with different types of content.
This demonstrates the functionality implemented in task 6.
"""

from linkwarden_enhancer.ai.specialized_analyzers import SpecializedAnalysisEngine


def demo_specialized_analysis():
    """Demonstrate specialized analysis on various content types"""
    
    engine = SpecializedAnalysisEngine()
    
    # Test cases for different content types
    test_cases = [
        {
            'name': 'Genshin Impact Content',
            'url': 'https://paimon.moe/calculator',
            'title': 'Genshin Impact Character Build Calculator',
            'content': 'Calculate optimal artifacts and weapons for Hu Tao, Ganyu, and other characters in Genshin Impact. Features constellation calculator and damage optimization.'
        },
        {
            'name': 'GitHub Repository',
            'url': 'https://github.com/microsoft/vscode',
            'title': 'Visual Studio Code - Microsoft',
            'content': 'TypeScript-based code editor with extensions, debugging support, and integrated terminal. Popular development tool for web development.'
        },
        {
            'name': 'Self-Hosting Tool',
            'url': 'https://linuxserver.io/docker-nextcloud',
            'title': 'Nextcloud Docker Container',
            'content': 'Self-hosted file storage and collaboration platform. Deploy with Docker and Kubernetes for your homelab setup.'
        },
        {
            'name': 'Research Paper',
            'url': 'https://arxiv.org/abs/2301.12345',
            'title': 'Deep Learning for Computer Vision Applications',
            'content': 'Research paper on neural networks and machine learning algorithms for computer vision tasks. Published in 2024.'
        },
        {
            'name': 'Educational Content',
            'url': 'https://coursera.org/learn/machine-learning',
            'title': 'Machine Learning Course - Stanford',
            'content': 'Online course tutorial about artificial intelligence, algorithms, and data science. Learn Python programming for ML.'
        },
        {
            'name': 'Hobby Content',
            'url': 'https://example.com/cooking-tutorial',
            'title': 'Best Chocolate Cake Recipe Tutorial',
            'content': 'Tutorial on how to bake a delicious chocolate cake with cooking tips and recipe instructions for beginners.'
        }
    ]
    
    print("=== Specialized Content Analysis Demo ===\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print(f"   URL: {test_case['url']}")
        print(f"   Title: {test_case['title']}")
        print()
        
        # Get the best analysis result
        result = engine.get_best_analysis(test_case['url'], test_case['title'], test_case['content'])
        
        if result:
            print(f"   Domain: {result.domain}")
            print(f"   Content Type: {result.content_type}")
            print(f"   Confidence: {result.confidence_score:.2f}")
            print(f"   Specialized Tags: {', '.join(result.specialized_tags[:5])}...")  # Show first 5 tags
            
            # Show key metadata
            key_metadata = {}
            for key, value in result.metadata.items():
                if key in ['game', 'platform', 'repo_owner', 'cloud_provider', 'publication_type', 'primary_interest']:
                    key_metadata[key] = value
            
            if key_metadata:
                print(f"   Key Metadata: {key_metadata}")
            
            # Show suggestions
            if result.suggestions:
                print(f"   Suggestions: {result.suggestions[0]}")  # Show first suggestion
        else:
            print("   No specialized analysis available")
        
        print("-" * 60)
        print()
    
    # Demonstrate combined analysis
    print("=== Combined Analysis Example ===")
    test_url = "https://github.com/unity/ml-agents"
    test_title = "Unity ML-Agents - Machine Learning for Games"
    test_content = "Unity game development with machine learning and AI research for training intelligent agents"
    
    print(f"URL: {test_url}")
    print(f"Title: {test_title}")
    print()
    
    # Get all analysis results
    all_results = engine.analyze_content(test_url, test_title, test_content)
    print(f"Number of analyzers that matched: {len(all_results)}")
    
    for result in all_results:
        print(f"- {result.domain}: {result.content_type} (confidence: {result.confidence_score:.2f})")
    
    # Get combined tags and metadata
    all_tags = engine.get_all_specialized_tags(test_url, test_title, test_content)
    combined_metadata = engine.get_combined_metadata(test_url, test_title, test_content)
    
    print(f"\nCombined Tags: {', '.join(all_tags[:10])}...")  # Show first 10 tags
    print(f"Combined Metadata Keys: {', '.join(combined_metadata.keys())}")


if __name__ == '__main__':
    demo_specialized_analysis()