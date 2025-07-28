"""
Tests for specialized content analyzers
"""

import unittest
from ai.specialized_analyzers import (
    GamingAnalyzer,
    DevelopmentAnalyzer,
    ResearchAnalyzer,
    SpecializedAnalysisEngine
)


class TestGamingAnalyzer(unittest.TestCase):
    """Test gaming-specific content analysis"""
    
    def setUp(self):
        self.analyzer = GamingAnalyzer()
    
    def test_genshin_impact_detection(self):
        """Test Genshin Impact content detection"""
        url = "https://paimon.moe/calculator"
        title = "Genshin Impact Character Calculator"
        content = "Calculate artifacts and builds for Hu Tao, Ganyu, and other characters in Genshin Impact"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertEqual(result.domain, "Gaming")
        self.assertIn("Genshin Impact", result.specialized_tags)
        self.assertIn("Character: Ganyu", result.specialized_tags)
        self.assertEqual(result.metadata['game'], 'Genshin Impact')
    
    def test_steam_platform_detection(self):
        """Test Steam platform detection"""
        url = "https://store.steampowered.com/app/123456/game"
        title = "Amazing RPG Game on Steam"
        content = "A fantastic role-playing game available on Steam with multiplayer features"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Steam", result.specialized_tags)
        self.assertIn("PC Gaming", result.specialized_tags)
        self.assertEqual(result.metadata['platform'], 'steam')
    
    def test_gaming_community_detection(self):
        """Test gaming community detection"""
        url = "https://reddit.com/r/gaming/post123"
        title = "Best RPG games of 2024"
        content = "Discussion about the best RPG games released this year"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Reddit", result.specialized_tags)
        self.assertIn("Gaming Community", result.specialized_tags)
        self.assertEqual(result.metadata['community_type'], 'Reddit')
    
    def test_game_development_tools(self):
        """Test game development tool detection"""
        url = "https://unity.com/tutorials"
        title = "Unity Game Development Tutorial"
        content = "Learn C# programming and Unity engine for game development"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Game Development", result.specialized_tags)
        self.assertIn("Engine: Unity", result.specialized_tags)
        self.assertEqual(result.metadata['engine'], 'unity')


class TestDevelopmentAnalyzer(unittest.TestCase):
    """Test development and self-hosting content analysis"""
    
    def setUp(self):
        self.analyzer = DevelopmentAnalyzer()
    
    def test_github_repository_detection(self):
        """Test GitHub repository analysis"""
        url = "https://github.com/microsoft/vscode"
        title = "Visual Studio Code - Microsoft"
        content = "TypeScript-based code editor with extensions and debugging support"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertEqual(result.domain, "Development")
        self.assertIn("GitHub", result.specialized_tags)
        self.assertIn("Repository", result.specialized_tags)
        self.assertEqual(result.metadata['repo_owner'], 'microsoft')
        self.assertEqual(result.metadata['repo_name'], 'vscode')
    
    def test_programming_language_detection(self):
        """Test programming language detection"""
        url = "https://docs.python.org/tutorial"
        title = "Python Tutorial"
        content = "Learn Python programming with Django framework and Flask web development"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Language: Python", result.specialized_tags)
        self.assertIn("python", result.metadata['programming_languages'])
    
    def test_cloud_platform_detection(self):
        """Test cloud platform detection"""
        url = "https://aws.amazon.com/ec2/"
        title = "Amazon EC2 Documentation"
        content = "AWS EC2 instances with S3 storage and Lambda functions"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("AWS", result.specialized_tags)
        self.assertIn("Cloud Platform", result.specialized_tags)
        self.assertEqual(result.metadata['cloud_provider'], 'AWS')
    
    def test_self_hosting_detection(self):
        """Test self-hosting content detection"""
        url = "https://linuxserver.io/docker-nextcloud"
        title = "Nextcloud Docker Container"
        content = "Self-hosted file storage with Docker and Kubernetes deployment"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Self-Hosting", result.specialized_tags)
        self.assertIn("File Storage", result.specialized_tags)
        self.assertTrue(result.metadata.get('selfhost_file_storage'))
    
    def test_documentation_detection(self):
        """Test documentation detection"""
        url = "https://docs.docker.com/get-started/"
        title = "Docker Documentation - Getting Started"
        content = "Complete guide and tutorial for Docker containerization"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Documentation", result.specialized_tags)
        # The documentation type should be detected from content patterns
        self.assertIn(result.metadata.get('documentation_type', ''), ['Documentation', 'Guide', 'Tutorial'])


class TestResearchAnalyzer(unittest.TestCase):
    """Test research and educational content analysis"""
    
    def setUp(self):
        self.analyzer = ResearchAnalyzer()
    
    def test_arxiv_paper_detection(self):
        """Test ArXiv paper detection"""
        url = "https://arxiv.org/abs/2301.12345"
        title = "Deep Learning for Computer Vision"
        content = "Research paper on neural networks and machine learning algorithms"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertEqual(result.domain, "Research & Education")
        self.assertIn("ArXiv", result.specialized_tags)
        self.assertIn("Research Paper", result.specialized_tags)
        self.assertEqual(result.metadata['arxiv_id'], '2301.12345')
    
    def test_wikipedia_detection(self):
        """Test Wikipedia content detection"""
        url = "https://en.wikipedia.org/wiki/Machine_Learning"
        title = "Machine Learning - Wikipedia"
        content = "Encyclopedia article about artificial intelligence and algorithms"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Wikipedia", result.specialized_tags)
        self.assertIn("Reference", result.specialized_tags)
        self.assertEqual(result.metadata['wikipedia_language'], 'en')
    
    def test_research_field_detection(self):
        """Test research field detection"""
        url = "https://example.com/ai-research"
        title = "Artificial Intelligence Research"
        content = "Study on machine learning, deep learning, and natural language processing"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Field: Computer Science", result.specialized_tags)
        self.assertIn("computer_science", result.metadata['research_fields'])
    
    def test_hobby_interest_detection(self):
        """Test hobby and interest detection"""
        url = "https://example.com/cooking-tutorial"
        title = "Best Chocolate Cake Recipe Tutorial"
        content = "Tutorial on how to bake a delicious chocolate cake with cooking tips and recipe instructions"
        
        # The analyzer should detect this as research/educational content due to "tutorial" keyword
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Interest: Cooking", result.specialized_tags)
        self.assertIn("cooking", result.metadata['hobby_interests'])
    
    def test_news_content_detection(self):
        """Test news content detection"""
        url = "https://techcrunch.com/2024/01/15/ai-breakthrough"
        title = "Major AI Breakthrough Announced"
        content = "Technology news about artificial intelligence advancement"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("News", result.specialized_tags)
        self.assertIn("Tech News", result.specialized_tags)
        self.assertEqual(result.metadata['news_category'], 'Technology')
    
    def test_educational_content_detection(self):
        """Test educational content detection"""
        url = "https://coursera.org/learn/machine-learning"
        title = "Machine Learning Course"
        content = "Online course tutorial about artificial intelligence and algorithms"
        
        self.assertTrue(self.analyzer.can_analyze(url, title, content))
        
        result = self.analyzer.analyze(url, title, content)
        self.assertIn("Coursera", result.specialized_tags)
        self.assertIn("Online Course", result.specialized_tags)
        self.assertEqual(result.metadata['platform'], 'Coursera')


class TestSpecializedAnalysisEngine(unittest.TestCase):
    """Test the main specialized analysis engine"""
    
    def setUp(self):
        self.engine = SpecializedAnalysisEngine()
    
    def test_multiple_analyzer_detection(self):
        """Test content that matches multiple analyzers"""
        url = "https://github.com/unity/ml-agents"
        title = "Unity ML-Agents - Machine Learning for Games"
        content = "Unity game development with machine learning and AI research"
        
        results = self.engine.analyze_content(url, title, content)
        
        # Should match both gaming and development analyzers
        self.assertGreaterEqual(len(results), 2)
        domains = [result.domain for result in results]
        self.assertIn("Gaming", domains)
        self.assertIn("Development", domains)
    
    def test_best_analysis_selection(self):
        """Test selection of best analysis result"""
        url = "https://paimon.moe/calculator"
        title = "Genshin Impact Calculator"
        content = "Character build calculator for Genshin Impact game"
        
        best_result = self.engine.get_best_analysis(url, title, content)
        
        self.assertIsNotNone(best_result)
        self.assertEqual(best_result.domain, "Gaming")
        self.assertIn("Genshin Impact", best_result.specialized_tags)
    
    def test_combined_tags_extraction(self):
        """Test extraction of all specialized tags"""
        url = "https://github.com/microsoft/vscode"
        title = "Visual Studio Code"
        content = "TypeScript code editor for software development"
        
        all_tags = self.engine.get_all_specialized_tags(url, title, content)
        
        self.assertGreater(len(all_tags), 0)
        # Should contain development-related tags
        self.assertTrue(any("GitHub" in tag or "TypeScript" in tag for tag in all_tags))
    
    def test_combined_metadata_extraction(self):
        """Test extraction of combined metadata"""
        url = "https://github.com/microsoft/vscode"
        title = "Visual Studio Code"
        content = "TypeScript code editor for software development"
        
        metadata = self.engine.get_combined_metadata(url, title, content)
        
        self.assertGreater(len(metadata), 0)
        # Should contain prefixed metadata from development analyzer
        self.assertTrue(any(key.startswith('development_') for key in metadata.keys()))


if __name__ == '__main__':
    unittest.main()