"""Tag dictionary with intelligent tag patterns and suggestions"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter

from utils.logging_utils import get_logger
from utils.text_utils import TextUtils
from utils.url_utils import UrlUtils

logger = get_logger(__name__)


class TagDictionary:
    """Intelligent tag dictionary with pattern-based suggestions"""
    
    def __init__(self):
        """Initialize tag dictionary with comprehensive patterns"""
        self.gaming_tags = self._load_gaming_tags()
        self.tech_tags = self._load_tech_tags()
        self.content_type_tags = self._load_content_type_tags()
        self.quality_tags = self._load_quality_tags()
        self.learned_tag_patterns = defaultdict(list)
        self.tag_frequency = defaultdict(int)
        self.tag_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        logger.info("Tag dictionary initialized with comprehensive patterns")
    
    def _load_gaming_tags(self) -> Dict[str, List[str]]:
        """Load gaming-specific tag patterns based on your data"""
        return {
            "platforms": [
                "PC", "Console", "Mobile", "Steam", "Epic Games", "GOG", 
                "PlayStation", "Xbox", "Nintendo", "Switch", "Android", "iOS"
            ],
            
            "genres": [
                "RPG", "FPS", "Strategy", "Puzzle", "Platformer", "Racing", 
                "Sports", "Fighting", "Horror", "Adventure", "Simulation",
                "MMORPG", "Battle Royale", "Roguelike", "Metroidvania"
            ],
            
            "features": [
                "Multiplayer", "Single Player", "Co-op", "PvP", "Open World", 
                "Indie", "Early Access", "Free to Play", "Subscription",
                "Cross Platform", "VR", "AR"
            ],
            
            "tools": [
                "Mod", "Cheat", "Guide", "Walkthrough", "Review", "News", 
                "Community", "Forum", "Wiki", "Database", "Tracker",
                "Achievement Tracking", "Statistics", "Interactive Map"
            ],
            
            "specific_games": [
                # From your actual data
                "Genshin Impact", "Wish Tracking", "Game Server", "Game Management",
                "Game Development", "Game Mods", "Gaming Tools", "Gaming Community",
                "Gaming Platform", "Indie Games", "Free Games", "Emulators"
            ]
        }
    
    def _load_tech_tags(self) -> Dict[str, List[str]]:
        """Load technology-specific tag patterns"""
        return {
            "languages": [
                "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", 
                "Rust", "PHP", "Ruby", "Swift", "Kotlin", "Dart", "Scala",
                "R", "MATLAB", "Shell", "PowerShell", "Bash"
            ],
            
            "frameworks": [
                "React", "Vue", "Angular", "Django", "Flask", "FastAPI", "Express",
                "Spring", "Laravel", "Rails", "ASP.NET", "Flutter", "React Native",
                "Electron", "Tauri", "Next.js", "Nuxt.js", "Svelte"
            ],
            
            "tools": [
                "Git", "Docker", "Kubernetes", "Jenkins", "GitHub Actions", "GitLab CI",
                "Terraform", "Ansible", "Prometheus", "Grafana", "Nginx", "Apache",
                "Redis", "Elasticsearch", "RabbitMQ", "Kafka"
            ],
            
            "platforms": [
                "AWS", "Azure", "GCP", "Heroku", "Vercel", "Netlify", "DigitalOcean",
                "Linode", "Vultr", "Cloudflare", "GitHub", "GitLab", "Bitbucket"
            ],
            
            "databases": [
                "MySQL", "PostgreSQL", "MongoDB", "Redis", "SQLite", "MariaDB",
                "Cassandra", "DynamoDB", "Firebase", "Supabase", "PlanetScale"
            ],
            
            "ai_ml": [
                # From your AI & ML interests
                "AI", "Machine Learning", "Deep Learning", "Neural Networks", "NLP",
                "Computer Vision", "TensorFlow", "PyTorch", "Scikit-learn", "Keras",
                "OpenAI", "Hugging Face", "LLM", "GPT", "Transformer", "BERT",
                "Ollama", "Local LLM", "AI Models", "AI Development", "AI Tools"
            ],
            
            "development_types": [
                "Frontend", "Backend", "Full Stack", "Mobile", "Desktop", "Web",
                "API", "Microservices", "Serverless", "DevOps", "MLOps", "DataOps"
            ]
        }
    
    def _load_content_type_tags(self) -> List[str]:
        """Load content type classification tags"""
        return [
            "Tutorial", "Documentation", "Reference", "Guide", "Tool", "Resource",
            "News", "Blog", "Article", "Video", "Podcast", "Course", "Book", 
            "Paper", "Research", "Demo", "Example", "Template", "Boilerplate",
            "Library", "Framework", "Plugin", "Extension", "Theme", "Component"
        ]
    
    def _load_quality_tags(self) -> List[str]:
        """Load quality and accessibility tags"""
        return [
            "Free", "Open Source", "Premium", "Paid", "Subscription", "Trial",
            "Beta", "Alpha", "Stable", "LTS", "Official", "Community", "Popular",
            "Trending", "New", "Updated", "Archived", "Deprecated", "Legacy",
            "Self-hosted", "Cloud", "SaaS", "On-premise"
        ]
    
    def suggest_tags_for_content(self, 
                                title: str, 
                                content: str, 
                                url: str,
                                existing_tags: List[str] = None) -> List[Tuple[str, float]]:
        """Suggest tags based on content analysis"""
        
        if existing_tags is None:
            existing_tags = []
        
        suggestions = []
        existing_tag_names = {tag.lower() for tag in existing_tags}
        
        try:
            # Combine all text for analysis
            text_content = f"{title} {content} {url}".lower()
            keywords = TextUtils.extract_keywords(text_content)
            domain = UrlUtils.extract_domain(url)
            
            # Gaming tag suggestions
            gaming_suggestions = self._suggest_gaming_tags(text_content, domain)
            suggestions.extend(gaming_suggestions)
            
            # Technology tag suggestions
            tech_suggestions = self._suggest_tech_tags(text_content, domain, keywords)
            suggestions.extend(tech_suggestions)
            
            # Content type suggestions
            content_type_suggestions = self._suggest_content_type_tags(text_content, url)
            suggestions.extend(content_type_suggestions)
            
            # Quality tag suggestions
            quality_suggestions = self._suggest_quality_tags(text_content, domain)
            suggestions.extend(quality_suggestions)
            
            # URL-based suggestions
            url_suggestions = self._suggest_url_based_tags(url, domain)
            suggestions.extend(url_suggestions)
            
            # Filter out existing tags and duplicates
            filtered_suggestions = []
            seen_tags = set()
            
            for tag, confidence in suggestions:
                tag_lower = tag.lower()
                if (tag_lower not in existing_tag_names and 
                    tag_lower not in seen_tags and
                    confidence > 0.3):  # Minimum confidence threshold
                    filtered_suggestions.append((tag, confidence))
                    seen_tags.add(tag_lower)
            
            # Sort by confidence and return top suggestions
            filtered_suggestions.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Generated {len(filtered_suggestions)} tag suggestions for {domain}")
            return filtered_suggestions[:10]  # Return top 10 suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest tags for content: {e}")
            return []
    
    def _suggest_gaming_tags(self, text_content: str, domain: str) -> List[Tuple[str, float]]:
        """Suggest gaming-specific tags"""
        suggestions = []
        
        # Check for gaming domains
        gaming_domains = ["twitch.tv", "itch.io", "steam", "gamebanana.com", "lutris.net"]
        if any(gaming_domain in domain for gaming_domain in gaming_domains):
            suggestions.append(("Gaming", 0.9))
        
        # Check for specific gaming patterns
        gaming_patterns = {
            "genshin impact": [("Genshin Impact", 0.95), ("Gaming", 0.9), ("RPG", 0.8)],
            "paimon": [("Genshin Impact", 0.9), ("Wish Tracking", 0.8)],
            "interactive map": [("Interactive Map", 0.9), ("Gaming Tools", 0.8)],
            "achievement": [("Achievement Tracking", 0.8), ("Gaming Tools", 0.7)],
            "game server": [("Game Server", 0.9), ("Gaming Tools", 0.8)],
            "multiplayer": [("Multiplayer", 0.8), ("Gaming", 0.7)],
            "indie game": [("Indie Games", 0.9), ("Gaming", 0.8)],
            "free game": [("Free Games", 0.8), ("Gaming", 0.7)],
            "game mod": [("Game Mods", 0.9), ("Gaming Tools", 0.8)],
            "esports": [("Esports", 0.9), ("Gaming", 0.8)],
            "streaming": [("Live Streaming", 0.8), ("Gaming", 0.6)]
        }
        
        for pattern, tags in gaming_patterns.items():
            if pattern in text_content:
                suggestions.extend(tags)
        
        # Check gaming genres and platforms
        for category, tags in self.gaming_tags.items():
            for tag in tags:
                if tag.lower() in text_content:
                    confidence = 0.7 if category == "specific_games" else 0.6
                    suggestions.append((tag, confidence))
        
        return suggestions
    
    def _suggest_tech_tags(self, text_content: str, domain: str, keywords: List[str]) -> List[Tuple[str, float]]:
        """Suggest technology-specific tags"""
        suggestions = []
        
        # Check for development domains
        dev_domains = ["github.com", "stackoverflow.com", "npmjs.com", "pypi.org"]
        if any(dev_domain in domain for dev_domain in dev_domains):
            suggestions.append(("Development", 0.9))
        
        # Check for AI/ML patterns
        ai_patterns = {
            "artificial intelligence": [("AI", 0.95), ("Machine Learning", 0.8)],
            "machine learning": [("Machine Learning", 0.95), ("AI", 0.9)],
            "deep learning": [("Deep Learning", 0.95), ("AI", 0.9)],
            "neural network": [("Neural Networks", 0.9), ("AI", 0.8)],
            "openai": [("OpenAI", 0.9), ("AI", 0.8)],
            "hugging face": [("Hugging Face", 0.9), ("AI Models", 0.8)],
            "ollama": [("Ollama", 0.9), ("Local LLM", 0.8)],
            "chatbot": [("Chatbot", 0.8), ("AI", 0.7)],
            "llm": [("LLM", 0.9), ("AI", 0.8)]
        }
        
        for pattern, tags in ai_patterns.items():
            if pattern in text_content:
                suggestions.extend(tags)
        
        # Check technology categories
        for category, tags in self.tech_tags.items():
            for tag in tags:
                if tag.lower() in text_content:
                    confidence = 0.8 if category == "ai_ml" else 0.7
                    suggestions.append((tag, confidence))
        
        # Special handling for GitHub repositories
        if "github.com" in domain:
            suggestions.append(("GitHub", 0.9))
            if "/starred" in text_content:
                suggestions.append(("GitHub Starred", 0.8))
            else:
                suggestions.append(("GitHub Repository", 0.8))
        
        return suggestions
    
    def _suggest_content_type_tags(self, text_content: str, url: str) -> List[Tuple[str, float]]:
        """Suggest content type tags"""
        suggestions = []
        
        content_patterns = {
            "documentation": [("Documentation", 0.9)],
            "tutorial": [("Tutorial", 0.9)],
            "guide": [("Guide", 0.8)],
            "reference": [("Reference", 0.8)],
            "api": [("API", 0.8), ("Documentation", 0.6)],
            "tool": [("Tool", 0.8)],
            "library": [("Library", 0.8)],
            "framework": [("Framework", 0.8)],
            "plugin": [("Plugin", 0.8)],
            "extension": [("Extension", 0.8)],
            "template": [("Template", 0.7)],
            "example": [("Example", 0.7)],
            "demo": [("Demo", 0.7)]
        }
        
        for pattern, tags in content_patterns.items():
            if pattern in text_content:
                suggestions.extend(tags)
        
        # URL-based content type detection
        if "/docs" in url or "/documentation" in url:
            suggestions.append(("Documentation", 0.9))
        if "/api" in url:
            suggestions.append(("API", 0.8))
        if "/tutorial" in url or "/guide" in url:
            suggestions.append(("Tutorial", 0.8))
        
        return suggestions
    
    def _suggest_quality_tags(self, text_content: str, domain: str) -> List[Tuple[str, float]]:
        """Suggest quality and accessibility tags"""
        suggestions = []
        
        quality_patterns = {
            "open source": [("Open Source", 0.9)],
            "free": [("Free", 0.8)],
            "premium": [("Premium", 0.8)],
            "paid": [("Paid", 0.8)],
            "subscription": [("Subscription", 0.8)],
            "self-hosted": [("Self-hosted", 0.9)],
            "cloud": [("Cloud", 0.7)],
            "saas": [("SaaS", 0.8)],
            "beta": [("Beta", 0.7)],
            "alpha": [("Alpha", 0.7)],
            "popular": [("Popular", 0.6)],
            "trending": [("Trending", 0.6)]
        }
        
        for pattern, tags in quality_patterns.items():
            if pattern in text_content:
                suggestions.extend(tags)
        
        # Domain-based quality suggestions
        if any(domain.endswith(tld) for tld in [".org", ".edu", ".gov"]):
            suggestions.append(("Official", 0.7))
        
        return suggestions
    
    def _suggest_url_based_tags(self, url: str, domain: str) -> List[Tuple[str, float]]:
        """Suggest tags based on URL patterns"""
        suggestions = []
        
        # Platform-specific tags
        platform_tags = {
            "youtube.com": [("YouTube", 0.9), ("Video", 0.8)],
            "github.com": [("GitHub", 0.9), ("Development", 0.8)],
            "reddit.com": [("Reddit", 0.9), ("Community", 0.7)],
            "stackoverflow.com": [("Stack Overflow", 0.9), ("Development", 0.8)],
            "medium.com": [("Medium", 0.8), ("Article", 0.7)],
            "dev.to": [("Dev.to", 0.8), ("Development", 0.8)],
            "hackernews": [("Hacker News", 0.8), ("News", 0.7)],
            "twitter.com": [("Twitter", 0.9), ("Social Media", 0.7)],
            "linkedin.com": [("LinkedIn", 0.9), ("Professional", 0.7)]
        }
        
        for platform, tags in platform_tags.items():
            if platform in domain:
                suggestions.extend(tags)
                break
        
        return suggestions
    
    def learn_from_bookmark_tags(self, bookmarks: List[Dict]) -> None:
        """Learn tag patterns from existing bookmark data"""
        
        logger.info(f"Learning tag patterns from {len(bookmarks)} bookmarks")
        
        for bookmark in bookmarks:
            try:
                tags = bookmark.get('tags', [])
                url = bookmark.get('url', '')
                title = bookmark.get('name', '')
                content = bookmark.get('content', {}).get('text_content', '') or ''
                
                # Extract tag names
                tag_names = []
                for tag in tags:
                    if isinstance(tag, dict):
                        tag_name = tag.get('name', '')
                    else:
                        tag_name = str(tag)
                    
                    if tag_name:
                        tag_names.append(tag_name)
                        self.tag_frequency[tag_name] += 1
                
                # Learn tag co-occurrence patterns
                for i, tag1 in enumerate(tag_names):
                    for tag2 in tag_names[i+1:]:
                        self.tag_cooccurrence[tag1][tag2] += 1
                        self.tag_cooccurrence[tag2][tag1] += 1
                
                # Learn content-to-tag associations
                if content or title:
                    text_content = f"{title} {content}".lower()
                    keywords = TextUtils.extract_keywords(text_content)
                    
                    for keyword in keywords[:10]:  # Limit to top 10 keywords
                        for tag_name in tag_names:
                            self.learned_tag_patterns[keyword].append(tag_name)
                
            except Exception as e:
                logger.warning(f"Failed to learn from bookmark tags: {e}")
                continue
        
        logger.info(f"Learned patterns for {len(self.learned_tag_patterns)} keywords")
        logger.info(f"Tag frequency data for {len(self.tag_frequency)} tags")
    
    def get_related_tags(self, tag: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get tags that frequently co-occur with the given tag"""
        
        if tag not in self.tag_cooccurrence:
            return []
        
        related = []
        total_occurrences = self.tag_frequency.get(tag, 1)
        
        for related_tag, count in self.tag_cooccurrence[tag].items():
            confidence = count / total_occurrences
            related.append((related_tag, confidence))
        
        # Sort by confidence and return top results
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:limit]
    
    def get_tag_stats(self) -> Dict[str, Any]:
        """Get statistics about the tag dictionary"""
        
        total_gaming_tags = sum(len(tags) for tags in self.gaming_tags.values())
        total_tech_tags = sum(len(tags) for tags in self.tech_tags.values())
        
        return {
            'gaming_categories': len(self.gaming_tags),
            'tech_categories': len(self.tech_tags),
            'total_gaming_tags': total_gaming_tags,
            'total_tech_tags': total_tech_tags,
            'content_type_tags': len(self.content_type_tags),
            'quality_tags': len(self.quality_tags),
            'learned_patterns': len(self.learned_tag_patterns),
            'tag_frequency_data': len(self.tag_frequency),
            'most_common_tags': dict(Counter(self.tag_frequency).most_common(10))
        }