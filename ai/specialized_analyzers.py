"""
Specialized content analyzers for different domains and content types.
Provides domain-specific analysis for gaming, development, research, and other specialized content.
"""

import re
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class SpecializedAnalysisResult:
    """Result of specialized content analysis"""
    domain: str
    content_type: str
    specialized_tags: List[str]
    confidence_score: float
    metadata: Dict[str, Any]
    suggestions: List[str]

class SpecializedAnalyzer(ABC):
    """Abstract base class for specialized content analyzers"""
    
    @abstractmethod
    def can_analyze(self, url: str, title: str, content: str) -> bool:
        """Check if this analyzer can handle the given content"""
        pass
    
    @abstractmethod
    def analyze(self, url: str, title: str, content: str) -> SpecializedAnalysisResult:
        """Perform specialized analysis on the content"""
        pass

class GamingAnalyzer(SpecializedAnalyzer):
    """Specialized analyzer for gaming content"""
    
    def __init__(self):
        self.gaming_domains = {
            'twitch.tv', 'itch.io', 'steam', 'steamcommunity.com', 'steampowered.com',
            'epicgames.com', 'gog.com', 'gamebanana.com', 'lutris.net', 'windowsgsm.com',
            'paimon.moe', 'hoyolab.com', 'stardb.gg', 'genshin-impact.fandom.com',
            'reddit.com', 'discord.com', 'gamedev.net', 'unity.com', 'unrealengine.com',
            'godotengine.org', 'itch.io', 'gamejolt.com', 'indiedb.com', 'moddb.com'
        }
        
        self.genshin_patterns = {
            'domains': ['paimon.moe', 'hoyolab.com', 'stardb.gg', 'genshin-impact.fandom.com'],
            'keywords': [
                'genshin', 'impact', 'teyvat', 'primogem', 'resin', 'artifact', 'constellation',
                'gacha', 'banner', 'pity', 'spiral abyss', 'domain', 'ley line', 'commission',
                'character', 'weapon', 'element', 'pyro', 'hydro', 'anemo', 'electro', 'dendro',
                'cryo', 'geo', 'traveler', 'archon', 'fatui', 'hilichurl', 'slime'
            ],
            'characters': [
                'albedo', 'amber', 'ayaka', 'ayato', 'barbara', 'beidou', 'bennett', 'childe',
                'chongyun', 'diluc', 'diona', 'eula', 'fischl', 'ganyu', 'gorou', 'hutao',
                'itto', 'jean', 'kazuha', 'keqing', 'klee', 'kokomi', 'lisa', 'mona',
                'ningguang', 'noelle', 'qiqi', 'raiden', 'razor', 'rosaria', 'sara',
                'sayu', 'sucrose', 'tartaglia', 'thoma', 'venti', 'xiangling', 'xiao',
                'xingqiu', 'xinyan', 'yae', 'yanfei', 'zhongli', 'collei', 'tighnari',
                'nahida', 'nilou', 'candace', 'cyno', 'wanderer', 'faruzan', 'yaoyao',
                'alhaitham', 'baizhu', 'kaveh', 'kirara', 'lyney', 'lynette', 'freminet',
                'neuvillette', 'wriothesley', 'furina', 'charlotte', 'xianyun', 'gaming',
                'chiori', 'arlecchino', 'sethos', 'clorinde', 'sigewinne', 'emilie'
            ]
        }
        
        self.gaming_platforms = {
            'steam': ['steam', 'steampowered', 'steamcommunity'],
            'epic': ['epicgames', 'unrealengine'],
            'gog': ['gog.com', 'cdprojekt'],
            'itch': ['itch.io'],
            'console': ['playstation', 'xbox', 'nintendo', 'switch'],
            'mobile': ['android', 'ios', 'mobile', 'play store', 'app store']
        }
        
        self.game_genres = [
            'rpg', 'fps', 'rts', 'moba', 'mmorpg', 'battle royale', 'platformer',
            'puzzle', 'racing', 'sports', 'fighting', 'horror', 'survival',
            'sandbox', 'simulation', 'strategy', 'action', 'adventure', 'indie'
        ]
        
        self.gaming_communities = {
            'reddit': [r'/r/gaming', r'/r/gamedev', r'/r/indiegaming', r'/r/pcgaming',
                      r'/r/genshinimpact', r'/r/unity3d', r'/r/unrealengine'],
            'discord': ['discord.gg', 'discord.com/invite'],
            'forums': ['gamedev.net', 'indiedb.com', 'moddb.com', 'gamefaqs.com']
        }
        
        self.game_dev_tools = {
            'engines': ['unity', 'unreal', 'godot', 'gamemaker', 'construct', 'defold'],
            'graphics': ['blender', 'maya', 'photoshop', 'gimp', 'aseprite', 'krita'],
            'audio': ['audacity', 'fmod', 'wwise', 'reaper', 'fl studio'],
            'programming': ['c#', 'c++', 'javascript', 'python', 'lua', 'gdscript']
        }
    
    def can_analyze(self, url: str, title: str, content: str) -> bool:
        """Check if content is gaming-related"""
        domain = urlparse(url).netloc.lower()
        
        # Check domain patterns
        if any(gaming_domain in domain for gaming_domain in self.gaming_domains):
            return True
        
        # Check content for gaming keywords
        text = f"{title} {content}".lower()
        gaming_keywords = [
            'game', 'gaming', 'gamer', 'gameplay', 'multiplayer', 'single player',
            'console', 'pc gaming', 'mobile gaming', 'esports', 'tournament',
            'mod', 'modding', 'cheat', 'walkthrough', 'guide', 'review',
            'trailer', 'screenshot', 'achievement', 'leaderboard'
        ]
        
        return any(keyword in text for keyword in gaming_keywords)
    
    def analyze(self, url: str, title: str, content: str) -> SpecializedAnalysisResult:
        """Analyze gaming content and extract specialized information"""
        domain = urlparse(url).netloc.lower()
        text = f"{title} {content}".lower()
        
        specialized_tags = []
        metadata = {}
        suggestions = []
        confidence_score = 0.0
        
        # Genshin Impact specific analysis
        if self._is_genshin_content(url, text):
            genshin_analysis = self._analyze_genshin_content(url, text)
            specialized_tags.extend(genshin_analysis['tags'])
            metadata.update(genshin_analysis['metadata'])
            suggestions.extend(genshin_analysis['suggestions'])
            confidence_score = max(confidence_score, 0.9)
        
        # Gaming platform detection
        platform_analysis = self._analyze_gaming_platform(url, text)
        specialized_tags.extend(platform_analysis['tags'])
        metadata.update(platform_analysis['metadata'])
        confidence_score = max(confidence_score, platform_analysis['confidence'])
        
        # Game genre detection
        genre_analysis = self._analyze_game_genres(text)
        specialized_tags.extend(genre_analysis['tags'])
        metadata.update(genre_analysis['metadata'])
        
        # Gaming community analysis
        community_analysis = self._analyze_gaming_community(url, text)
        specialized_tags.extend(community_analysis['tags'])
        metadata.update(community_analysis['metadata'])
        
        # Game development tools analysis
        gamedev_analysis = self._analyze_gamedev_tools(url, text)
        specialized_tags.extend(gamedev_analysis['tags'])
        metadata.update(gamedev_analysis['metadata'])
        
        # Remove duplicates and sort by relevance
        specialized_tags = list(set(specialized_tags))
        
        # Generate suggestions based on analysis
        if not suggestions:
            suggestions = self._generate_gaming_suggestions(specialized_tags, metadata)
        
        return SpecializedAnalysisResult(
            domain="Gaming",
            content_type=metadata.get('content_type', 'General Gaming'),
            specialized_tags=specialized_tags,
            confidence_score=confidence_score,
            metadata=metadata,
            suggestions=suggestions
        )
    
    def _is_genshin_content(self, url: str, text: str) -> bool:
        """Check if content is Genshin Impact related"""
        domain = urlparse(url).netloc.lower()
        
        # Check Genshin-specific domains
        if any(genshin_domain in domain for genshin_domain in self.genshin_patterns['domains']):
            return True
        
        # Check for Genshin keywords
        return any(keyword in text for keyword in self.genshin_patterns['keywords'])
    
    def _analyze_genshin_content(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze Genshin Impact specific content"""
        tags = ['Genshin Impact']
        metadata = {'content_type': 'Genshin Impact', 'game': 'Genshin Impact'}
        suggestions = ['Gaming', 'RPG', 'Gacha', 'Mobile Gaming']
        
        # Character detection
        found_characters = [char for char in self.genshin_patterns['characters'] 
                          if char in text]
        if found_characters:
            tags.extend([f"Character: {char.title()}" for char in found_characters[:3]])
            metadata['characters'] = found_characters
        
        # Content type detection
        if any(word in text for word in ['build', 'artifact', 'weapon']):
            tags.append('Character Build')
            metadata['content_subtype'] = 'Character Build'
        elif any(word in text for word in ['guide', 'walkthrough', 'tutorial']):
            tags.append('Guide')
            metadata['content_subtype'] = 'Guide'
        elif any(word in text for word in ['news', 'update', 'patch']):
            tags.append('News')
            metadata['content_subtype'] = 'News'
        elif any(word in text for word in ['tier list', 'ranking']):
            tags.append('Tier List')
            metadata['content_subtype'] = 'Tier List'
        
        # Domain-specific analysis
        domain = urlparse(url).netloc.lower()
        if 'paimon.moe' in domain:
            tags.append('Paimon.moe')
            metadata['tool_type'] = 'Genshin Calculator'
        elif 'hoyolab.com' in domain:
            tags.append('HoyoLAB')
            metadata['tool_type'] = 'Official Community'
        elif 'stardb.gg' in domain:
            tags.append('StarDB')
            metadata['tool_type'] = 'Database'
        
        return {'tags': tags, 'metadata': metadata, 'suggestions': suggestions}
    
    def _analyze_gaming_platform(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze gaming platform information"""
        domain = urlparse(url).netloc.lower()
        tags = []
        metadata = {}
        confidence = 0.0
        
        for platform, patterns in self.gaming_platforms.items():
            if any(pattern in domain or pattern in text for pattern in patterns):
                tags.append(platform.title())
                metadata['platform'] = platform
                confidence = 0.8
                break
        
        # Steam specific analysis
        if 'steam' in domain:
            if '/app/' in url:
                # Extract Steam app ID
                app_id_match = re.search(r'/app/(\d+)', url)
                if app_id_match:
                    metadata['steam_app_id'] = app_id_match.group(1)
            tags.extend(['Steam', 'PC Gaming'])
            confidence = 0.9
        
        return {'tags': tags, 'metadata': metadata, 'confidence': confidence}
    
    def _analyze_game_genres(self, text: str) -> Dict[str, Any]:
        """Analyze game genres mentioned in content"""
        tags = []
        metadata = {}
        
        found_genres = [genre for genre in self.game_genres if genre in text]
        if found_genres:
            tags.extend([f"Genre: {genre.upper()}" for genre in found_genres])
            metadata['genres'] = found_genres
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_gaming_community(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze gaming community content"""
        tags = []
        metadata = {}
        
        # Reddit gaming communities
        for pattern in self.gaming_communities['reddit']:
            if pattern in url:
                tags.extend(['Reddit', 'Gaming Community'])
                metadata['community_type'] = 'Reddit'
                metadata['subreddit'] = pattern
                break
        
        # Discord servers
        if any(pattern in url for pattern in self.gaming_communities['discord']):
            tags.extend(['Discord', 'Gaming Community'])
            metadata['community_type'] = 'Discord'
        
        # Gaming forums
        domain = urlparse(url).netloc.lower()
        for forum in self.gaming_communities['forums']:
            if forum in domain:
                tags.extend(['Forum', 'Gaming Community'])
                metadata['community_type'] = 'Forum'
                break
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_gamedev_tools(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze game development tools and resources"""
        domain = urlparse(url).netloc.lower()
        tags = []
        metadata = {}
        
        # Game engines
        for engine in self.game_dev_tools['engines']:
            if engine in domain or engine in text:
                tags.extend(['Game Development', f"Engine: {engine.title()}"])
                metadata['dev_tool_type'] = 'Game Engine'
                metadata['engine'] = engine
                break
        
        # Graphics tools
        for tool in self.game_dev_tools['graphics']:
            if tool in text:
                tags.extend(['Game Development', 'Graphics'])
                metadata['dev_tool_type'] = 'Graphics Tool'
                break
        
        # Audio tools
        for tool in self.game_dev_tools['audio']:
            if tool in text:
                tags.extend(['Game Development', 'Audio'])
                metadata['dev_tool_type'] = 'Audio Tool'
                break
        
        # Programming languages
        for lang in self.game_dev_tools['programming']:
            if lang in text:
                tags.extend(['Game Development', f"Language: {lang.upper()}"])
                metadata['programming_language'] = lang
                break
        
        return {'tags': tags, 'metadata': metadata}
    
    def _generate_gaming_suggestions(self, tags: List[str], metadata: Dict[str, Any]) -> List[str]:
        """Generate collection and organization suggestions for gaming content"""
        suggestions = []
        
        if 'Genshin Impact' in tags:
            suggestions.append('Create a "Genshin Impact" collection for all related content')
        
        if any('Character:' in tag for tag in tags):
            suggestions.append('Consider organizing by character builds and guides')
        
        if 'Game Development' in tags:
            suggestions.append('Separate game development resources from gaming content')
        
        if metadata.get('platform'):
            platform = metadata['platform']
            suggestions.append(f'Group {platform} content in a dedicated collection')
        
        if 'Gaming Community' in tags:
            suggestions.append('Create a "Gaming Communities" collection for forums and social content')
        
        return suggestions
class DevelopmentAnalyzer(SpecializedAnalyzer):
    """Specialized analyzer for development and self-hosting content"""
    
    def __init__(self):
        self.dev_domains = {
            'github.com', 'gitlab.com', 'bitbucket.org', 'sourceforge.net',
            'stackoverflow.com', 'stackexchange.com', 'serverfault.com',
            'docker.com', 'hub.docker.com', 'kubernetes.io', 'k8s.io',
            'aws.amazon.com', 'cloud.google.com', 'azure.microsoft.com',
            'heroku.com', 'vercel.com', 'netlify.com', 'digitalocean.com',
            'linode.com', 'vultr.com', 'cloudflare.com', 'jsdelivr.com',
            'npmjs.com', 'pypi.org', 'packagist.org', 'rubygems.org',
            'nuget.org', 'crates.io', 'go.dev', 'pkg.go.dev'
        }
        
        self.self_hosting_domains = {
            'awesome-selfhosted.net', 'linuxserver.io', 'nextcloud.com',
            'jellyfin.org', 'plex.tv', 'emby.media', 'homeassistant.io',
            'pihole.net', 'wireguard.com', 'tailscale.com', 'traefik.io',
            'nginx.com', 'apache.org', 'caddy.community', 'portainer.io',
            'proxmox.com', 'truenas.com', 'opnsense.org', 'pfsense.org'
        }
        
        self.programming_languages = {
            'python': ['python', 'py', 'django', 'flask', 'fastapi', 'pandas', 'numpy'],
            'javascript': ['javascript', 'js', 'node', 'react', 'vue', 'angular', 'express'],
            'typescript': ['typescript', 'ts', 'angular', 'nest'],
            'java': ['java', 'spring', 'maven', 'gradle', 'hibernate'],
            'csharp': ['c#', 'csharp', '.net', 'dotnet', 'asp.net', 'blazor'],
            'cpp': ['c++', 'cpp', 'cmake', 'qt', 'boost'],
            'go': ['golang', 'go', 'gin', 'echo', 'fiber'],
            'rust': ['rust', 'cargo', 'actix', 'rocket', 'tokio'],
            'php': ['php', 'laravel', 'symfony', 'composer', 'wordpress'],
            'ruby': ['ruby', 'rails', 'sinatra', 'gem'],
            'swift': ['swift', 'ios', 'xcode', 'cocoapods'],
            'kotlin': ['kotlin', 'android', 'gradle'],
            'dart': ['dart', 'flutter'],
            'scala': ['scala', 'akka', 'play'],
            'elixir': ['elixir', 'phoenix', 'erlang'],
            'haskell': ['haskell', 'cabal', 'stack']
        }
        
        self.frameworks_and_tools = {
            'web_frameworks': ['react', 'vue', 'angular', 'svelte', 'django', 'flask', 'express', 'spring'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'cassandra'],
            'cloud_platforms': ['aws', 'azure', 'gcp', 'google cloud', 'heroku', 'vercel', 'netlify'],
            'devops_tools': ['docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions', 'terraform', 'ansible'],
            'monitoring': ['prometheus', 'grafana', 'elk', 'datadog', 'new relic', 'sentry'],
            'message_queues': ['rabbitmq', 'kafka', 'redis', 'celery', 'sidekiq']
        }
        
        self.self_hosting_categories = {
            'media_servers': ['plex', 'jellyfin', 'emby', 'kodi', 'subsonic', 'airsonic'],
            'file_storage': ['nextcloud', 'owncloud', 'seafile', 'syncthing', 'resilio'],
            'home_automation': ['home assistant', 'openhab', 'domoticz', 'homebridge'],
            'network_tools': ['pihole', 'adguard', 'wireguard', 'openvpn', 'tailscale'],
            'reverse_proxy': ['traefik', 'nginx', 'caddy', 'haproxy', 'cloudflare tunnel'],
            'containers': ['docker', 'podman', 'lxc', 'kubernetes', 'portainer', 'yacht'],
            'virtualization': ['proxmox', 'esxi', 'virtualbox', 'qemu', 'kvm'],
            'nas_solutions': ['truenas', 'openmediavault', 'unraid', 'freenas'],
            'monitoring': ['grafana', 'prometheus', 'zabbix', 'nagios', 'uptime kuma']
        }
        
        self.documentation_patterns = [
            'documentation', 'docs', 'api reference', 'tutorial', 'guide', 'readme',
            'wiki', 'manual', 'handbook', 'getting started', 'quickstart',
            'installation', 'setup', 'configuration', 'troubleshooting'
        ]
    
    def can_analyze(self, url: str, title: str, content: str) -> bool:
        """Check if content is development or self-hosting related"""
        domain = urlparse(url).netloc.lower()
        
        # Check development domains
        if any(dev_domain in domain for dev_domain in self.dev_domains):
            return True
        
        # Check self-hosting domains
        if any(self_host_domain in domain for self_host_domain in self.self_hosting_domains):
            return True
        
        # Check content for development keywords
        text = f"{title} {content}".lower()
        dev_keywords = [
            'programming', 'development', 'coding', 'software', 'api', 'framework',
            'library', 'repository', 'github', 'docker', 'kubernetes', 'devops',
            'self-hosted', 'homelab', 'server', 'infrastructure', 'deployment'
        ]
        
        return any(keyword in text for keyword in dev_keywords)
    
    def analyze(self, url: str, title: str, content: str) -> SpecializedAnalysisResult:
        """Analyze development and self-hosting content"""
        domain = urlparse(url).netloc.lower()
        text = f"{title} {content}".lower()
        
        specialized_tags = []
        metadata = {}
        suggestions = []
        confidence_score = 0.0
        
        # GitHub repository analysis
        if 'github.com' in domain:
            github_analysis = self._analyze_github_repository(url, title, content)
            specialized_tags.extend(github_analysis['tags'])
            metadata.update(github_analysis['metadata'])
            suggestions.extend(github_analysis['suggestions'])
            confidence_score = max(confidence_score, 0.9)
        
        # Programming language detection
        lang_analysis = self._analyze_programming_languages(text)
        specialized_tags.extend(lang_analysis['tags'])
        metadata.update(lang_analysis['metadata'])
        
        # Framework and tool detection
        framework_analysis = self._analyze_frameworks_and_tools(text)
        specialized_tags.extend(framework_analysis['tags'])
        metadata.update(framework_analysis['metadata'])
        
        # Cloud platform analysis
        cloud_analysis = self._analyze_cloud_platforms(url, text)
        specialized_tags.extend(cloud_analysis['tags'])
        metadata.update(cloud_analysis['metadata'])
        
        # Self-hosting analysis
        selfhost_analysis = self._analyze_self_hosting(url, text)
        specialized_tags.extend(selfhost_analysis['tags'])
        metadata.update(selfhost_analysis['metadata'])
        
        # Documentation analysis
        doc_analysis = self._analyze_documentation(url, text)
        specialized_tags.extend(doc_analysis['tags'])
        metadata.update(doc_analysis['metadata'])
        
        # Remove duplicates
        specialized_tags = list(set(specialized_tags))
        
        # Generate suggestions if none exist
        if not suggestions:
            suggestions = self._generate_development_suggestions(specialized_tags, metadata)
        
        # Determine content type
        content_type = self._determine_dev_content_type(metadata, specialized_tags)
        
        return SpecializedAnalysisResult(
            domain="Development",
            content_type=content_type,
            specialized_tags=specialized_tags,
            confidence_score=confidence_score or 0.7,
            metadata=metadata,
            suggestions=suggestions
        )
    
    def _analyze_github_repository(self, url: str, title: str, content: str) -> Dict[str, Any]:
        """Analyze GitHub repository information"""
        tags = ['GitHub', 'Repository']
        metadata = {'platform': 'GitHub'}
        suggestions = []
        
        # Extract repository information from URL
        repo_match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
        if repo_match:
            owner, repo_name = repo_match.groups()
            metadata['repo_owner'] = owner
            metadata['repo_name'] = repo_name
            
            # Check if it's a user or organization
            if owner.lower() in ['microsoft', 'google', 'facebook', 'apple', 'amazon', 'netflix']:
                tags.append('Official Repository')
                metadata['repo_type'] = 'Official'
        
        # Analyze repository content from title and description
        text = f"{title} {content}".lower()
        
        # Check for specific repository types
        if any(word in text for word in ['awesome', 'curated', 'list']):
            tags.append('Awesome List')
            metadata['repo_category'] = 'Curated List'
        elif any(word in text for word in ['template', 'boilerplate', 'starter']):
            tags.append('Template')
            metadata['repo_category'] = 'Template'
        elif any(word in text for word in ['tutorial', 'example', 'demo']):
            tags.append('Tutorial')
            metadata['repo_category'] = 'Tutorial'
        elif any(word in text for word in ['tool', 'utility', 'cli']):
            tags.append('Tool')
            metadata['repo_category'] = 'Tool'
        elif any(word in text for word in ['library', 'package', 'module']):
            tags.append('Library')
            metadata['repo_category'] = 'Library'
        elif any(word in text for word in ['framework']):
            tags.append('Framework')
            metadata['repo_category'] = 'Framework'
        
        suggestions.append('Consider organizing repositories by programming language or purpose')
        
        return {'tags': tags, 'metadata': metadata, 'suggestions': suggestions}
    
    def _analyze_programming_languages(self, text: str) -> Dict[str, Any]:
        """Analyze programming languages mentioned in content"""
        tags = []
        metadata = {}
        detected_languages = []
        
        for language, keywords in self.programming_languages.items():
            if any(keyword in text for keyword in keywords):
                detected_languages.append(language)
                tags.append(f"Language: {language.title()}")
        
        if detected_languages:
            metadata['programming_languages'] = detected_languages
            metadata['primary_language'] = detected_languages[0]  # First detected as primary
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_frameworks_and_tools(self, text: str) -> Dict[str, Any]:
        """Analyze frameworks and development tools"""
        tags = []
        metadata = {}
        
        for category, tools in self.frameworks_and_tools.items():
            detected_tools = [tool for tool in tools if tool in text]
            if detected_tools:
                category_name = category.replace('_', ' ').title()
                tags.append(category_name)
                metadata[category] = detected_tools
                
                # Add specific tool tags
                for tool in detected_tools[:3]:  # Limit to top 3
                    tags.append(f"Tool: {tool.title()}")
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_cloud_platforms(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze cloud platform and infrastructure content"""
        domain = urlparse(url).netloc.lower()
        tags = []
        metadata = {}
        
        # AWS analysis
        if 'aws.amazon.com' in domain or 'aws' in text:
            tags.extend(['AWS', 'Cloud Platform'])
            metadata['cloud_provider'] = 'AWS'
            
            # AWS service detection
            aws_services = ['ec2', 's3', 'lambda', 'rds', 'dynamodb', 'cloudformation', 'eks']
            detected_services = [service for service in aws_services if service in text]
            if detected_services:
                metadata['aws_services'] = detected_services
        
        # Google Cloud analysis
        elif 'cloud.google.com' in domain or 'gcp' in text or 'google cloud' in text:
            tags.extend(['GCP', 'Cloud Platform'])
            metadata['cloud_provider'] = 'GCP'
        
        # Azure analysis
        elif 'azure.microsoft.com' in domain or 'azure' in text:
            tags.extend(['Azure', 'Cloud Platform'])
            metadata['cloud_provider'] = 'Azure'
        
        # Other cloud platforms
        elif any(platform in domain for platform in ['heroku.com', 'vercel.com', 'netlify.com']):
            platform_name = domain.split('.')[0].title()
            tags.extend([platform_name, 'Cloud Platform'])
            metadata['cloud_provider'] = platform_name
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_self_hosting(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze self-hosting and homelab content"""
        domain = urlparse(url).netloc.lower()
        tags = []
        metadata = {}
        
        # Check for self-hosting domains
        if any(selfhost_domain in domain for selfhost_domain in self.self_hosting_domains):
            tags.append('Self-Hosting')
            metadata['content_category'] = 'Self-Hosting'
        
        # Analyze self-hosting categories
        for category, tools in self.self_hosting_categories.items():
            detected_tools = [tool for tool in tools if tool in text]
            if detected_tools:
                category_name = category.replace('_', ' ').title()
                tags.extend(['Self-Hosting', category_name])
                metadata[f'selfhost_{category}'] = detected_tools
                
                # Add specific tool tags
                for tool in detected_tools[:2]:  # Limit to top 2
                    tags.append(f"Tool: {tool.title()}")
        
        # Homelab specific detection
        if any(word in text for word in ['homelab', 'home lab', 'home server', 'nas']):
            tags.append('Homelab')
            metadata['homelab_related'] = True
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_documentation(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze documentation and tutorial content"""
        tags = []
        metadata = {}
        
        # Check for documentation patterns
        doc_type = None
        for pattern in self.documentation_patterns:
            if pattern in text:
                doc_type = pattern.title()
                break
        
        if doc_type:
            tags.extend(['Documentation', doc_type])
            metadata['documentation_type'] = doc_type
        
        # Check URL patterns for documentation
        if any(pattern in url for pattern in ['/docs/', '/documentation/', '/wiki/', '/api/']):
            tags.append('Documentation')
            metadata['is_documentation'] = True
        
        return {'tags': tags, 'metadata': metadata}
    
    def _determine_dev_content_type(self, metadata: Dict[str, Any], tags: List[str]) -> str:
        """Determine the specific type of development content"""
        if 'GitHub' in tags:
            return metadata.get('repo_category', 'Repository')
        elif 'Self-Hosting' in tags:
            return 'Self-Hosting'
        elif 'Cloud Platform' in tags:
            return 'Cloud Infrastructure'
        elif 'Documentation' in tags:
            return metadata.get('documentation_type', 'Documentation')
        elif any('Language:' in tag for tag in tags):
            return 'Programming Resource'
        else:
            return 'Development Tool'
    
    def _generate_development_suggestions(self, tags: List[str], metadata: Dict[str, Any]) -> List[str]:
        """Generate organization suggestions for development content"""
        suggestions = []
        
        if 'GitHub' in tags:
            suggestions.append('Create separate collections for repositories by language or purpose')
        
        if 'Self-Hosting' in tags:
            suggestions.append('Organize self-hosting resources by service type (media, networking, etc.)')
        
        if any('Language:' in tag for tag in tags):
            lang = metadata.get('primary_language', '').title()
            if lang:
                suggestions.append(f'Group {lang} resources in a dedicated collection')
        
        if 'Cloud Platform' in tags:
            provider = metadata.get('cloud_provider', '')
            if provider:
                suggestions.append(f'Create a {provider} collection for cloud resources')
        
        if 'Documentation' in tags:
            suggestions.append('Consider separating documentation from code repositories')
        
        return suggestions
    
class ResearchAnalyzer(SpecializedAnalyzer):
    """Specialized analyzer for research, educational content, and diverse interests"""
    
    def __init__(self):
        self.academic_domains = {
            'arxiv.org', 'scholar.google.com', 'researchgate.net', 'academia.edu',
            'jstor.org', 'pubmed.ncbi.nlm.nih.gov', 'ieee.org', 'acm.org',
            'springer.com', 'sciencedirect.com', 'nature.com', 'science.org',
            'plos.org', 'biorxiv.org', 'papers.withcode.com', 'distill.pub'
        }
        
        self.news_domains = {
            'bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org',
            'theguardian.com', 'nytimes.com', 'washingtonpost.com',
            'techcrunch.com', 'arstechnica.com', 'theverge.com', 'wired.com',
            'news.ycombinator.com', 'reddit.com', 'medium.com'
        }
        
        self.educational_domains = {
            'coursera.org', 'edx.org', 'udemy.com', 'khanacademy.org',
            'mit.edu', 'stanford.edu', 'harvard.edu', 'berkeley.edu',
            'wikipedia.org', 'wikimedia.org', 'britannica.com',
            'youtube.com', 'vimeo.com', 'ted.com'
        }
        
        self.research_fields = {
            'computer_science': [
                'artificial intelligence', 'machine learning', 'deep learning',
                'computer vision', 'natural language processing', 'robotics',
                'algorithms', 'data structures', 'software engineering',
                'cybersecurity', 'blockchain', 'quantum computing'
            ],
            'physics': [
                'quantum physics', 'particle physics', 'astrophysics',
                'condensed matter', 'thermodynamics', 'electromagnetism',
                'relativity', 'cosmology', 'nuclear physics'
            ],
            'biology': [
                'genetics', 'molecular biology', 'biochemistry', 'neuroscience',
                'ecology', 'evolution', 'microbiology', 'immunology',
                'bioinformatics', 'synthetic biology'
            ],
            'mathematics': [
                'algebra', 'calculus', 'statistics', 'probability',
                'topology', 'number theory', 'graph theory',
                'optimization', 'numerical analysis'
            ],
            'chemistry': [
                'organic chemistry', 'inorganic chemistry', 'physical chemistry',
                'analytical chemistry', 'materials science', 'catalysis'
            ],
            'psychology': [
                'cognitive psychology', 'behavioral psychology', 'neuroscience',
                'social psychology', 'developmental psychology'
            ],
            'economics': [
                'macroeconomics', 'microeconomics', 'behavioral economics',
                'game theory', 'econometrics', 'finance'
            ],
            'social_sciences': [
                'sociology', 'anthropology', 'political science',
                'international relations', 'public policy'
            ]
        }
        
        self.content_types = {
            'academic': ['paper', 'journal', 'conference', 'thesis', 'dissertation', 'preprint'],
            'educational': ['course', 'tutorial', 'lecture', 'textbook', 'guide', 'lesson'],
            'news': ['article', 'news', 'report', 'analysis', 'opinion', 'editorial'],
            'reference': ['encyclopedia', 'dictionary', 'handbook', 'manual', 'wiki'],
            'media': ['video', 'podcast', 'documentary', 'interview', 'presentation']
        }
        
        self.hobby_interests = {
            'cooking': ['recipe', 'cooking', 'baking', 'cuisine', 'food', 'chef'],
            'fitness': ['workout', 'exercise', 'fitness', 'gym', 'health', 'nutrition'],
            'travel': ['travel', 'tourism', 'destination', 'vacation', 'trip', 'culture'],
            'photography': ['photography', 'camera', 'lens', 'photo', 'image', 'visual'],
            'music': ['music', 'instrument', 'song', 'album', 'artist', 'concert'],
            'art': ['art', 'painting', 'drawing', 'sculpture', 'design', 'creative'],
            'literature': ['book', 'novel', 'poetry', 'literature', 'author', 'writing'],
            'history': ['history', 'historical', 'ancient', 'medieval', 'war', 'civilization'],
            'science': ['science', 'experiment', 'discovery', 'research', 'theory'],
            'philosophy': ['philosophy', 'ethics', 'logic', 'metaphysics', 'epistemology'],
            'diy': ['diy', 'craft', 'maker', 'build', 'project', 'handmade'],
            'gardening': ['garden', 'plant', 'flower', 'vegetable', 'grow', 'seed']
        }
        
        self.language_patterns = {
            'english': ['english', 'en', 'eng'],
            'spanish': ['spanish', 'español', 'es', 'spa'],
            'french': ['french', 'français', 'fr', 'fra'],
            'german': ['german', 'deutsch', 'de', 'ger'],
            'chinese': ['chinese', '中文', 'zh', 'chi'],
            'japanese': ['japanese', '日本語', 'ja', 'jpn'],
            'korean': ['korean', '한국어', 'ko', 'kor'],
            'russian': ['russian', 'русский', 'ru', 'rus'],
            'portuguese': ['portuguese', 'português', 'pt', 'por'],
            'italian': ['italian', 'italiano', 'it', 'ita']
        }
    
    def can_analyze(self, url: str, title: str, content: str) -> bool:
        """Check if content is research, educational, or general interest related"""
        domain = urlparse(url).netloc.lower()
        
        # Check academic/research domains
        if any(academic_domain in domain for academic_domain in self.academic_domains):
            return True
        
        # Check news domains
        if any(news_domain in domain for news_domain in self.news_domains):
            return True
        
        # Check educational domains
        if any(edu_domain in domain for edu_domain in self.educational_domains):
            return True
        
        # Check content for research/educational keywords
        text = f"{title} {content}".lower()
        research_keywords = [
            'research', 'study', 'analysis', 'paper', 'journal', 'academic',
            'education', 'learning', 'tutorial', 'course', 'lecture',
            'news', 'article', 'report', 'analysis', 'review'
        ]
        
        return any(keyword in text for keyword in research_keywords)
    
    def analyze(self, url: str, title: str, content: str) -> SpecializedAnalysisResult:
        """Analyze research, educational, and general interest content"""
        domain = urlparse(url).netloc.lower()
        text = f"{title} {content}".lower()
        
        specialized_tags = []
        metadata = {}
        suggestions = []
        confidence_score = 0.0
        
        # Academic/research analysis
        if any(academic_domain in domain for academic_domain in self.academic_domains):
            academic_analysis = self._analyze_academic_content(url, text)
            specialized_tags.extend(academic_analysis['tags'])
            metadata.update(academic_analysis['metadata'])
            suggestions.extend(academic_analysis['suggestions'])
            confidence_score = max(confidence_score, 0.9)
        
        # News analysis
        if any(news_domain in domain for news_domain in self.news_domains):
            news_analysis = self._analyze_news_content(url, text)
            specialized_tags.extend(news_analysis['tags'])
            metadata.update(news_analysis['metadata'])
            confidence_score = max(confidence_score, 0.8)
        
        # Educational content analysis
        educational_analysis = self._analyze_educational_content(url, text)
        specialized_tags.extend(educational_analysis['tags'])
        metadata.update(educational_analysis['metadata'])
        
        # Research field detection
        field_analysis = self._analyze_research_fields(text)
        specialized_tags.extend(field_analysis['tags'])
        metadata.update(field_analysis['metadata'])
        
        # Hobby and interest analysis
        hobby_analysis = self._analyze_hobby_interests(text)
        specialized_tags.extend(hobby_analysis['tags'])
        metadata.update(hobby_analysis['metadata'])
        
        # Content type analysis
        content_analysis = self._analyze_content_type(url, text)
        specialized_tags.extend(content_analysis['tags'])
        metadata.update(content_analysis['metadata'])
        
        # Language detection
        language_analysis = self._analyze_language(text)
        specialized_tags.extend(language_analysis['tags'])
        metadata.update(language_analysis['metadata'])
        
        # Remove duplicates
        specialized_tags = list(set(specialized_tags))
        
        # Generate suggestions if none exist
        if not suggestions:
            suggestions = self._generate_research_suggestions(specialized_tags, metadata)
        
        # Determine content type
        content_type = self._determine_research_content_type(metadata, specialized_tags)
        
        return SpecializedAnalysisResult(
            domain="Research & Education",
            content_type=content_type,
            specialized_tags=specialized_tags,
            confidence_score=confidence_score or 0.6,
            metadata=metadata,
            suggestions=suggestions
        )
    
    def _analyze_academic_content(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze academic and research content"""
        domain = urlparse(url).netloc.lower()
        tags = []
        metadata = {'content_category': 'Academic'}
        suggestions = []
        
        # ArXiv analysis
        if 'arxiv.org' in domain:
            tags.extend(['ArXiv', 'Preprint', 'Research Paper'])
            metadata['publication_type'] = 'Preprint'
            
            # Extract ArXiv ID
            arxiv_match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url)
            if arxiv_match:
                metadata['arxiv_id'] = arxiv_match.group(1)
        
        # PubMed analysis
        elif 'pubmed.ncbi.nlm.nih.gov' in domain:
            tags.extend(['PubMed', 'Medical Research', 'Research Paper'])
            metadata['publication_type'] = 'Medical Paper'
        
        # IEEE analysis
        elif 'ieee.org' in domain:
            tags.extend(['IEEE', 'Conference Paper', 'Research Paper'])
            metadata['publication_type'] = 'Conference Paper'
        
        # General academic content
        else:
            tags.append('Academic')
            
            # Detect publication type from content
            if any(word in text for word in ['conference', 'proceedings']):
                tags.append('Conference Paper')
                metadata['publication_type'] = 'Conference Paper'
            elif any(word in text for word in ['journal', 'article']):
                tags.append('Journal Article')
                metadata['publication_type'] = 'Journal Article'
            elif any(word in text for word in ['thesis', 'dissertation']):
                tags.append('Thesis')
                metadata['publication_type'] = 'Thesis'
        
        # Extract publication year
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            metadata['publication_year'] = year_match.group(0)
        
        suggestions.append('Organize research papers by field or publication year')
        
        return {'tags': tags, 'metadata': metadata, 'suggestions': suggestions}
    
    def _analyze_news_content(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze news and article content"""
        domain = urlparse(url).netloc.lower()
        tags = ['News']
        metadata = {'content_category': 'News'}
        
        # Tech news
        if any(tech_domain in domain for tech_domain in ['techcrunch.com', 'arstechnica.com', 'theverge.com']):
            tags.append('Tech News')
            metadata['news_category'] = 'Technology'
        
        # Hacker News
        elif 'news.ycombinator.com' in domain:
            tags.extend(['Hacker News', 'Tech Community'])
            metadata['news_category'] = 'Tech Community'
        
        # Reddit
        elif 'reddit.com' in domain:
            tags.extend(['Reddit', 'Social Media'])
            metadata['news_category'] = 'Social Media'
            
            # Extract subreddit
            subreddit_match = re.search(r'reddit\.com/r/([^/]+)', url)
            if subreddit_match:
                subreddit = subreddit_match.group(1)
                metadata['subreddit'] = subreddit
                tags.append(f"r/{subreddit}")
        
        # General news analysis
        if any(word in text for word in ['breaking', 'urgent', 'alert']):
            tags.append('Breaking News')
        elif any(word in text for word in ['analysis', 'opinion', 'editorial']):
            tags.append('Analysis')
        elif any(word in text for word in ['review', 'critique']):
            tags.append('Review')
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_educational_content(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze educational content"""
        domain = urlparse(url).netloc.lower()
        tags = []
        metadata = {}
        
        # Online course platforms
        if 'coursera.org' in domain:
            tags.extend(['Coursera', 'Online Course'])
            metadata['platform'] = 'Coursera'
        elif 'edx.org' in domain:
            tags.extend(['edX', 'Online Course'])
            metadata['platform'] = 'edX'
        elif 'udemy.com' in domain:
            tags.extend(['Udemy', 'Online Course'])
            metadata['platform'] = 'Udemy'
        elif 'khanacademy.org' in domain:
            tags.extend(['Khan Academy', 'Educational'])
            metadata['platform'] = 'Khan Academy'
        
        # Wikipedia
        elif 'wikipedia.org' in domain:
            tags.extend(['Wikipedia', 'Reference'])
            metadata['content_type'] = 'Encyclopedia'
            
            # Extract Wikipedia language
            lang_match = re.search(r'(\w+)\.wikipedia\.org', domain)
            if lang_match:
                lang_code = lang_match.group(1)
                metadata['wikipedia_language'] = lang_code
        
        # YouTube educational content
        elif 'youtube.com' in domain:
            if any(word in text for word in ['tutorial', 'lesson', 'course', 'lecture']):
                tags.extend(['YouTube', 'Video Tutorial'])
                metadata['content_type'] = 'Video Tutorial'
        
        # TED Talks
        elif 'ted.com' in domain:
            tags.extend(['TED', 'Presentation'])
            metadata['content_type'] = 'TED Talk'
        
        # General educational content detection
        if any(word in text for word in ['tutorial', 'guide', 'how to', 'learn']):
            tags.append('Tutorial')
            metadata['educational_type'] = 'Tutorial'
        elif any(word in text for word in ['course', 'lesson', 'class']):
            tags.append('Course')
            metadata['educational_type'] = 'Course'
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_research_fields(self, text: str) -> Dict[str, Any]:
        """Analyze research fields and academic disciplines"""
        tags = []
        metadata = {}
        detected_fields = []
        
        for field, keywords in self.research_fields.items():
            if any(keyword in text for keyword in keywords):
                detected_fields.append(field)
                field_name = field.replace('_', ' ').title()
                tags.append(f"Field: {field_name}")
        
        if detected_fields:
            metadata['research_fields'] = detected_fields
            metadata['primary_field'] = detected_fields[0]
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_hobby_interests(self, text: str) -> Dict[str, Any]:
        """Analyze hobby and personal interest content"""
        tags = []
        metadata = {}
        detected_interests = []
        
        for interest, keywords in self.hobby_interests.items():
            if any(keyword in text for keyword in keywords):
                detected_interests.append(interest)
                interest_name = interest.replace('_', ' ').title()
                tags.append(f"Interest: {interest_name}")
        
        if detected_interests:
            metadata['hobby_interests'] = detected_interests
            metadata['primary_interest'] = detected_interests[0]
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_content_type(self, url: str, text: str) -> Dict[str, Any]:
        """Analyze the type of content"""
        tags = []
        metadata = {}
        
        for content_category, types in self.content_types.items():
            detected_types = [content_type for content_type in types if content_type in text]
            if detected_types:
                category_name = content_category.title()
                tags.append(category_name)
                metadata[f'{content_category}_types'] = detected_types
        
        # Video content detection
        if any(video_domain in url for video_domain in ['youtube.com', 'vimeo.com', 'twitch.tv']):
            tags.append('Video Content')
            metadata['media_type'] = 'Video'
        
        # Podcast detection
        elif any(word in text for word in ['podcast', 'episode', 'audio']):
            tags.append('Podcast')
            metadata['media_type'] = 'Audio'
        
        return {'tags': tags, 'metadata': metadata}
    
    def _analyze_language(self, text: str) -> Dict[str, Any]:
        """Analyze content language"""
        tags = []
        metadata = {}
        
        # Simple language detection based on common words/patterns
        for language, patterns in self.language_patterns.items():
            if any(pattern in text for pattern in patterns):
                if language != 'english':  # Don't tag English as it's default
                    tags.append(f"Language: {language.title()}")
                    metadata['content_language'] = language
                break
        
        return {'tags': tags, 'metadata': metadata}
    
    def _determine_research_content_type(self, metadata: Dict[str, Any], tags: List[str]) -> str:
        """Determine the specific type of research/educational content"""
        if 'Academic' in tags:
            return metadata.get('publication_type', 'Academic Paper')
        elif 'News' in tags:
            return metadata.get('news_category', 'News Article')
        elif any('Course' in tag or 'Tutorial' in tag for tag in tags):
            return 'Educational Content'
        elif 'Wikipedia' in tags:
            return 'Reference Material'
        elif any('Interest:' in tag for tag in tags):
            interest = metadata.get('primary_interest', '').title()
            return f"{interest} Content" if interest else 'Hobby Content'
        elif any('Field:' in tag for tag in tags):
            field = metadata.get('primary_field', '').replace('_', ' ').title()
            return f"{field} Research" if field else 'Research Content'
        else:
            return 'General Knowledge'
    
    def _generate_research_suggestions(self, tags: List[str], metadata: Dict[str, Any]) -> List[str]:
        """Generate organization suggestions for research and educational content"""
        suggestions = []
        
        if 'Academic' in tags:
            suggestions.append('Create collections by research field or publication type')
        
        if any('Field:' in tag for tag in tags):
            field = metadata.get('primary_field', '').replace('_', ' ').title()
            if field:
                suggestions.append(f'Group {field} content in a dedicated collection')
        
        if 'News' in tags:
            suggestions.append('Separate news articles from academic content')
        
        if any('Interest:' in tag for tag in tags):
            suggestions.append('Create hobby-specific collections for personal interests')
        
        if 'Wikipedia' in tags:
            suggestions.append('Consider a "Reference Materials" collection for encyclopedic content')
        
        if any('Course' in tag or 'Tutorial' in tag for tag in tags):
            suggestions.append('Group educational content by subject or skill level')
        
        return suggestions


class SpecializedAnalysisEngine:
    """Main engine that coordinates all specialized analyzers"""
    
    def __init__(self):
        self.analyzers = [
            GamingAnalyzer(),
            DevelopmentAnalyzer(),
            ResearchAnalyzer()
        ]
    
    def analyze_content(self, url: str, title: str, content: str) -> List[SpecializedAnalysisResult]:
        """Analyze content with all applicable specialized analyzers"""
        results = []
        
        for analyzer in self.analyzers:
            if analyzer.can_analyze(url, title, content):
                try:
                    result = analyzer.analyze(url, title, content)
                    results.append(result)
                except Exception as e:
                    # Log error but continue with other analyzers
                    print(f"Error in {analyzer.__class__.__name__}: {e}")
                    continue
        
        return results
    
    def get_best_analysis(self, url: str, title: str, content: str) -> Optional[SpecializedAnalysisResult]:
        """Get the best analysis result based on confidence scores"""
        results = self.analyze_content(url, title, content)
        
        if not results:
            return None
        
        # Return the result with highest confidence score
        return max(results, key=lambda x: x.confidence_score)
    
    def get_all_specialized_tags(self, url: str, title: str, content: str) -> List[str]:
        """Get all specialized tags from all applicable analyzers"""
        results = self.analyze_content(url, title, content)
        all_tags = []
        
        for result in results:
            all_tags.extend(result.specialized_tags)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(all_tags))
    
    def get_combined_metadata(self, url: str, title: str, content: str) -> Dict[str, Any]:
        """Get combined metadata from all applicable analyzers"""
        results = self.analyze_content(url, title, content)
        combined_metadata = {}
        
        for result in results:
            # Prefix metadata keys with analyzer domain to avoid conflicts
            domain_prefix = result.domain.lower().replace(' ', '_').replace('&', 'and')
            for key, value in result.metadata.items():
                combined_metadata[f"{domain_prefix}_{key}"] = value
        
        return combined_metadata