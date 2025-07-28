"""Category dictionary with domain and content patterns"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

from ..utils.logging_utils import get_logger
from ..utils.url_utils import UrlUtils
from ..utils.text_utils import TextUtils

logger = get_logger(__name__)


class CategoryDictionary:
    """Intelligent category dictionary with domain and content patterns"""
    
    def __init__(self):
        """Initialize category dictionary with comprehensive patterns"""
        self.domain_patterns = self._load_domain_patterns()
        self.url_patterns = self._load_url_patterns()
        self.content_patterns = self._load_content_patterns()
        self.learned_associations = defaultdict(list)
        self.confidence_scores = defaultdict(float)
        
        logger.info("Category dictionary initialized with comprehensive patterns")
    
    def _load_domain_patterns(self) -> Dict[str, List[str]]:
        """Load domain-to-category patterns based on your actual bookmarks"""
        return {
            "Gaming": [
                # Gaming platforms and communities
                "twitch.tv", "itch.io", "steam", "epic", "gog.com", "origin.com",
                "uplay.com", "battle.net", "minecraft.net", "roblox.com",
                
                # Gaming-specific sites from your data
                "gamebanana.com", "lutris.net", "windowsgsm.com", "paimon.moe", 
                "hoyolab.com", "stardb.gg", "fatetrigger.com",
                
                # Gaming news and communities
                "ign.com", "gamespot.com", "polygon.com", "kotaku.com", "pcgamer.com",
                "rockpapershotgun.com", "gamefaqs.com", "metacritic.com",
                
                # Gaming tools and mods
                "nexusmods.com", "moddb.com", "curseforge.com", "thunderstore.io"
            ],
            
            "Development": [
                # Code repositories and development platforms
                "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net",
                "codeberg.org", "gitea.io",
                
                # Development tools and services
                "stackoverflow.com", "stackexchange.com", "developer.mozilla.org",
                "w3schools.com", "freecodecamp.org", "codecademy.com",
                
                # Package managers and registries
                "npmjs.com", "pypi.org", "packagist.org", "nuget.org", "crates.io",
                "rubygems.org", "maven.apache.org",
                
                # Development documentation
                "docs.python.org", "nodejs.org", "reactjs.org", "vuejs.org",
                "angular.io", "django.readthedocs.io", "flask.palletsprojects.com"
            ],
            
            "Cloud & Infrastructure": [
                # Major cloud providers
                "aws.amazon.com", "console.aws.amazon.com", "cloud.google.com",
                "azure.microsoft.com", "portal.azure.com", "digitalocean.com",
                "linode.com", "vultr.com", "hetzner.com",
                
                # Infrastructure tools from your data
                "cloudflare.com", "oracle.com", "nextcloud.com", "tailscale.com",
                
                # Container and orchestration
                "docker.com", "hub.docker.com", "kubernetes.io", "rancher.com",
                "portainer.io", "traefik.io", "nginx.com", "apache.org",
                
                # Monitoring and DevOps
                "prometheus.io", "grafana.com", "elastic.co", "splunk.com",
                "datadog.com", "newrelic.com", "sentry.io"
            ],
            
            "Self-Hosting": [
                # Self-hosting resources from your data
                "awesome-selfhosted.net", "linuxserver.io", "nextcloud.com",
                "jellyfin.org", "plex.tv", "emby.media", "homeassistant.io",
                "pihole.net", "wireguard.com", "openvpn.net",
                
                # Home lab and self-hosting
                "unraid.net", "truenas.com", "proxmox.com", "opnsense.org",
                "pfsense.org", "caddy.community", "letsencrypt.org"
            ],
            
            "AI & Machine Learning": [
                # AI platforms and services
                "openai.com", "anthropic.com", "cohere.ai", "replicate.com",
                "runpod.io", "paperspace.com", "vast.ai",
                
                # ML frameworks and tools
                "tensorflow.org", "pytorch.org", "huggingface.co", "keras.io",
                "scikit-learn.org", "opencv.org", "spacy.io",
                
                # Research and papers
                "arxiv.org", "papers.withcode.com", "distill.pub", "towards.ai",
                "machinelearningmastery.com", "fast.ai",
                
                # AI development tools
                "jupyter.org", "colab.research.google.com", "kaggle.com",
                "wandb.ai", "mlflow.org", "dvc.org"
            ],
            
            "Social Media & Forums": [
                # Social platforms from your data
                "twitter.com", "x.com", "reddit.com", "discord.com", "telegram.org",
                "facebook.com", "instagram.com", "linkedin.com", "mastodon.social",
                
                # Forums and communities
                "hackernews.ycombinator.com", "lobste.rs", "dev.to", "hashnode.com",
                "medium.com", "substack.com", "ghost.org"
            ],
            
            "Entertainment & Media": [
                # Streaming and media from your data
                "youtube.com", "youtu.be", "netflix.com", "hulu.com", "disney.com",
                "amazon.com/prime", "hbo.com", "crunchyroll.com", "funimation.com",
                
                # Music and audio
                "spotify.com", "apple.com/music", "youtube.com/music", "soundcloud.com",
                "bandcamp.com", "last.fm", "discogs.com",
                
                # Anime and manga (from your data)
                "myanimelist.net", "anilist.co", "kitsu.io", "animeplanet.com"
            ],
            
            "Productivity & Tools": [
                # Productivity tools
                "notion.so", "obsidian.md", "roamresearch.com", "logseq.com",
                "todoist.com", "trello.com", "asana.com", "monday.com",
                
                # Communication and collaboration
                "slack.com", "zoom.us", "meet.google.com", "teams.microsoft.com",
                "calendly.com", "doodle.com",
                
                # Automation and workflows
                "zapier.com", "ifttt.com", "n8n.io", "make.com"
            ],
            
            "News & Information": [
                # News and information sources
                "wikipedia.org", "news.ycombinator.com", "techcrunch.com",
                "arstechnica.com", "theverge.com", "wired.com", "engadget.com",
                
                # General news
                "bbc.com", "cnn.com", "reuters.com", "ap.org", "npr.org",
                "guardian.com", "nytimes.com"
            ],
            
            "Education & Learning": [
                # Online learning platforms
                "coursera.org", "edx.org", "udemy.com", "pluralsight.com",
                "lynda.com", "skillshare.com", "masterclass.com",
                
                # Educational resources
                "khanacademy.org", "mit.edu", "stanford.edu", "harvard.edu",
                "coursera.org", "edx.org"
            ]
        }
    
    def _load_url_patterns(self) -> Dict[str, List[str]]:
        """Load URL path patterns for category detection"""
        return {
            "Gaming": [
                r"/games?/", r"/gaming/", r"/mods?/", r"/cheats?/", r"/guides?/",
                r"/walkthrough/", r"/review/", r"/trailer/", r"/gameplay/", 
                r"/esports?/", r"/tournament/", r"/leaderboard/", r"/achievement/"
            ],
            
            "Development": [
                r"/docs?/", r"/api/", r"/github/", r"/repository/", r"/package/",
                r"/library/", r"/framework/", r"/tutorial/", r"/guide/", 
                r"/reference/", r"/documentation/", r"/sdk/", r"/cli/"
            ],
            
            "AI": [
                r"/ai/", r"/ml/", r"/machine-learning/", r"/neural/", r"/deep-learning/",
                r"/nlp/", r"/computer-vision/", r"/chatbot/", r"/llm/", r"/gpt/",
                r"/model/", r"/dataset/", r"/training/"
            ],
            
            "Cloud": [
                r"/cloud/", r"/aws/", r"/azure/", r"/gcp/", r"/infrastructure/",
                r"/devops/", r"/kubernetes/", r"/docker/", r"/container/",
                r"/deployment/", r"/ci-cd/", r"/pipeline/"
            ],
            
            "Entertainment": [
                r"/watch/", r"/video/", r"/stream/", r"/movie/", r"/tv/",
                r"/anime/", r"/manga/", r"/music/", r"/podcast/", r"/playlist/"
            ]
        }
    
    def _load_content_patterns(self) -> Dict[str, List[str]]:
        """Load content keyword patterns for category detection"""
        return {
            "Gaming": [
                # General gaming terms
                "game", "gaming", "player", "multiplayer", "single-player", "rpg", 
                "fps", "mmorpg", "indie", "console", "pc gaming", "mobile gaming",
                
                # Gaming mechanics and features
                "esports", "tournament", "leaderboard", "achievement", "mod", "cheat",
                "walkthrough", "guide", "review", "gameplay", "streaming",
                
                # Specific games from your data
                "genshin impact", "paimon", "wish tracking", "interactive map",
                "achievement tracker", "game server", "game management"
            ],
            
            "Development": [
                # Programming concepts
                "code", "programming", "developer", "software", "framework", "library",
                "api", "database", "frontend", "backend", "fullstack", "devops",
                
                # Development practices
                "ci/cd", "docker", "kubernetes", "microservices", "architecture", 
                "deployment", "testing", "debugging", "version control", "git",
                
                # Technologies and languages
                "javascript", "python", "java", "react", "vue", "angular", "node",
                "django", "flask", "spring", "laravel", "rails"
            ],
            
            "AI & Machine Learning": [
                # AI/ML concepts
                "artificial intelligence", "machine learning", "deep learning", 
                "neural network", "nlp", "computer vision", "chatbot", "llm",
                
                # AI/ML techniques
                "gpt", "transformer", "model", "training", "inference", "dataset",
                "algorithm", "automation", "classification", "regression",
                
                # AI/ML tools and frameworks
                "tensorflow", "pytorch", "scikit", "keras", "opencv", "hugging face",
                "jupyter", "colab", "kaggle", "wandb"
            ],
            
            "Cloud & Infrastructure": [
                # Cloud concepts
                "cloud", "aws", "azure", "gcp", "infrastructure", "devops", "ci/cd",
                "docker", "kubernetes", "container", "microservices", "serverless",
                
                # Infrastructure tools
                "terraform", "ansible", "jenkins", "gitlab", "github actions",
                "prometheus", "grafana", "monitoring", "logging", "security"
            ],
            
            "Self-Hosting": [
                # Self-hosting concepts
                "self-hosted", "home lab", "server", "nas", "media server", "vpn",
                "reverse proxy", "ssl", "certificate", "domain", "dns",
                
                # Self-hosting software
                "nextcloud", "jellyfin", "plex", "home assistant", "pi-hole",
                "wireguard", "traefik", "nginx", "caddy", "docker compose"
            ],
            
            "Entertainment": [
                # Media types
                "video", "movie", "tv show", "series", "documentary", "anime", "manga",
                "music", "podcast", "audiobook", "streaming", "download",
                
                # Entertainment platforms
                "netflix", "youtube", "twitch", "spotify", "crunchyroll", "plex",
                "jellyfin", "kodi", "media center", "home theater"
            ]
        }
    
    def suggest_categories_for_url(self, url: str, content: str = "") -> List[Tuple[str, float]]:
        """Suggest categories for a URL with confidence scores"""
        suggestions = []
        
        try:
            domain = UrlUtils.extract_domain(url)
            path_segments = UrlUtils.extract_path_segments(url)
            url_lower = url.lower()
            content_lower = content.lower()
            
            # Check domain patterns
            for category, domains in self.domain_patterns.items():
                for domain_pattern in domains:
                    if domain_pattern in domain:
                        confidence = 0.9  # High confidence for domain matches
                        suggestions.append((category, confidence))
                        break
            
            # Check URL path patterns
            for category, patterns in self.url_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, url_lower):
                        confidence = 0.7  # Medium confidence for URL patterns
                        suggestions.append((category, confidence))
                        break
            
            # Check content patterns
            if content:
                content_words = TextUtils.extract_keywords(content_lower)
                for category, keywords in self.content_patterns.items():
                    matches = sum(1 for keyword in keywords if keyword in content_lower)
                    if matches > 0:
                        confidence = min(0.8, matches * 0.1)  # Scale confidence by matches
                        suggestions.append((category, confidence))
            
            # Remove duplicates and sort by confidence
            unique_suggestions = {}
            for category, confidence in suggestions:
                if category not in unique_suggestions or confidence > unique_suggestions[category]:
                    unique_suggestions[category] = confidence
            
            # Sort by confidence (highest first)
            sorted_suggestions = sorted(unique_suggestions.items(), key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Category suggestions for {domain}: {sorted_suggestions}")
            return sorted_suggestions[:5]  # Return top 5 suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest categories for {url}: {e}")
            return [("General", 0.1)]
    
    def learn_from_bookmark_data(self, bookmarks: List[Dict]) -> None:
        """Learn category patterns from existing bookmark data"""
        
        logger.info(f"Learning category patterns from {len(bookmarks)} bookmarks")
        
        category_associations = defaultdict(list)
        
        for bookmark in bookmarks:
            try:
                url = bookmark.get('url', '')
                collection_name = bookmark.get('collection_name', '')
                tags = [tag.get('name', '') if isinstance(tag, dict) else str(tag) 
                       for tag in bookmark.get('tags', [])]
                
                if not url or not collection_name:
                    continue
                
                domain = UrlUtils.extract_domain(url)
                if domain:
                    # Learn domain -> collection associations
                    category_associations[collection_name].append(domain)
                    
                    # Learn tag -> collection associations
                    for tag in tags:
                        if tag:
                            self.learned_associations[f"tag:{tag}"].append(collection_name)
                
            except Exception as e:
                logger.warning(f"Failed to learn from bookmark: {e}")
                continue
        
        # Calculate confidence scores for learned associations
        for category, domains in category_associations.items():
            domain_counts = defaultdict(int)
            for domain in domains:
                domain_counts[domain] += 1
            
            # Store learned domain patterns
            for domain, count in domain_counts.items():
                confidence = min(0.8, count / len(domains))  # Scale by frequency
                self.confidence_scores[f"{category}:{domain}"] = confidence
        
        logger.info(f"Learned {len(self.learned_associations)} associations with confidence scores")
    
    def get_learned_suggestions(self, url: str) -> List[Tuple[str, float]]:
        """Get category suggestions based on learned patterns"""
        
        suggestions = []
        domain = UrlUtils.extract_domain(url)
        
        # Check learned domain associations
        for key, confidence in self.confidence_scores.items():
            if ':' in key:
                category, learned_domain = key.split(':', 1)
                if learned_domain in domain:
                    suggestions.append((category, confidence))
        
        # Sort by confidence
        return sorted(suggestions, key=lambda x: x[1], reverse=True)[:3]
    
    def get_category_stats(self) -> Dict[str, Any]:
        """Get statistics about the category dictionary"""
        
        total_domains = sum(len(domains) for domains in self.domain_patterns.values())
        total_patterns = sum(len(patterns) for patterns in self.url_patterns.values())
        total_keywords = sum(len(keywords) for keywords in self.content_patterns.values())
        
        return {
            'total_categories': len(self.domain_patterns),
            'total_domain_patterns': total_domains,
            'total_url_patterns': total_patterns,
            'total_content_keywords': total_keywords,
            'learned_associations': len(self.learned_associations),
            'confidence_scores': len(self.confidence_scores),
            'categories': list(self.domain_patterns.keys())
        }
    
    def export_learned_patterns(self) -> Dict[str, Any]:
        """Export learned patterns for backup or sharing"""
        
        return {
            'learned_associations': dict(self.learned_associations),
            'confidence_scores': dict(self.confidence_scores),
            'version': '1.0',
            'total_patterns': len(self.learned_associations)
        }
    
    def import_learned_patterns(self, patterns: Dict[str, Any]) -> bool:
        """Import previously learned patterns"""
        
        try:
            if 'learned_associations' in patterns:
                self.learned_associations.update(patterns['learned_associations'])
            
            if 'confidence_scores' in patterns:
                self.confidence_scores.update(patterns['confidence_scores'])
            
            logger.info(f"Imported {len(patterns.get('learned_associations', {}))} learned patterns")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import learned patterns: {e}")
            return False