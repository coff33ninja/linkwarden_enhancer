"""Domain classification system for intelligent categorization"""

import re
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict

from ..utils.logging_utils import get_logger
from ..utils.url_utils import UrlUtils

logger = get_logger(__name__)


class DomainClassifier:
    """Classify domains into categories based on patterns and learned data"""
    
    def __init__(self):
        """Initialize domain classifier with comprehensive patterns"""
        self.gaming_domains = self._load_gaming_domains()
        self.dev_domains = self._load_dev_domains()
        self.ai_domains = self._load_ai_domains()
        self.cloud_domains = self._load_cloud_domains()
        self.social_domains = self._load_social_domains()
        self.entertainment_domains = self._load_entertainment_domains()
        self.productivity_domains = self._load_productivity_domains()
        self.news_domains = self._load_news_domains()
        
        # Learned domain classifications
        self.learned_classifications = defaultdict(list)
        self.domain_confidence = defaultdict(float)
        
        logger.info("Domain classifier initialized with comprehensive patterns")
    
    def _load_gaming_domains(self) -> Set[str]:
        """Load gaming-related domains based on your actual data"""
        return {
            # Gaming platforms
            'twitch.tv', 'itch.io', 'steam', 'steamcommunity.com', 'steampowered.com',
            'epic', 'epicgames.com', 'gog.com', 'origin.com', 'uplay.com', 'battle.net',
            
            # Gaming-specific sites from your data
            'gamebanana.com', 'lutris.net', 'windowsgsm.com', 'paimon.moe', 
            'hoyolab.com', 'stardb.gg', 'fatetrigger.com',
            
            # Gaming news and communities
            'ign.com', 'gamespot.com', 'polygon.com', 'kotaku.com', 'pcgamer.com',
            'rockpapershotgun.com', 'gamefaqs.com', 'metacritic.com', 'gamedev.net',
            
            # Gaming tools and mods
            'nexusmods.com', 'moddb.com', 'curseforge.com', 'thunderstore.io',
            'modrinth.com', 'planetminecraft.com',
            
            # Game-specific communities
            'reddit.com/r/gaming', 'reddit.com/r/games', 'discord.gg',
            'fandom.com', 'wikia.com', 'gamepedia.com'
        }
    
    def _load_dev_domains(self) -> Set[str]:
        """Load development-related domains"""
        return {
            # Code repositories
            'github.com', 'gitlab.com', 'bitbucket.org', 'sourceforge.net',
            'codeberg.org', 'gitea.io', 'gitee.com',
            
            # Development platforms and tools
            'stackoverflow.com', 'stackexchange.com', 'serverfault.com',
            'superuser.com', 'askubuntu.com',
            
            # Documentation and learning
            'developer.mozilla.org', 'w3schools.com', 'freecodecamp.org',
            'codecademy.com', 'pluralsight.com', 'udemy.com',
            
            # Package managers and registries
            'npmjs.com', 'pypi.org', 'packagist.org', 'nuget.org', 'crates.io',
            'rubygems.org', 'maven.apache.org', 'cocoapods.org',
            
            # Development services
            'replit.com', 'codepen.io', 'jsfiddle.net', 'codesandbox.io',
            'glitch.com', 'observable.com',
            
            # API and documentation
            'postman.com', 'insomnia.rest', 'swagger.io', 'openapi.org',
            'readme.io', 'gitbook.com', 'notion.so'
        }
    
    def _load_ai_domains(self) -> Set[str]:
        """Load AI and machine learning domains"""
        return {
            # AI platforms and services
            'openai.com', 'anthropic.com', 'cohere.ai', 'replicate.com',
            'runpod.io', 'paperspace.com', 'vast.ai', 'lambda.cloud',
            
            # ML frameworks and tools
            'tensorflow.org', 'pytorch.org', 'huggingface.co', 'keras.io',
            'scikit-learn.org', 'opencv.org', 'spacy.io', 'nltk.org',
            
            # Research and papers
            'arxiv.org', 'papers.withcode.com', 'distill.pub', 'towards.ai',
            'machinelearningmastery.com', 'fast.ai', 'deeplearning.ai',
            
            # AI development tools
            'jupyter.org', 'colab.research.google.com', 'kaggle.com',
            'wandb.ai', 'mlflow.org', 'dvc.org', 'neptune.ai',
            
            # AI news and communities
            'ai.googleblog.com', 'openai.com/blog', 'deepmind.com',
            'research.facebook.com', 'ai.meta.com'
        }
    
    def _load_cloud_domains(self) -> Set[str]:
        """Load cloud and infrastructure domains"""
        return {
            # Major cloud providers
            'aws.amazon.com', 'console.aws.amazon.com', 'cloud.google.com',
            'azure.microsoft.com', 'portal.azure.com', 'digitalocean.com',
            'linode.com', 'vultr.com', 'hetzner.com', 'ovh.com',
            
            # Infrastructure tools from your data
            'cloudflare.com', 'oracle.com', 'nextcloud.com', 'tailscale.com',
            
            # Container and orchestration
            'docker.com', 'hub.docker.com', 'kubernetes.io', 'rancher.com',
            'portainer.io', 'traefik.io', 'nginx.com', 'apache.org',
            
            # Monitoring and DevOps
            'prometheus.io', 'grafana.com', 'elastic.co', 'splunk.com',
            'datadog.com', 'newrelic.com', 'sentry.io', 'rollbar.com',
            
            # CI/CD and automation
            'jenkins.io', 'circleci.com', 'travis-ci.org', 'github.com/actions',
            'gitlab.com/ci', 'drone.io', 'buildkite.com'
        }
    
    def _load_social_domains(self) -> Set[str]:
        """Load social media and community domains"""
        return {
            # Major social platforms
            'twitter.com', 'x.com', 'facebook.com', 'instagram.com',
            'linkedin.com', 'tiktok.com', 'snapchat.com', 'pinterest.com',
            
            # Community platforms from your data
            'reddit.com', 'discord.com', 'telegram.org', 'slack.com',
            'mastodon.social', 'mastodon.world', 'fosstodon.org',
            
            # Forums and discussion
            'hackernews.ycombinator.com', 'lobste.rs', 'dev.to', 'hashnode.com',
            'medium.com', 'substack.com', 'ghost.org', 'wordpress.com',
            
            # Professional networks
            'behance.net', 'dribbble.com', 'deviantart.com', 'artstation.com'
        }
    
    def _load_entertainment_domains(self) -> Set[str]:
        """Load entertainment and media domains"""
        return {
            # Video streaming
            'youtube.com', 'youtu.be', 'netflix.com', 'hulu.com', 'disney.com',
            'amazon.com/prime', 'hbo.com', 'paramount.com', 'peacocktv.com',
            
            # Anime and manga from your data
            'crunchyroll.com', 'funimation.com', 'myanimelist.net',
            'anilist.co', 'kitsu.io', 'animeplanet.com',
            
            # Music and audio
            'spotify.com', 'apple.com/music', 'youtube.com/music', 'soundcloud.com',
            'bandcamp.com', 'last.fm', 'discogs.com', 'pandora.com',
            
            # Podcasts
            'anchor.fm', 'buzzsprout.com', 'libsyn.com', 'spreaker.com'
        }
    
    def _load_productivity_domains(self) -> Set[str]:
        """Load productivity and tool domains"""
        return {
            # Note-taking and knowledge management
            'notion.so', 'obsidian.md', 'roamresearch.com', 'logseq.com',
            'evernote.com', 'onenote.com', 'bear.app', 'craft.do',
            
            # Task and project management
            'todoist.com', 'trello.com', 'asana.com', 'monday.com',
            'clickup.com', 'airtable.com', 'basecamp.com',
            
            # Communication and collaboration
            'zoom.us', 'meet.google.com', 'teams.microsoft.com',
            'calendly.com', 'doodle.com', 'when2meet.com',
            
            # Automation and workflows
            'zapier.com', 'ifttt.com', 'n8n.io', 'make.com', 'automate.io'
        }
    
    def _load_news_domains(self) -> Set[str]:
        """Load news and information domains"""
        return {
            # Tech news
            'techcrunch.com', 'arstechnica.com', 'theverge.com', 'wired.com',
            'engadget.com', 'gizmodo.com', 'mashable.com', 'recode.net',
            
            # General news
            'bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org',
            'guardian.com', 'nytimes.com', 'washingtonpost.com',
            
            # Information and reference
            'wikipedia.org', 'wikimedia.org', 'britannica.com', 'dictionary.com'
        }
    
    def classify_domain(self, url: str) -> List[Tuple[str, float]]:
        """Classify domain into categories with confidence scores"""
        
        try:
            domain = UrlUtils.extract_domain(url)
            if not domain:
                return [("General", 0.1)]
            
            classifications = []
            
            # Check against predefined domain sets
            domain_categories = [
                ("Gaming", self.gaming_domains),
                ("Development", self.dev_domains),
                ("AI & Machine Learning", self.ai_domains),
                ("Cloud & Infrastructure", self.cloud_domains),
                ("Social Media", self.social_domains),
                ("Entertainment", self.entertainment_domains),
                ("Productivity", self.productivity_domains),
                ("News & Information", self.news_domains)
            ]
            
            for category, domain_set in domain_categories:
                confidence = self._calculate_domain_confidence(domain, domain_set)
                if confidence > 0:
                    classifications.append((category, confidence))
            
            # Check learned classifications
            learned_classifications = self._get_learned_classifications(domain)
            classifications.extend(learned_classifications)
            
            # If no specific classification found, try pattern matching
            if not classifications:
                pattern_classifications = self._classify_by_patterns(domain, url)
                classifications.extend(pattern_classifications)
            
            # Sort by confidence and remove duplicates
            unique_classifications = {}
            for category, confidence in classifications:
                if category not in unique_classifications or confidence > unique_classifications[category]:
                    unique_classifications[category] = confidence
            
            sorted_classifications = sorted(unique_classifications.items(), key=lambda x: x[1], reverse=True)
            
            # Return top classifications or default
            if sorted_classifications:
                return sorted_classifications[:3]
            else:
                return [("General", 0.1)]
                
        except Exception as e:
            logger.error(f"Failed to classify domain {url}: {e}")
            return [("General", 0.1)]
    
    def _calculate_domain_confidence(self, domain: str, domain_set: Set[str]) -> float:
        """Calculate confidence score for domain classification"""
        
        # Exact match
        if domain in domain_set:
            return 0.95
        
        # Subdomain match
        for known_domain in domain_set:
            if domain.endswith(f".{known_domain}") or known_domain.endswith(f".{domain}"):
                return 0.85
        
        # Partial match
        for known_domain in domain_set:
            if known_domain in domain or domain in known_domain:
                return 0.7
        
        return 0.0
    
    def _get_learned_classifications(self, domain: str) -> List[Tuple[str, float]]:
        """Get classifications based on learned patterns"""
        
        classifications = []
        
        if domain in self.learned_classifications:
            for category in self.learned_classifications[domain]:
                confidence = self.domain_confidence.get(f"{domain}:{category}", 0.6)
                classifications.append((category, confidence))
        
        return classifications
    
    def _classify_by_patterns(self, domain: str, url: str) -> List[Tuple[str, float]]:
        """Classify domain using pattern matching"""
        
        classifications = []
        
        # Pattern-based classification
        patterns = {
            "Development": [
                r'(api|dev|developer|code|git|repo|docs|documentation)',
                r'(sdk|cli|tool|lib|framework|package)'
            ],
            "AI": [
                r'(ai|ml|machinelearning|neural|deep|nlp|gpt|llm)',
                r'(model|dataset|training|inference)'
            ],
            "Gaming": [
                r'(game|gaming|play|player|esports|mod|cheat)',
                r'(steam|epic|gog|console|arcade)'
            ],
            "Cloud": [
                r'(cloud|aws|azure|gcp|server|host|deploy)',
                r'(docker|kubernetes|container|devops)'
            ],
            "Social": [
                r'(social|community|forum|chat|message|connect)',
                r'(share|follow|friend|network|group)'
            ]
        }
        
        domain_lower = domain.lower()
        url_lower = url.lower()
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, domain_lower) or re.search(pattern, url_lower):
                    classifications.append((category, 0.6))
                    break
        
        return classifications
    
    def learn_from_classifications(self, domain: str, category: str, confidence: float = 0.8) -> None:
        """Learn new domain classifications"""
        
        try:
            self.learned_classifications[domain].append(category)
            self.domain_confidence[f"{domain}:{category}"] = confidence
            
            logger.debug(f"Learned classification: {domain} -> {category} (confidence: {confidence})")
            
        except Exception as e:
            logger.warning(f"Failed to learn classification for {domain}: {e}")
    
    def get_domain_stats(self) -> Dict[str, Any]:
        """Get statistics about domain classifications"""
        
        total_predefined = (
            len(self.gaming_domains) + len(self.dev_domains) + len(self.ai_domains) +
            len(self.cloud_domains) + len(self.social_domains) + len(self.entertainment_domains) +
            len(self.productivity_domains) + len(self.news_domains)
        )
        
        return {
            'predefined_domains': {
                'gaming': len(self.gaming_domains),
                'development': len(self.dev_domains),
                'ai_ml': len(self.ai_domains),
                'cloud': len(self.cloud_domains),
                'social': len(self.social_domains),
                'entertainment': len(self.entertainment_domains),
                'productivity': len(self.productivity_domains),
                'news': len(self.news_domains),
                'total': total_predefined
            },
            'learned_classifications': len(self.learned_classifications),
            'confidence_scores': len(self.domain_confidence),
            'categories': [
                "Gaming", "Development", "AI & Machine Learning", "Cloud & Infrastructure",
                "Social Media", "Entertainment", "Productivity", "News & Information"
            ]
        }
    
    def export_learned_domains(self) -> Dict[str, Any]:
        """Export learned domain classifications"""
        
        return {
            'learned_classifications': dict(self.learned_classifications),
            'domain_confidence': dict(self.domain_confidence),
            'version': '1.0',
            'total_learned': len(self.learned_classifications)
        }
    
    def import_learned_domains(self, data: Dict[str, Any]) -> bool:
        """Import learned domain classifications"""
        
        try:
            if 'learned_classifications' in data:
                for domain, categories in data['learned_classifications'].items():
                    self.learned_classifications[domain].extend(categories)
            
            if 'domain_confidence' in data:
                self.domain_confidence.update(data['domain_confidence'])
            
            logger.info(f"Imported {len(data.get('learned_classifications', {}))} learned domain classifications")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import learned domains: {e}")
            return False