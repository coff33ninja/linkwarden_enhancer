"""GitHub importer for starred repositories and user repositories with caching"""

import time
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

try:
    from github import Github, RateLimitExceededException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

from .base_importer import BaseImporter
from ..data_models import ImportResult, GitHubBookmark
from ..utils.logging_utils import get_logger
from ..utils.text_utils import TextUtils
from ..utils.json_handler import JsonHandler

logger = get_logger(__name__)


class GitHubImporter(BaseImporter):
    """Import GitHub starred repositories and user repositories with intelligent caching"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize GitHub importer with caching support"""
        super().__init__(config)
        
        if not GITHUB_AVAILABLE:
            raise ImportError("PyGithub is required for GitHub import. Install with: pip install PyGithub")
        
        self.github_config = config.get('github', {})
        self.token = self.github_config.get('token') or os.getenv('GITHUB_TOKEN')
        self.username = self.github_config.get('username') or os.getenv('GITHUB_USERNAME')
        self.rate_limit_requests_per_hour = self.github_config.get('rate_limit_requests_per_hour', 5000)
        
        # Cache configuration
        self.cache_config = self.github_config.get('cache', {})
        self.enable_cache = self.cache_config.get('enabled', True)
        self.cache_ttl_hours = self.cache_config.get('ttl_hours', 24)  # Cache for 24 hours by default
        self.cache_dir = Path(config.get('directories', {}).get('cache_dir', 'cache')) / 'github'
        self.force_refresh = self.cache_config.get('force_refresh', False)
        
        # Create cache directory using os module
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if not self.token:
            raise ValueError("GitHub token is required. Set GITHUB_TOKEN environment variable or configure in settings.")
        
        if not self.username:
            raise ValueError("GitHub username is required. Set GITHUB_USERNAME environment variable or configure in settings.")
        
        # Initialize GitHub client
        self.github = Github(self.token)
        self.user = None
        
        # Cache file paths
        self.starred_cache_file = self.cache_dir / f"{self.username}_starred.json"
        self.owned_cache_file = self.cache_dir / f"{self.username}_owned.json"
        self.cache_metadata_file = self.cache_dir / f"{self.username}_metadata.json"
        
        logger.info(f"GitHub importer initialized for user: {self.username}")
        if self.enable_cache:
            logger.info(f"Cache enabled: TTL={self.cache_ttl_hours}h, Dir={self.cache_dir}")
    
    def validate_config(self) -> bool:
        """Validate GitHub configuration"""
        try:
            # Test GitHub connection
            self.user = self.github.get_user(self.username)
            rate_limit = self.github.get_rate_limit()
            
            logger.info(f"GitHub connection validated. Rate limit: {rate_limit.core.remaining}/{rate_limit.core.limit}")
            
            if rate_limit.core.remaining < 100:
                self.add_warning(f"Low GitHub API rate limit remaining: {rate_limit.core.remaining}")
            
            return True
            
        except Exception as e:
            self.add_error(f"GitHub validation failed: {e}")
            return False
    
    def import_data(self, 
                   import_starred: bool = True, 
                   import_owned: bool = True,
                   max_repos: Optional[int] = None,
                   force_refresh: bool = False) -> ImportResult:
        """Import GitHub data with intelligent caching"""
        
        if not self.validate_config():
            return ImportResult(
                bookmarks=[],
                total_imported=0,
                import_source="github",
                errors=self.errors,
                warnings=self.warnings
            )
        
        all_bookmarks = []
        cache_used = False
        
        try:
            # Import starred repositories
            if import_starred:
                logger.info("Importing starred repositories...")
                starred_bookmarks, starred_from_cache = self._import_starred_repositories_cached(max_repos, force_refresh)
                all_bookmarks.extend(starred_bookmarks)
                cache_used = cache_used or starred_from_cache
                logger.info(f"Imported {len(starred_bookmarks)} starred repositories {'(from cache)' if starred_from_cache else '(from API)'}")
            
            # Import owned repositories
            if import_owned:
                logger.info("Importing owned repositories...")
                owned_bookmarks, owned_from_cache = self._import_user_repositories_cached(max_repos, force_refresh)
                all_bookmarks.extend(owned_bookmarks)
                cache_used = cache_used or owned_from_cache
                logger.info(f"Imported {len(owned_bookmarks)} owned repositories {'(from cache)' if owned_from_cache else '(from API)'}")
            
            # Update cache metadata
            if not cache_used:
                self._update_cache_metadata()
            
            return ImportResult(
                bookmarks=[bookmark.__dict__ for bookmark in all_bookmarks],
                total_imported=len(all_bookmarks),
                import_source="github",
                errors=self.errors,
                warnings=self.warnings
            )
            
        except RateLimitExceededException as e:
            self.add_error(f"GitHub rate limit exceeded: {e}")
            # Try to fall back to cache if available
            if self.enable_cache and not force_refresh:
                logger.warning("Falling back to cached data due to rate limit")
                cached_bookmarks = self._load_from_cache_fallback(import_starred, import_owned, max_repos)
                all_bookmarks.extend(cached_bookmarks)
            
            return ImportResult(
                bookmarks=[bookmark.__dict__ for bookmark in all_bookmarks],
                total_imported=len(all_bookmarks),
                import_source="github",
                errors=self.errors,
                warnings=self.warnings
            )
        except Exception as e:
            self.add_error(f"GitHub import failed: {e}")
            return ImportResult(
                bookmarks=[],
                total_imported=0,
                import_source="github",
                errors=self.errors,
                warnings=self.warnings
            )
    
    def _import_starred_repositories(self, max_repos: Optional[int] = None) -> List[GitHubBookmark]:
        """Import user's starred repositories"""
        starred_repos = []
        
        try:
            starred = self.user.get_starred()
            count = 0
            
            for repo in starred:
                if max_repos and count >= max_repos:
                    break
                
                try:
                    bookmark = self._convert_repo_to_bookmark(repo, "starred")
                    starred_repos.append(bookmark)
                    count += 1
                    
                    # Rate limiting
                    if count % 50 == 0:
                        logger.info(f"Processed {count} starred repositories...")
                        time.sleep(1)  # Brief pause to respect rate limits
                        
                except Exception as e:
                    self.add_warning(f"Failed to process starred repo {repo.full_name}: {e}")
                    continue
            
            logger.info(f"Successfully imported {len(starred_repos)} starred repositories")
            
        except Exception as e:
            self.add_error(f"Failed to import starred repositories: {e}")
        
        return starred_repos
    
    def _import_user_repositories(self, max_repos: Optional[int] = None) -> List[GitHubBookmark]:
        """Import user's own repositories"""
        user_repos = []
        
        try:
            repos = self.user.get_repos()
            count = 0
            
            for repo in repos:
                if max_repos and count >= max_repos:
                    break
                
                try:
                    bookmark = self._convert_repo_to_bookmark(repo, "owned")
                    user_repos.append(bookmark)
                    count += 1
                    
                    # Rate limiting
                    if count % 50 == 0:
                        logger.info(f"Processed {count} owned repositories...")
                        time.sleep(1)  # Brief pause to respect rate limits
                        
                except Exception as e:
                    self.add_warning(f"Failed to process owned repo {repo.full_name}: {e}")
                    continue
            
            logger.info(f"Successfully imported {len(user_repos)} owned repositories")
            
        except Exception as e:
            self.add_error(f"Failed to import owned repositories: {e}")
        
        return user_repos
    
    def _convert_repo_to_bookmark(self, repo, bookmark_type: str) -> GitHubBookmark:
        """Convert GitHub repository to bookmark format"""
        try:
            # Get repository languages (this makes an API call)
            languages = []
            try:
                languages_dict = repo.get_languages()
                languages = list(languages_dict.keys())
            except Exception as e:
                logger.debug(f"Could not get languages for {repo.full_name}: {e}")
            
            # Get topics
            topics = []
            try:
                topics = list(repo.get_topics())
            except Exception as e:
                logger.debug(f"Could not get topics for {repo.full_name}: {e}")
            
            # Generate intelligent tags
            tags = self._generate_repo_tags(repo, languages, topics, bookmark_type)
            
            # Determine appropriate collection
            collection = self._suggest_repo_collection(repo, languages, topics)
            
            # Create bookmark
            bookmark = GitHubBookmark(
                name=repo.full_name,
                url=repo.html_url,
                description=repo.description or "",
                tags=tags,
                suggested_collection=collection,
                metadata={
                    'stars': repo.stargazers_count,
                    'forks': repo.forks_count,
                    'language': repo.language,
                    'languages': languages,
                    'topics': topics,
                    'created_at': repo.created_at.isoformat() if repo.created_at else None,
                    'updated_at': repo.updated_at.isoformat() if repo.updated_at else None,
                    'bookmark_type': bookmark_type,
                    'is_fork': repo.fork,
                    'has_wiki': repo.has_wiki,
                    'has_pages': repo.has_pages,
                    'size': repo.size,
                    'default_branch': repo.default_branch,
                    'archived': repo.archived,
                    'disabled': repo.disabled,
                    'private': repo.private
                }
            )
            
            return bookmark
            
        except Exception as e:
            logger.error(f"Failed to convert repo {repo.full_name} to bookmark: {e}")
            raise
    
    def _generate_repo_tags(self, repo, languages: List[str], topics: List[str], bookmark_type: str) -> List[str]:
        """Generate intelligent tags for GitHub repository"""
        tags = []
        
        # Add bookmark type
        tags.append(f"GitHub {bookmark_type.title()}")
        
        # Add primary language
        if repo.language:
            tags.append(repo.language)
        
        # Add additional languages (limit to top 3)
        for lang in languages[:3]:
            if lang != repo.language and lang not in tags:
                tags.append(lang)
        
        # Framework/library detection based on name and description
        text_content = TextUtils.clean_text(f"{repo.name} {repo.description or ''}").lower()
        
        # JavaScript frameworks
        if any(lang in ['JavaScript', 'TypeScript'] for lang in languages):
            if any(keyword in text_content for keyword in ['react', 'reactjs']):
                tags.append('React')
            elif any(keyword in text_content for keyword in ['vue', 'vuejs']):
                tags.append('Vue')
            elif any(keyword in text_content for keyword in ['angular', 'angularjs']):
                tags.append('Angular')
            elif any(keyword in text_content for keyword in ['node', 'nodejs', 'express']):
                tags.append('Node.js')
        
        # Python frameworks
        if 'Python' in languages:
            if any(keyword in text_content for keyword in ['django']):
                tags.append('Django')
            elif any(keyword in text_content for keyword in ['flask']):
                tags.append('Flask')
            elif any(keyword in text_content for keyword in ['fastapi']):
                tags.append('FastAPI')
        
        # Add topic-based tags (limit to 5)
        for topic in topics[:5]:
            topic_formatted = topic.replace('-', ' ').title()
            if topic_formatted not in tags:
                tags.append(topic_formatted)
        
        # Add special characteristics
        if repo.stargazers_count > 1000:
            tags.append('Popular')
        if repo.stargazers_count > 10000:
            tags.append('Highly Popular')
        if repo.fork:
            tags.append('Fork')
        if repo.has_wiki:
            tags.append('Documentation')
        if repo.archived:
            tags.append('Archived')
        if repo.private:
            tags.append('Private')
        
        # Content-based tags
        if any(keyword in text_content for keyword in ['api', 'rest', 'graphql']):
            tags.append('API')
        if any(keyword in text_content for keyword in ['cli', 'command', 'terminal']):
            tags.append('CLI Tool')
        if any(keyword in text_content for keyword in ['docker', 'container']):
            tags.append('Docker')
        if any(keyword in text_content for keyword in ['ai', 'ml', 'machine learning', 'neural']):
            tags.append('AI/ML')
        
        return tags[:15]  # Limit total tags
    
    def _suggest_repo_collection(self, repo, languages: List[str], topics: List[str]) -> str:
        """Suggest appropriate collection for repository"""
        name_lower = TextUtils.clean_text(repo.name).lower()
        desc_lower = TextUtils.clean_text(repo.description or '').lower()
        text_content = f"{name_lower} {desc_lower}"
        
        # Gaming-related repositories
        if any(keyword in text_content for keyword in [
            'game', 'gaming', 'unity', 'unreal', 'godot', 'pygame', 'phaser'
        ]):
            return 'Game Development'
        
        # AI/ML repositories
        if any(keyword in text_content for keyword in [
            'ai', 'ml', 'machine-learning', 'neural', 'tensorflow', 'pytorch', 
            'scikit', 'keras', 'opencv', 'nlp', 'deep-learning'
        ]):
            return 'AI & Machine Learning'
        
        # Web development
        if any(lang in languages for lang in ['JavaScript', 'TypeScript', 'HTML', 'CSS']):
            if any(keyword in text_content for keyword in ['react', 'vue', 'angular']):
                return 'Frontend Development'
            elif any(keyword in text_content for keyword in ['api', 'server', 'backend']):
                return 'Backend Development'
            else:
                return 'Web Development'
        
        # Mobile development
        if any(lang in languages for lang in ['Swift', 'Kotlin', 'Java']) or \
           any(keyword in text_content for keyword in ['android', 'ios', 'mobile', 'flutter', 'react-native']):
            return 'Mobile Development'
        
        # Infrastructure/DevOps
        if any(keyword in text_content for keyword in [
            'docker', 'kubernetes', 'terraform', 'ansible', 'devops', 'ci/cd', 
            'jenkins', 'github-actions', 'deployment'
        ]):
            return 'Infrastructure & DevOps'
        
        # Data Science
        if any(keyword in text_content for keyword in [
            'data', 'analytics', 'visualization', 'pandas', 'numpy', 'jupyter'
        ]):
            return 'Data Science'
        
        # Security
        if any(keyword in text_content for keyword in [
            'security', 'crypto', 'encryption', 'auth', 'oauth', 'jwt'
        ]):
            return 'Security'
        
        # Tools and utilities
        if any(keyword in text_content for keyword in [
            'tool', 'utility', 'cli', 'script', 'automation', 'helper'
        ]):
            return 'Tools & Utilities'
        
        # Language-specific collections
        if repo.language:
            return f'{repo.language} Projects'
        
        # Default
        return 'Development'
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current GitHub API rate limit information"""
        try:
            rate_limit = self.github.get_rate_limit()
            return {
                'core': {
                    'limit': rate_limit.core.limit,
                    'remaining': rate_limit.core.remaining,
                    'reset': rate_limit.core.reset.isoformat() if rate_limit.core.reset else None
                },
                'search': {
                    'limit': rate_limit.search.limit,
                    'remaining': rate_limit.search.remaining,
                    'reset': rate_limit.search.reset.isoformat() if rate_limit.search.reset else None
                }
            }
        except Exception as e:
            logger.error(f"Failed to get rate limit info: {e}")
            return {}
    
    # ============================================================================
    # CACHING METHODS
    # ============================================================================
    
    def _import_starred_repositories_cached(self, max_repos: Optional[int] = None, force_refresh: bool = False) -> tuple[List[GitHubBookmark], bool]:
        """Import starred repositories with caching support"""
        if self.enable_cache and not force_refresh and not self.force_refresh:
            cached_data = self._load_starred_from_cache()
            if cached_data:
                limited_data = cached_data[:max_repos] if max_repos else cached_data
                bookmarks = [self._dict_to_bookmark(bookmark_dict) for bookmark_dict in limited_data]
                return bookmarks, True
        
        # Fetch from API
        starred_repos = self._import_starred_repositories(max_repos)
        
        # Save to cache
        if self.enable_cache and starred_repos:
            self._save_starred_to_cache([bookmark.__dict__ for bookmark in starred_repos])
        
        return starred_repos, False
    
    def _import_user_repositories_cached(self, max_repos: Optional[int] = None, force_refresh: bool = False) -> tuple[List[GitHubBookmark], bool]:
        """Import user repositories with caching support"""
        if self.enable_cache and not force_refresh and not self.force_refresh:
            cached_data = self._load_owned_from_cache()
            if cached_data:
                limited_data = cached_data[:max_repos] if max_repos else cached_data
                bookmarks = [self._dict_to_bookmark(bookmark_dict) for bookmark_dict in limited_data]
                return bookmarks, True
        
        # Fetch from API
        owned_repos = self._import_user_repositories(max_repos)
        
        # Save to cache
        if self.enable_cache and owned_repos:
            self._save_owned_to_cache([bookmark.__dict__ for bookmark in owned_repos])
        
        return owned_repos, False
    
    def _load_starred_from_cache(self) -> Optional[List[Dict[str, Any]]]:
        """Load starred repositories from cache if valid"""
        try:
            if not self.starred_cache_file.exists():
                logger.debug("No starred repositories cache found")
                return None
            
            if not self._is_cache_valid(self.starred_cache_file):
                logger.info("Starred repositories cache expired")
                return None
            
            # Use json module for loading cache data
            with open(self.starred_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} starred repositories from cache")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load starred repositories from cache: {e}")
            return None
    
    def _load_owned_from_cache(self) -> Optional[List[Dict[str, Any]]]:
        """Load owned repositories from cache if valid"""
        try:
            if not self.owned_cache_file.exists():
                logger.debug("No owned repositories cache found")
                return None
            
            if not self._is_cache_valid(self.owned_cache_file):
                logger.info("Owned repositories cache expired")
                return None
            
            # Use json module for loading cache data
            with open(self.owned_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} owned repositories from cache")
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load owned repositories from cache: {e}")
            return None
    
    def _save_starred_to_cache(self, bookmarks: List[Dict[str, Any]]) -> None:
        """Save starred repositories to cache"""
        try:
            # Use json module for saving cache data
            with open(self.starred_cache_file, 'w', encoding='utf-8') as f:
                json.dump(bookmarks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(bookmarks)} starred repositories to cache")
        except Exception as e:
            logger.warning(f"Failed to save starred repositories to cache: {e}")
    
    def _save_owned_to_cache(self, bookmarks: List[Dict[str, Any]]) -> None:
        """Save owned repositories to cache"""
        try:
            # Use json module for saving cache data
            with open(self.owned_cache_file, 'w', encoding='utf-8') as f:
                json.dump(bookmarks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(bookmarks)} owned repositories to cache")
        except Exception as e:
            logger.warning(f"Failed to save owned repositories to cache: {e}")
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file is still valid based on TTL"""
        try:
            if not cache_file.exists():
                return False
            
            # Check file modification time
            file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            cache_expiry = file_mtime + timedelta(hours=self.cache_ttl_hours)
            
            is_valid = datetime.now() < cache_expiry
            
            if not is_valid:
                logger.debug(f"Cache file {cache_file.name} expired (created: {file_mtime}, expires: {cache_expiry})")
            
            return is_valid
            
        except Exception as e:
            logger.warning(f"Failed to check cache validity for {cache_file}: {e}")
            return False
    
    def _update_cache_metadata(self) -> None:
        """Update cache metadata with current timestamp and rate limit info"""
        try:
            metadata = {
                'username': self.username,
                'last_updated': datetime.now().isoformat(),
                'cache_ttl_hours': self.cache_ttl_hours,
                'rate_limit_info': self.get_rate_limit_info()
            }
            
            JsonHandler.save_json(metadata, str(self.cache_metadata_file))
            logger.debug("Updated cache metadata")
            
        except Exception as e:
            logger.warning(f"Failed to update cache metadata: {e}")
    
    def _load_from_cache_fallback(self, import_starred: bool, import_owned: bool, max_repos: Optional[int] = None) -> List[GitHubBookmark]:
        """Load data from cache as fallback when API fails"""
        bookmarks = []
        
        try:
            if import_starred:
                starred_data = []
                if self.starred_cache_file.exists():
                    with open(self.starred_cache_file, 'r', encoding='utf-8') as f:
                        starred_data = json.load(f)
                limited_starred_data = starred_data[:max_repos] if max_repos else starred_data
                starred_bookmarks = [self._dict_to_bookmark(bookmark_dict) for bookmark_dict in limited_starred_data]
                bookmarks.extend(starred_bookmarks)
                logger.info(f"Loaded {len(starred_bookmarks)} starred repositories from cache fallback")
            
            if import_owned:
                owned_data = []
                if self.owned_cache_file.exists():
                    with open(self.owned_cache_file, 'r', encoding='utf-8') as f:
                        owned_data = json.load(f)
                limited_owned_data = owned_data[:max_repos] if max_repos else owned_data
                owned_bookmarks = [self._dict_to_bookmark(bookmark_dict) for bookmark_dict in limited_owned_data]
                bookmarks.extend(owned_bookmarks)
                logger.info(f"Loaded {len(owned_bookmarks)} owned repositories from cache fallback")
                
        except Exception as e:
            logger.error(f"Failed to load fallback cache data: {e}")
        
        return bookmarks
    
    def _dict_to_bookmark(self, bookmark_dict: Dict[str, Any]) -> GitHubBookmark:
        """Convert dictionary back to GitHubBookmark object"""
        return GitHubBookmark(
            name=bookmark_dict['name'],
            url=bookmark_dict['url'],
            description=bookmark_dict['description'],
            tags=bookmark_dict['tags'],
            suggested_collection=bookmark_dict['suggested_collection'],
            metadata=bookmark_dict['metadata']
        )
    
    def clear_cache(self) -> None:
        """Clear all cached GitHub data"""
        try:
            files_removed = 0
            for cache_file in [self.starred_cache_file, self.owned_cache_file, self.cache_metadata_file]:
                if cache_file.exists():
                    cache_file.unlink()
                    files_removed += 1
            
            logger.info(f"Cleared GitHub cache: {files_removed} files removed")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state"""
        try:
            cache_info = {
                'enabled': self.enable_cache,
                'ttl_hours': self.cache_ttl_hours,
                'cache_dir': str(self.cache_dir),
                'files': {}
            }
            
            for name, cache_file in [
                ('starred', self.starred_cache_file),
                ('owned', self.owned_cache_file),
                ('metadata', self.cache_metadata_file)
            ]:
                if cache_file.exists():
                    stat = cache_file.stat()
                    cache_info['files'][name] = {
                        'exists': True,
                        'size_bytes': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'valid': self._is_cache_valid(cache_file) if name != 'metadata' else True
                    }
                else:
                    cache_info['files'][name] = {'exists': False}
            
            return cache_info
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {'error': str(e)}