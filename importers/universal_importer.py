"""Universal importer that can handle multiple bookmark sources"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from importers.github_importer import GitHubImporter
from importers.linkwarden_importer import LinkwardenImporter
from data_models import ImportResult
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ImportConfig:
    """Configuration for universal import"""
    # Linkwarden
    linkwarden_backup_path: Optional[str] = None
    
    # GitHub
    github_token: Optional[str] = None
    github_username: Optional[str] = None
    import_github_starred: bool = True
    import_github_owned: bool = True
    max_github_repos: Optional[int] = None
    
    # Browser (future)
    browser_bookmarks_path: Optional[str] = None
    
    # General
    dry_run: bool = False
    verbose: bool = False


@dataclass
class CombinedImportResult:
    """Result from importing multiple sources"""
    source_results: Dict[str, ImportResult] = field(default_factory=dict)
    total_bookmarks: int = 0
    total_sources: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_source_result(self, source_name: str, result: ImportResult) -> None:
        """Add result from a specific source"""
        self.source_results[source_name] = result
        self.total_bookmarks += result.total_imported
        self.total_sources += 1
        self.errors.extend(result.errors)
        self.warnings.extend(result.warnings)
    
    def add_bookmarks(self, source_name: str, bookmarks: List[Dict[str, Any]]) -> None:
        """Add bookmarks from a source"""
        result = ImportResult(
            bookmarks=bookmarks,
            total_imported=len(bookmarks),
            import_source=source_name,
            errors=[],
            warnings=[]
        )
        self.add_source_result(source_name, result)
    
    def get_all_bookmarks(self) -> List[Dict[str, Any]]:
        """Get all bookmarks from all sources"""
        all_bookmarks = []
        for result in self.source_results.values():
            all_bookmarks.extend(result.bookmarks)
        return all_bookmarks
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of import results"""
        return {
            'total_bookmarks': self.total_bookmarks,
            'total_sources': self.total_sources,
            'sources': list(self.source_results.keys()),
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'source_breakdown': {
                name: result.total_imported 
                for name, result in self.source_results.items()
            }
        }


class UniversalImporter:
    """Universal importer for multiple bookmark sources"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize universal importer"""
        self.config = config
        self.github_importer = None
        self.linkwarden_importer = LinkwardenImporter(config)
        
        logger.info("Universal importer initialized")
    
    def import_all_sources(self, import_config: ImportConfig) -> CombinedImportResult:
        """Import from all configured sources"""
        
        results = CombinedImportResult()
        
        logger.info("Starting universal import from multiple sources")
        
        # Import Linkwarden backup
        if import_config.linkwarden_backup_path:
            logger.info("Importing from Linkwarden backup...")
            try:
                linkwarden_result = self.linkwarden_importer.import_data(
                    import_config.linkwarden_backup_path
                )
                results.add_source_result('linkwarden', linkwarden_result)
                logger.info(f"Linkwarden import completed: {linkwarden_result.total_imported} bookmarks")
            except Exception as e:
                error_msg = f"Linkwarden import failed: {e}"
                logger.error(error_msg)
                results.errors.append(error_msg)
        
        # Import GitHub data
        if (import_config.github_token and import_config.github_username and 
            (import_config.import_github_starred or import_config.import_github_owned)):
            
            logger.info("Importing from GitHub...")
            try:
                # Initialize GitHub importer with updated config
                github_config = self.config.copy()
                github_config['github']['token'] = import_config.github_token
                github_config['github']['username'] = import_config.github_username
                
                self.github_importer = GitHubImporter(github_config)
                
                github_result = self.github_importer.import_data(
                    import_starred=import_config.import_github_starred,
                    import_owned=import_config.import_github_owned,
                    max_repos=import_config.max_github_repos
                )
                results.add_source_result('github', github_result)
                logger.info(f"GitHub import completed: {github_result.total_imported} bookmarks")
                
            except Exception as e:
                error_msg = f"GitHub import failed: {e}"
                logger.error(error_msg)
                results.errors.append(error_msg)
        
        # Future: Browser bookmarks import
        if import_config.browser_bookmarks_path:
            logger.warning("Browser bookmarks import not yet implemented")
            results.warnings.append("Browser bookmarks import not yet implemented")
        
        logger.info(f"Universal import completed: {results.total_bookmarks} total bookmarks from {results.total_sources} sources")
        
        return results
    
    def preview_all_sources(self, import_config: ImportConfig) -> Dict[str, Any]:
        """Preview what would be imported from all sources"""
        
        preview = {
            'sources': {},
            'total_estimated_bookmarks': 0,
            'estimated_processing_time': 0.0
        }
        
        # Preview Linkwarden backup
        if import_config.linkwarden_backup_path:
            try:
                linkwarden_preview = self.linkwarden_importer.preview_import(
                    import_config.linkwarden_backup_path
                )
                preview['sources']['linkwarden'] = linkwarden_preview
                preview['total_estimated_bookmarks'] += linkwarden_preview.get('total_bookmarks', 0)
            except Exception as e:
                logger.error(f"Failed to preview Linkwarden backup: {e}")
                preview['sources']['linkwarden'] = {'error': str(e)}
        
        # Preview GitHub data
        if (import_config.github_token and import_config.github_username):
            try:
                # Initialize GitHub importer for preview
                github_config = self.config.copy()
                github_config['github']['token'] = import_config.github_token
                github_config['github']['username'] = import_config.github_username
                
                github_importer = GitHubImporter(github_config)
                
                # Get user info for preview
                user = github_importer.user or github_importer.github.get_user(import_config.github_username)
                
                github_preview = {
                    'username': user.login,
                    'name': user.name,
                    'public_repos': user.public_repos,
                    'starred_repos': user.get_starred().totalCount if hasattr(user.get_starred(), 'totalCount') else 'Unknown',
                    'estimated_bookmarks': 0
                }
                
                if import_config.import_github_starred:
                    github_preview['estimated_bookmarks'] += github_preview.get('starred_repos', 0)
                if import_config.import_github_owned:
                    github_preview['estimated_bookmarks'] += github_preview.get('public_repos', 0)
                
                preview['sources']['github'] = github_preview
                preview['total_estimated_bookmarks'] += github_preview['estimated_bookmarks']
                
            except Exception as e:
                logger.error(f"Failed to preview GitHub data: {e}")
                preview['sources']['github'] = {'error': str(e)}
        
        # Estimate processing time (rough calculation)
        preview['estimated_processing_time'] = preview['total_estimated_bookmarks'] * 0.1  # ~0.1 seconds per bookmark
        
        return preview
    
    def validate_import_config(self, import_config: ImportConfig) -> List[str]:
        """Validate import configuration"""
        
        errors = []
        
        # Check if at least one source is configured
        has_source = False
        
        if import_config.linkwarden_backup_path:
            has_source = True
            # Check if file exists
            from pathlib import Path
            if not Path(import_config.linkwarden_backup_path).exists():
                errors.append(f"Linkwarden backup file not found: {import_config.linkwarden_backup_path}")
        
        if import_config.github_token and import_config.github_username:
            has_source = True
            # Validate GitHub credentials
            try:
                github_config = self.config.copy()
                github_config['github']['token'] = import_config.github_token
                github_config['github']['username'] = import_config.github_username
                
                github_importer = GitHubImporter(github_config)
                if not github_importer.validate_config():
                    errors.extend(github_importer.errors)
            except Exception as e:
                errors.append(f"GitHub configuration invalid: {e}")
        
        if not has_source:
            errors.append("No import sources configured. Specify at least one source.")
        
        return errors