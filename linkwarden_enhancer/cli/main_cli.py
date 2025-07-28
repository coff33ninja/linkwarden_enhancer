"""Main CLI application with comprehensive argument parsing and features"""

import argparse
import sys
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

from .interactive import InteractiveReviewer, InteractiveMenu
from .help_system import HelpSystem
from ..core.safety_manager import SafetyManager
from ..intelligence.dictionary_manager import SmartDictionaryManager
from ..intelligence.continuous_learner import ContinuousLearner
from ..intelligence.adaptive_intelligence import AdaptiveIntelligence
from ..intelligence.intelligence_manager import IntelligenceManager
from ..reporting.report_generator import ReportGenerator, ReportFormat
from ..reporting.metrics_collector import MetricsCollector
from ..config.settings import load_config
from ..utils.logging_utils import get_logger, setup_logging, setup_verbose_logging, get_component_logger
from ..utils.progress_utils import DetailedProgressTracker

logger = get_logger(__name__)


class MainCLI:
    """Main command-line interface for Linkwarden Enhancer"""
    
    def __init__(self):
        """Initialize CLI application"""
        self.config = None
        self.safety_manager = None
        self.interactive_reviewer = None
        self.interactive_menu = None
        self.metrics_collector = None
        self.help_system = None
        self.progress_tracker = None
        
        # CLI state
        self.verbose = False
        self.interactive_mode = False
        self.dry_run = False
        self.debug_mode = False
        
        # Component loggers
        self.component_loggers = {}
        
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI application"""
        
        try:
            # Parse arguments
            parser = self._create_argument_parser()
            parsed_args = parser.parse_args(args)
            
            # Setup logging based on verbosity
            log_level = 'DEBUG' if parsed_args.verbose else 'INFO'
            setup_logging(log_level)
            
            # Load configuration
            self.config = load_config(parsed_args.config)
            
            # Apply CLI overrides to config
            self._apply_cli_overrides(parsed_args)
            
            # Initialize components
            self._initialize_components()
            
            # Execute command
            return self._execute_command(parsed_args)
            
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return 1
        except Exception as e:
            logger.error(f"CLI execution failed: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create comprehensive argument parser"""
        
        parser = argparse.ArgumentParser(
            prog='linkwarden-enhancer',
            description='🔖 Linkwarden Enhancer - AI-powered bookmark management with safety features',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process bookmarks with safety checks
  linkwarden-enhancer process input.json output.json
  
  # Interactive mode with learning
  linkwarden-enhancer process input.json output.json --interactive
  
  # Import from GitHub
  linkwarden-enhancer import --github --github-token TOKEN --github-username USER
  
  # Generate reports
  linkwarden-enhancer report --operation-report input.json output.json
  
  # Validate data integrity
  linkwarden-enhancer validate input.json
  
  # Show learning statistics
  linkwarden-enhancer stats --learning
            """
        )
        
        # Global options
        parser.add_argument('-v', '--verbose', action='store_true',
                          help='Enable verbose logging and debugging output')
        parser.add_argument('--debug', action='store_true',
                          help='Enable debug mode with detailed component logging')
        parser.add_argument('-c', '--config', type=str,
                          help='Path to configuration file')
        parser.add_argument('--dry-run', action='store_true',
                          help='Perform dry run without making changes')
        parser.add_argument('--interactive', action='store_true',
                          help='Enable interactive mode for reviewing suggestions')
        parser.add_argument('--progress-detail', choices=['minimal', 'standard', 'detailed'], 
                          default='standard', help='Progress indicator detail level')
        parser.add_argument('--log-file', type=str,
                          help='Path to log file for detailed logging')
        parser.add_argument('--component-debug', nargs='+',
                          help='Enable debug logging for specific components')
        parser.add_argument('--learning-feedback', action='store_true',
                          help='Enable learning feedback collection and display')
        parser.add_argument('--performance-metrics', action='store_true',
                          help='Enable detailed performance metrics collection')
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Process command
        self._add_process_command(subparsers)
        
        # Import command
        self._add_import_command(subparsers)
        
        # Validate command
        self._add_validate_command(subparsers)
        
        # Report command
        self._add_report_command(subparsers)
        
        # Stats command
        self._add_stats_command(subparsers)
        
        # Backup command
        self._add_backup_command(subparsers)
        
        # Intelligence command
        self._add_intelligence_command(subparsers)
        
        # Cache command
        self._add_cache_command(subparsers)
        
        # Interactive menu command
        self._add_menu_command(subparsers)
        
        # Help command
        self._add_help_command(subparsers)
        
        return parser
    
    def _add_process_command(self, subparsers) -> None:
        """Add process command"""
        
        process_parser = subparsers.add_parser('process', 
                                             help='Process bookmarks with AI enhancement and safety checks')
        
        process_parser.add_argument('input_file', help='Input JSON file path')
        process_parser.add_argument('output_file', help='Output JSON file path')
        
        # Import options
        process_parser.add_argument('--import-github', action='store_true',
                                  help='Import GitHub starred repositories')
        process_parser.add_argument('--github-token', type=str,
                                  help='GitHub API token')
        process_parser.add_argument('--github-username', type=str,
                                  help='GitHub username')
        process_parser.add_argument('--import-browser', type=str,
                                  help='Browser bookmarks file path')
        
        # Cache options for GitHub import
        process_parser.add_argument('--force-refresh-github', action='store_true',
                                  help='Force refresh GitHub cache during import')
        process_parser.add_argument('--disable-github-cache', action='store_true',
                                  help='Disable GitHub caching for this operation')
        
        # Enhancement options
        process_parser.add_argument('--enable-scraping', action='store_true',
                                  help='Enable web scraping for metadata enhancement')
        process_parser.add_argument('--enable-ai-analysis', action='store_true',
                                  help='Enable AI content analysis and suggestions')
        process_parser.add_argument('--enable-learning', action='store_true',
                                  help='Enable continuous learning from processed data')
        process_parser.add_argument('--enable-clustering', action='store_true',
                                  help='Enable AI-powered bookmark clustering')
        process_parser.add_argument('--enable-similarity-detection', action='store_true',
                                  help='Enable duplicate and similar bookmark detection')
        process_parser.add_argument('--enable-smart-tagging', action='store_true',
                                  help='Enable AI-powered smart tag suggestions')
        process_parser.add_argument('--enable-network-analysis', action='store_true',
                                  help='Enable bookmark relationship network analysis')
        
        # AI configuration options
        process_parser.add_argument('--ollama-model', type=str, default='llama2',
                                  help='Ollama model for LLM features (default: llama2)')
        process_parser.add_argument('--similarity-threshold', type=float, default=0.85,
                                  help='Similarity threshold for duplicate detection (default: 0.85)')
        process_parser.add_argument('--max-clusters', type=int, default=50,
                                  help='Maximum number of clusters for organization (default: 50)')
        process_parser.add_argument('--confidence-threshold', type=float, default=0.7,
                                  help='Minimum confidence for AI suggestions (default: 0.7)')
        
        # Dictionary and learning options
        process_parser.add_argument('--enable-dictionary-learning', action='store_true',
                                  help='Enable smart dictionary learning from data')
        process_parser.add_argument('--dictionary-update-mode', choices=['incremental', 'full', 'none'],
                                  default='incremental', help='Dictionary update mode')
        process_parser.add_argument('--learning-rate', type=float, default=0.1,
                                  help='Learning rate for adaptive intelligence (default: 0.1)')
        process_parser.add_argument('--feedback-weight', type=float, default=1.0,
                                  help='Weight for user feedback in learning (default: 1.0)')
        
        # Safety options
        process_parser.add_argument('--max-deletion-percent', type=float, default=10.0,
                                  help='Maximum percentage of items that can be deleted (default: 10%%)')
        process_parser.add_argument('--backup-before-processing', action='store_true', default=True,
                                  help='Create backup before processing (default: enabled)')
        process_parser.add_argument('--skip-integrity-check', action='store_true',
                                  help='Skip integrity checks (not recommended)')
        process_parser.add_argument('--safety-pause-threshold', type=int, default=100,
                                  help='Pause for confirmation after N changes (default: 100)')
        process_parser.add_argument('--auto-approve-low-risk', action='store_true',
                                  help='Auto-approve low-risk changes without confirmation')
        
        # Output options
        process_parser.add_argument('--generate-report', action='store_true',
                                  help='Generate detailed processing report')
        process_parser.add_argument('--report-format', choices=['json', 'html', 'csv', 'md'],
                                  default='json', help='Report format (default: json)')
        process_parser.add_argument('--export-learning-data', action='store_true',
                                  help='Export learning data after processing')
        process_parser.add_argument('--show-suggestions-summary', action='store_true',
                                  help='Show summary of AI suggestions made')
    
    def _add_import_command(self, subparsers) -> None:
        """Add import command"""
        
        import_parser = subparsers.add_parser('import', 
                                            help='Import bookmarks from various sources')
        
        # Source options
        import_parser.add_argument('--github', action='store_true',
                                 help='Import from GitHub')
        import_parser.add_argument('--github-token', type=str,
                                 help='GitHub API token')
        import_parser.add_argument('--github-username', type=str,
                                 help='GitHub username')
        import_parser.add_argument('--github-starred', action='store_true', default=True,
                                 help='Import starred repositories (default: enabled)')
        import_parser.add_argument('--github-owned', action='store_true',
                                 help='Import owned repositories')
        import_parser.add_argument('--max-repos', type=int,
                                 help='Maximum number of repositories to import')
        
        import_parser.add_argument('--browser', type=str,
                                 help='Browser bookmarks file path')
        import_parser.add_argument('--linkwarden-backup', type=str,
                                 help='Linkwarden backup JSON file path')
        
        # Cache options
        import_parser.add_argument('--force-refresh', action='store_true',
                                 help='Force refresh cached data (ignore cache)')
        import_parser.add_argument('--disable-cache', action='store_true',
                                 help='Disable caching for this operation')
        
        # Output options
        import_parser.add_argument('-o', '--output', type=str, required=True,
                                 help='Output file path')
        import_parser.add_argument('--merge-with', type=str,
                                 help='Existing file to merge imports with')
    
    def _add_validate_command(self, subparsers) -> None:
        """Add validate command"""
        
        validate_parser = subparsers.add_parser('validate',
                                              help='Validate bookmark data integrity')
        
        validate_parser.add_argument('input_file', help='Input JSON file to validate')
        validate_parser.add_argument('--fix-issues', action='store_true',
                                   help='Attempt to fix validation issues')
        validate_parser.add_argument('--detailed-report', action='store_true',
                                   help='Generate detailed validation report')
    
    def _add_report_command(self, subparsers) -> None:
        """Add report command"""
        
        report_parser = subparsers.add_parser('report',
                                            help='Generate various reports')
        
        report_subparsers = report_parser.add_subparsers(dest='report_type', 
                                                       help='Report types')
        
        # Operation report
        op_report = report_subparsers.add_parser('operation',
                                               help='Generate operation comparison report')
        op_report.add_argument('before_file', help='Before state file')
        op_report.add_argument('after_file', help='After state file')
        op_report.add_argument('--operation-name', type=str, default='operation',
                             help='Name of the operation')
        
        # Period report
        period_report = report_subparsers.add_parser('period',
                                                   help='Generate period activity report')
        period_report.add_argument('--hours', type=int, default=24,
                                 help='Time period in hours (default: 24)')
        
        # Performance report
        perf_report = report_subparsers.add_parser('performance',
                                                 help='Generate performance metrics report')
        perf_report.add_argument('--export-metrics', action='store_true',
                               help='Export raw metrics data')
        
        # Common report options
        for subparser in [op_report, period_report, perf_report]:
            subparser.add_argument('--format', choices=['json', 'html', 'csv', 'md'],
                                 action='append', help='Output format(s)')
            subparser.add_argument('--output-dir', type=str, default='reports',
                                 help='Output directory for reports')
    
    def _add_stats_command(self, subparsers) -> None:
        """Add stats command"""
        
        stats_parser = subparsers.add_parser('stats',
                                           help='Show system statistics')
        
        stats_parser.add_argument('--learning', action='store_true',
                                help='Show learning statistics')
        stats_parser.add_argument('--intelligence', action='store_true',
                                help='Show intelligence system statistics')
        stats_parser.add_argument('--performance', action='store_true',
                                help='Show performance statistics')
        stats_parser.add_argument('--safety', action='store_true',
                                help='Show safety system statistics')
        stats_parser.add_argument('--all', action='store_true',
                                help='Show all statistics')
        stats_parser.add_argument('--export', type=str,
                                help='Export statistics to file')
    
    def _add_backup_command(self, subparsers) -> None:
        """Add backup command"""
        
        backup_parser = subparsers.add_parser('backup',
                                            help='Backup and recovery operations')
        
        backup_subparsers = backup_parser.add_subparsers(dest='backup_action',
                                                        help='Backup actions')
        
        # Create backup
        create_backup = backup_subparsers.add_parser('create',
                                                   help='Create backup')
        create_backup.add_argument('input_file', help='File to backup')
        create_backup.add_argument('--description', type=str,
                                 help='Backup description')
        
        # List backups
        list_backups = backup_subparsers.add_parser('list',
                                                  help='List available backups')
        list_backups.add_argument('--operation', type=str,
                                help='Filter by operation name')
        
        # Restore backup
        restore_backup = backup_subparsers.add_parser('restore',
                                                    help='Restore from backup')
        restore_backup.add_argument('backup_file', help='Backup file to restore')
        restore_backup.add_argument('target_file', help='Target file for restoration')
        
        # Cleanup backups
        cleanup_backups = backup_subparsers.add_parser('cleanup',
                                                     help='Clean up old backups')
        cleanup_backups.add_argument('--days', type=int, default=30,
                                   help='Keep backups newer than N days')
    
    def _add_intelligence_command(self, subparsers) -> None:
        """Add intelligence command"""
        
        intel_parser = subparsers.add_parser('intelligence',
                                           help='Intelligence system operations')
        
        intel_subparsers = intel_parser.add_subparsers(dest='intel_action',
                                                     help='Intelligence actions')
        
        # Export intelligence
        export_intel = intel_subparsers.add_parser('export',
                                                 help='Export intelligence data')
        export_intel.add_argument('--output', type=str, required=True,
                                help='Output file path')
        export_intel.add_argument('--components', nargs='+',
                                choices=['dictionary', 'learning', 'adaptation'],
                                help='Components to export')
        export_intel.add_argument('--description', type=str,
                                help='Export description')
        
        # Import intelligence
        import_intel = intel_subparsers.add_parser('import',
                                                 help='Import intelligence data')
        import_intel.add_argument('input_file', help='Intelligence data file')
        import_intel.add_argument('--components', nargs='+',
                                choices=['dictionary', 'learning', 'adaptation'],
                                help='Components to import')
        
        # Train intelligence
        train_intel = intel_subparsers.add_parser('train',
                                                help='Train intelligence from data')
        train_intel.add_argument('training_data', help='Training data file')
        train_intel.add_argument('--incremental', action='store_true',
                               help='Incremental training (preserve existing)')
    
    def _add_cache_command(self, subparsers) -> None:
        """Add cache management command"""
        
        cache_parser = subparsers.add_parser('cache',
                                           help='Cache management operations')
        
        cache_subparsers = cache_parser.add_subparsers(dest='cache_action',
                                                     help='Cache actions')
        
        # Show cache info
        info_cache = cache_subparsers.add_parser('info',
                                               help='Show cache information')
        info_cache.add_argument('--source', choices=['github', 'all'], default='all',
                              help='Show cache info for specific source')
        
        # Clear cache
        clear_cache = cache_subparsers.add_parser('clear',
                                                help='Clear cached data')
        clear_cache.add_argument('--source', choices=['github', 'all'], default='all',
                               help='Clear cache for specific source')
        clear_cache.add_argument('--confirm', action='store_true',
                               help='Skip confirmation prompt')
        
        # Refresh cache
        refresh_cache = cache_subparsers.add_parser('refresh',
                                                  help='Force refresh cached data')
        refresh_cache.add_argument('--source', choices=['github'], required=True,
                                 help='Source to refresh')
        refresh_cache.add_argument('--github-username', type=str,
                                 help='GitHub username')
        refresh_cache.add_argument('--github-token', type=str,
                                 help='GitHub token')
    
    def _add_menu_command(self, subparsers) -> None:
        """Add interactive menu command"""
        
        menu_parser = subparsers.add_parser('menu',
                                          help='Launch interactive menu interface')
        menu_parser.add_argument('--auto-load-config', action='store_true',
                               help='Automatically load default configuration')
    
    def _add_help_command(self, subparsers) -> None:
        """Add help command"""
        
        help_parser = subparsers.add_parser('help',
                                          help='Show comprehensive help and documentation')
        help_parser.add_argument('topic', nargs='?', default='overview',
                               help='Help topic (overview, getting_started, safety_features, ai_features, etc.)')
        help_parser.add_argument('--topics', action='store_true',
                               help='List all available help topics')
        help_parser.add_argument('--search', type=str,
                               help='Search help content for a term')
        help_parser.add_argument('--quick', action='store_true',
                               help='Show quick reference guide')
    
    def _apply_cli_overrides(self, args) -> None:
        """Apply CLI argument overrides to configuration"""
        
        self.verbose = args.verbose
        self.debug_mode = getattr(args, 'debug', False)
        self.interactive_mode = args.interactive
        self.dry_run = args.dry_run
        
        # Apply safety overrides
        if hasattr(args, 'max_deletion_percent'):
            self.config.setdefault('safety', {})['max_deletion_percentage'] = args.max_deletion_percent
        
        if hasattr(args, 'safety_pause_threshold'):
            self.config.setdefault('safety', {})['pause_threshold'] = args.safety_pause_threshold
        
        if hasattr(args, 'auto_approve_low_risk'):
            self.config.setdefault('safety', {})['auto_approve_low_risk'] = args.auto_approve_low_risk
        
        # Apply dry run mode
        if self.dry_run:
            self.config.setdefault('safety', {})['dry_run_mode'] = True
        
        # Apply AI configuration
        if hasattr(args, 'ollama_model'):
            self.config.setdefault('ai', {})['ollama_model'] = args.ollama_model
        
        if hasattr(args, 'similarity_threshold'):
            self.config.setdefault('ai', {})['similarity_threshold'] = args.similarity_threshold
        
        if hasattr(args, 'max_clusters'):
            self.config.setdefault('ai', {})['max_clusters'] = args.max_clusters
        
        if hasattr(args, 'confidence_threshold'):
            self.config.setdefault('ai', {})['confidence_threshold'] = args.confidence_threshold
        
        # Apply enhancement options
        if hasattr(args, 'enable_scraping'):
            self.config.setdefault('enhancement', {})['enable_scraping'] = args.enable_scraping
        
        if hasattr(args, 'enable_ai_analysis'):
            self.config.setdefault('ai', {})['enable_analysis'] = args.enable_ai_analysis
        
        if hasattr(args, 'enable_clustering'):
            self.config.setdefault('ai', {})['enable_clustering'] = args.enable_clustering
        
        if hasattr(args, 'enable_similarity_detection'):
            self.config.setdefault('ai', {})['enable_similarity'] = args.enable_similarity_detection
        
        if hasattr(args, 'enable_smart_tagging'):
            self.config.setdefault('ai', {})['enable_smart_tagging'] = args.enable_smart_tagging
        
        if hasattr(args, 'enable_network_analysis'):
            self.config.setdefault('ai', {})['enable_network_analysis'] = args.enable_network_analysis
        
        # Apply learning configuration
        if hasattr(args, 'enable_learning'):
            self.config.setdefault('intelligence', {})['enable_learning'] = args.enable_learning
        
        if hasattr(args, 'enable_dictionary_learning'):
            self.config.setdefault('intelligence', {})['enable_dictionary_learning'] = args.enable_dictionary_learning
        
        if hasattr(args, 'dictionary_update_mode'):
            self.config.setdefault('intelligence', {})['dictionary_update_mode'] = args.dictionary_update_mode
        
        if hasattr(args, 'learning_rate'):
            self.config.setdefault('intelligence', {})['learning_rate'] = args.learning_rate
        
        if hasattr(args, 'feedback_weight'):
            self.config.setdefault('intelligence', {})['feedback_weight'] = args.feedback_weight
        
        # Apply GitHub configuration
        if hasattr(args, 'github_token') and args.github_token:
            self.config.setdefault('github', {})['token'] = args.github_token
        
        if hasattr(args, 'github_username') and args.github_username:
            self.config.setdefault('github', {})['username'] = args.github_username
        
        # Apply cache configuration
        if hasattr(args, 'force_refresh') and args.force_refresh:
            self.config.setdefault('github', {}).setdefault('cache', {})['force_refresh'] = True
        
        if hasattr(args, 'force_refresh_github') and args.force_refresh_github:
            self.config.setdefault('github', {}).setdefault('cache', {})['force_refresh'] = True
        
        if hasattr(args, 'disable_cache') and args.disable_cache:
            self.config.setdefault('github', {}).setdefault('cache', {})['enabled'] = False
        
        if hasattr(args, 'disable_github_cache') and args.disable_github_cache:
            self.config.setdefault('github', {}).setdefault('cache', {})['enabled'] = False
        
        # Apply logging configuration
        if hasattr(args, 'log_file') and args.log_file:
            self.config.setdefault('logging', {})['file'] = args.log_file
        
        if hasattr(args, 'component_debug') and args.component_debug:
            self.config.setdefault('logging', {})['component_debug'] = args.component_debug
        
        # Apply progress configuration
        if hasattr(args, 'progress_detail'):
            self.config.setdefault('cli', {})['progress_detail'] = args.progress_detail
        
        if hasattr(args, 'learning_feedback'):
            self.config.setdefault('cli', {})['learning_feedback'] = args.learning_feedback
        
        if hasattr(args, 'performance_metrics'):
            self.config.setdefault('cli', {})['performance_metrics'] = args.performance_metrics
    
    def _initialize_components(self) -> None:
        """Initialize system components"""
        
        try:
            # Setup enhanced logging if verbose or debug mode
            if self.verbose or self.debug_mode:
                log_file = self.config.get('logging', {}).get('file')
                if self.debug_mode:
                    log_file = log_file or 'logs/debug.log'
                
                setup_verbose_logging(
                    enable_debug=self.debug_mode,
                    component_filters=self._get_component_log_filters(),
                    log_file=log_file
                )
            
            # Initialize component loggers
            self._initialize_component_loggers()
            
            # Initialize core components
            self.safety_manager = SafetyManager(self.config)
            self.metrics_collector = MetricsCollector(self.config)
            self.help_system = HelpSystem()
            
            # Initialize progress tracker for multi-phase operations
            progress_detail = self.config.get('cli', {}).get('progress_detail', 'standard')
            if progress_detail == 'detailed':
                phases = ['validation', 'backup', 'enhancement', 'ai_analysis', 'learning', 'output']
                self.progress_tracker = DetailedProgressTracker(phases, verbose=self.verbose)
            
            # Initialize interactive components if needed
            if self.interactive_mode:
                self.interactive_reviewer = InteractiveReviewer(self.config)
                self.interactive_menu = InteractiveMenu()
            
            logger.info("CLI components initialized successfully")
            
            if self.debug_mode:
                logger.debug(f"Configuration: {json.dumps(self.config, indent=2, default=str)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _get_component_log_filters(self) -> Dict[str, str]:
        """Get component-specific log level filters"""
        
        filters = {
            'linkwarden_enhancer.core': 'DEBUG' if self.debug_mode else 'INFO',
            'linkwarden_enhancer.ai': 'DEBUG' if self.debug_mode else 'INFO',
            'linkwarden_enhancer.intelligence': 'DEBUG' if self.debug_mode else 'INFO',
            'linkwarden_enhancer.enhancement': 'DEBUG' if self.debug_mode else 'INFO',
            'linkwarden_enhancer.importers': 'DEBUG' if self.debug_mode else 'INFO',
            'linkwarden_enhancer.reporting': 'DEBUG' if self.debug_mode else 'INFO',
            'linkwarden_enhancer.cli': 'DEBUG' if self.debug_mode else 'INFO',
            'urllib3': 'WARNING',
            'requests': 'WARNING',
            'selenium': 'WARNING',
            'transformers': 'INFO'
        }
        
        # Apply component-specific debug settings
        component_debug = self.config.get('logging', {}).get('component_debug', [])
        for component in component_debug:
            filters[f'linkwarden_enhancer.{component}'] = 'DEBUG'
        
        return filters
    
    def _initialize_component_loggers(self) -> None:
        """Initialize enhanced component loggers"""
        
        components = ['core', 'ai', 'intelligence', 'enhancement', 'importers', 'reporting', 'cli']
        
        for component in components:
            self.component_loggers[component] = get_component_logger(
                f'linkwarden_enhancer.{component}',
                verbose=self.verbose or self.debug_mode
            )
    
    def _execute_command(self, args) -> int:
        """Execute the specified command"""
        
        try:
            if not args.command:
                print("❌ No command specified. Use --help for usage information.")
                return 1
            
            # Track command execution
            with self.metrics_collector.track_operation(f"cli_{args.command}"):
                
                if args.command == 'process':
                    return self._execute_process_command(args)
                elif args.command == 'import':
                    return self._execute_import_command(args)
                elif args.command == 'validate':
                    return self._execute_validate_command(args)
                elif args.command == 'report':
                    return self._execute_report_command(args)
                elif args.command == 'stats':
                    return self._execute_stats_command(args)
                elif args.command == 'backup':
                    return self._execute_backup_command(args)
                elif args.command == 'intelligence':
                    return self._execute_intelligence_command(args)
                elif args.command == 'cache':
                    return self._execute_cache_command(args)
                elif args.command == 'menu':
                    return self._execute_menu_command(args)
                elif args.command == 'help':
                    return self._execute_help_command(args)
                else:
                    print(f"❌ Unknown command: {args.command}")
                    return 1
            
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return 1 
   
    def _execute_process_command(self, args) -> int:
        """Execute process command with enhanced progress tracking and learning feedback"""
        
        try:
            print(f"🔄 Processing bookmarks: {args.input_file} -> {args.output_file}")
            
            if self.dry_run:
                print("🔍 DRY RUN MODE - No changes will be made")
            
            # Initialize detailed progress tracking if enabled
            learning_feedback_enabled = getattr(args, 'learning_feedback', False) or \
                                     self.config.get('cli', {}).get('learning_feedback', False)
            
            performance_metrics_enabled = getattr(args, 'performance_metrics', False) or \
                                        self.config.get('cli', {}).get('performance_metrics', False)
            
            # Define processing phases for detailed tracking
            processing_phases = [
                'validation',
                'backup', 
                'import',
                'enhancement',
                'ai_analysis',
                'intelligence_learning',
                'output_generation'
            ]
            
            # Initialize detailed progress tracker if requested
            if self.progress_tracker:
                progress_summary = self._execute_process_with_detailed_tracking(
                    args, processing_phases, learning_feedback_enabled, performance_metrics_enabled
                )
            else:
                # Use standard processing
                progress_summary = self._execute_process_standard(args)
            
            # Show enhanced results with learning feedback
            if progress_summary.get('success', False):
                self._display_enhanced_process_results(
                    progress_summary, learning_feedback_enabled, performance_metrics_enabled
                )
                
                # Generate comprehensive report if requested
                if getattr(args, 'generate_report', False):
                    self._generate_enhanced_processing_report(args, progress_summary)
                
                return 0
            else:
                self._display_process_errors(progress_summary)
                return 1
                
        except Exception as e:
            logger.error(f"Process command failed: {e}")
            print(f"❌ Processing failed: {e}")
            return 1
    
    def _execute_process_with_detailed_tracking(self, args, phases, learning_feedback, performance_metrics):
        """Execute process with detailed phase tracking and learning feedback"""
        
        from ..intelligence.intelligence_manager import IntelligenceManager
        from ..intelligence.continuous_learner import ContinuousLearner
        from ..intelligence.adaptive_intelligence import AdaptiveIntelligence
        
        # Initialize intelligence components for learning feedback
        intelligence_manager = None
        continuous_learner = None
        adaptive_intelligence = None
        
        if learning_feedback:
            try:
                intelligence_manager = IntelligenceManager(self.config)
                continuous_learner = ContinuousLearner(self.config)
                adaptive_intelligence = AdaptiveIntelligence(self.config)
                logger.info("Intelligence components initialized for learning feedback")
            except Exception as e:
                logger.warning(f"Failed to initialize intelligence components: {e}")
                learning_feedback = False
        
        # Start detailed progress tracking
        progress_summary = {
            'success': False,
            'phases_completed': {},
            'learning_stats': {},
            'performance_metrics': {},
            'total_items_processed': 0,
            'execution_time': 0,
            'warnings': [],
            'errors': []
        }
        
        start_time = time.time()
        
        try:
            # Phase 1: Validation
            validation_progress = self.progress_tracker.start_phase('validation', 1)
            validation_progress.update(0, "Validating input file and data schema")
            
            # Load and validate input
            import json
            from pathlib import Path
            
            if not Path(args.input_file).exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
            
            with open(args.input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            validation_result = self.safety_manager.validator.validate_json_schema(
                input_data, "linkwarden_backup"
            )
            
            if not validation_result.valid:
                progress_summary['errors'].extend(validation_result.errors)
                validation_progress.finish("Validation failed")
                self.progress_tracker.finish_phase('validation', 0)
                return progress_summary
            
            total_items = (
                validation_result.total_bookmarks + 
                validation_result.total_collections + 
                validation_result.total_tags
            )
            
            validation_progress.finish("Validation completed")
            self.progress_tracker.finish_phase('validation', 1, {
                'total_bookmarks': validation_result.total_bookmarks,
                'total_collections': validation_result.total_collections,
                'total_tags': validation_result.total_tags,
                'schema_valid': validation_result.valid
            })
            
            # Phase 2: Backup
            backup_progress = self.progress_tracker.start_phase('backup', 1)
            backup_progress.update(0, "Creating safety backup")
            
            backups_created = []
            if not self.dry_run:
                backup_path = self.safety_manager.backup_system.create_backup(
                    args.input_file, description="Pre-processing backup"
                )
                backups_created.append(backup_path)
            
            backup_progress.finish("Backup completed")
            self.progress_tracker.finish_phase('backup', len(backups_created), {
                'backups_created': len(backups_created),
                'dry_run': self.dry_run
            })
            
            # Phase 3: Import (if requested)
            import_stats = {'items_imported': 0, 'sources': []}
            if getattr(args, 'import_github', False) or getattr(args, 'import_browser', None):
                import_progress = self.progress_tracker.start_phase('import', 1)
                import_progress.update(0, "Importing from external sources")
                
                # Handle GitHub import
                if getattr(args, 'import_github', False):
                    try:
                        from ..importers.github_importer import GitHubImporter
                        github_importer = GitHubImporter(self.config)
                        github_data = github_importer.import_starred_repos()
                        import_stats['items_imported'] += len(github_data.get('bookmarks', []))
                        import_stats['sources'].append('github')
                    except Exception as e:
                        progress_summary['warnings'].append(f"GitHub import failed: {e}")
                
                # Handle browser import
                if getattr(args, 'import_browser', None):
                    try:
                        from ..importers.universal_importer import UniversalImporter
                        universal_importer = UniversalImporter(self.config)
                        browser_data = universal_importer.import_browser_bookmarks(args.import_browser)
                        import_stats['items_imported'] += len(browser_data.get('bookmarks', []))
                        import_stats['sources'].append('browser')
                    except Exception as e:
                        progress_summary['warnings'].append(f"Browser import failed: {e}")
                
                import_progress.finish("Import completed")
                self.progress_tracker.finish_phase('import', import_stats['items_imported'], import_stats)
            
            # Phase 4: Enhancement
            enhancement_progress = self.progress_tracker.start_phase('enhancement', total_items)
            enhancement_stats = {
                'items_enhanced': 0,
                'metadata_extracted': 0,
                'scraping_successes': 0,
                'scraping_failures': 0
            }
            
            if getattr(args, 'enable_scraping', False):
                try:
                    from ..enhancement.link_enhancement_engine import LinkEnhancementEngine
                    enhancement_engine = LinkEnhancementEngine(self.config)
                    
                    bookmarks = input_data.get('bookmarks', [])
                    for i, bookmark in enumerate(bookmarks):
                        enhancement_progress.update(i, f"Enhancing bookmark: {bookmark.get('name', 'Unknown')}")
                        
                        try:
                            enhanced_data = enhancement_engine.enhance_bookmark(bookmark)
                            if enhanced_data:
                                enhancement_stats['items_enhanced'] += 1
                                enhancement_stats['metadata_extracted'] += len(enhanced_data.get('metadata', {}))
                                enhancement_stats['scraping_successes'] += 1
                        except Exception as e:
                            enhancement_stats['scraping_failures'] += 1
                            logger.debug(f"Enhancement failed for bookmark {bookmark.get('url', '')}: {e}")
                    
                except Exception as e:
                    progress_summary['warnings'].append(f"Enhancement engine failed: {e}")
            
            enhancement_progress.finish("Enhancement completed")
            self.progress_tracker.finish_phase('enhancement', enhancement_stats['items_enhanced'], enhancement_stats)
            
            # Phase 5: AI Analysis
            ai_progress = self.progress_tracker.start_phase('ai_analysis', total_items)
            ai_stats = {
                'content_analyzed': 0,
                'clusters_created': 0,
                'similarities_found': 0,
                'tags_suggested': 0,
                'categories_suggested': 0
            }
            
            if getattr(args, 'enable_ai_analysis', False):
                try:
                    from ..ai.content_analyzer import ContentAnalyzer
                    from ..ai.clustering_engine import ClusteringEngine
                    from ..ai.similarity_engine import SimilarityEngine
                    from ..ai.tag_predictor import TagPredictor
                    
                    content_analyzer = ContentAnalyzer(self.config)
                    clustering_engine = ClusteringEngine(self.config)
                    similarity_engine = SimilarityEngine(self.config)
                    tag_predictor = TagPredictor(self.config)
                    
                    bookmarks = input_data.get('bookmarks', [])
                    
                    # Content analysis
                    for i, bookmark in enumerate(bookmarks):
                        ai_progress.update(i, f"Analyzing content: {bookmark.get('name', 'Unknown')}")
                        
                        try:
                            analysis_result = content_analyzer.analyze_bookmark(bookmark)
                            if analysis_result:
                                ai_stats['content_analyzed'] += 1
                        except Exception as e:
                            logger.debug(f"Content analysis failed: {e}")
                    
                    # Clustering
                    if getattr(args, 'enable_clustering', False):
                        try:
                            clusters = clustering_engine.cluster_bookmarks(bookmarks)
                            ai_stats['clusters_created'] = len(clusters)
                        except Exception as e:
                            logger.debug(f"Clustering failed: {e}")
                    
                    # Similarity detection
                    if getattr(args, 'enable_similarity_detection', False):
                        try:
                            similarities = similarity_engine.find_similar_bookmarks(bookmarks)
                            ai_stats['similarities_found'] = len(similarities)
                        except Exception as e:
                            logger.debug(f"Similarity detection failed: {e}")
                    
                    # Tag prediction
                    if getattr(args, 'enable_smart_tagging', False):
                        try:
                            for bookmark in bookmarks:
                                predicted_tags = tag_predictor.predict_tags(bookmark)
                                ai_stats['tags_suggested'] += len(predicted_tags)
                        except Exception as e:
                            logger.debug(f"Tag prediction failed: {e}")
                    
                except Exception as e:
                    progress_summary['warnings'].append(f"AI analysis failed: {e}")
            
            ai_progress.finish("AI analysis completed")
            self.progress_tracker.finish_phase('ai_analysis', ai_stats['content_analyzed'], ai_stats)
            
            # Phase 6: Intelligence Learning
            learning_stats = {
                'patterns_learned': 0,
                'dictionary_updates': 0,
                'feedback_processed': 0,
                'adaptations_made': 0
            }
            
            if learning_feedback and getattr(args, 'enable_learning', False):
                learning_progress = self.progress_tracker.start_phase('intelligence_learning', total_items)
                learning_progress.update(0, "Processing learning data")
                
                try:
                    # Continuous learning from processed data
                    if continuous_learner:
                        learning_result = continuous_learner.learn_from_bookmarks(input_data.get('bookmarks', []))
                        learning_stats['patterns_learned'] = learning_result.get('patterns_learned', 0)
                        learning_stats['dictionary_updates'] = learning_result.get('dictionary_updates', 0)
                    
                    # Adaptive intelligence updates
                    if adaptive_intelligence:
                        adaptation_result = adaptive_intelligence.adapt_from_usage(input_data)
                        learning_stats['adaptations_made'] = adaptation_result.get('adaptations_made', 0)
                    
                    # Intelligence manager coordination
                    if intelligence_manager:
                        feedback_result = intelligence_manager.process_learning_feedback(input_data)
                        learning_stats['feedback_processed'] = feedback_result.get('feedback_processed', 0)
                    
                except Exception as e:
                    progress_summary['warnings'].append(f"Intelligence learning failed: {e}")
                
                learning_progress.finish("Learning completed")
                self.progress_tracker.finish_phase('intelligence_learning', learning_stats['patterns_learned'], learning_stats)
            
            # Phase 7: Output Generation
            output_progress = self.progress_tracker.start_phase('output_generation', 1)
            output_progress.update(0, "Generating output file")
            
            # Execute the actual safe cleanup
            result = self.safety_manager.execute_safe_cleanup(
                input_file=args.input_file,
                output_file=args.output_file,
                import_github=getattr(args, 'import_github', False),
                import_browser=getattr(args, 'import_browser', None)
            )
            
            output_progress.finish("Output generated")
            self.progress_tracker.finish_phase('output_generation', 1, {
                'output_file': args.output_file,
                'success': result.success
            })
            
            # Compile final results
            progress_summary.update({
                'success': result.success,
                'total_items_processed': total_items,
                'execution_time': time.time() - start_time,
                'learning_stats': learning_stats,
                'enhancement_stats': enhancement_stats,
                'ai_stats': ai_stats,
                'import_stats': import_stats,
                'result': result
            })
            
            if performance_metrics:
                progress_summary['performance_metrics'] = self._collect_performance_metrics()
            
            return progress_summary
            
        except Exception as e:
            progress_summary['errors'].append(str(e))
            progress_summary['execution_time'] = time.time() - start_time
            return progress_summary
    
    def _execute_process_standard(self, args):
        """Execute standard process without detailed tracking"""
        
        start_time = time.time()
        
        try:
            result = self.safety_manager.execute_safe_cleanup(
                input_file=args.input_file,
                output_file=args.output_file,
                import_github=getattr(args, 'import_github', False),
                import_browser=getattr(args, 'import_browser', None)
            )
            
            return {
                'success': result.success,
                'result': result,
                'execution_time': time.time() - start_time,
                'warnings': result.warnings if hasattr(result, 'warnings') else [],
                'errors': result.errors if hasattr(result, 'errors') else []
            }
            
        except Exception as e:
            return {
                'success': False,
                'execution_time': time.time() - start_time,
                'errors': [str(e)],
                'warnings': []
            }
    
    def _display_enhanced_process_results(self, progress_summary, learning_feedback, performance_metrics):
        """Display enhanced processing results with learning feedback"""
        
        print("✅ Processing completed successfully!")
        print(f"⏱️ Execution time: {progress_summary['execution_time']:.2f} seconds")
        print(f"📊 Total items processed: {progress_summary['total_items_processed']}")
        
        # Show phase completion summary
        if self.progress_tracker:
            self.progress_tracker.show_overall_progress()
        
        # Show learning statistics
        if learning_feedback and 'learning_stats' in progress_summary:
            learning_stats = progress_summary['learning_stats']
            print("\n🧠 Learning Statistics:")
            print(f"   • Patterns learned: {learning_stats.get('patterns_learned', 0)}")
            print(f"   • Dictionary updates: {learning_stats.get('dictionary_updates', 0)}")
            print(f"   • Feedback processed: {learning_stats.get('feedback_processed', 0)}")
            print(f"   • Adaptations made: {learning_stats.get('adaptations_made', 0)}")
        
        # Show enhancement statistics
        if 'enhancement_stats' in progress_summary:
            enhancement_stats = progress_summary['enhancement_stats']
            print("\n🔧 Enhancement Statistics:")
            print(f"   • Items enhanced: {enhancement_stats.get('items_enhanced', 0)}")
            print(f"   • Metadata extracted: {enhancement_stats.get('metadata_extracted', 0)}")
            print(f"   • Scraping successes: {enhancement_stats.get('scraping_successes', 0)}")
            print(f"   • Scraping failures: {enhancement_stats.get('scraping_failures', 0)}")
        
        # Show AI analysis statistics
        if 'ai_stats' in progress_summary:
            ai_stats = progress_summary['ai_stats']
            print("\n🤖 AI Analysis Statistics:")
            print(f"   • Content analyzed: {ai_stats.get('content_analyzed', 0)}")
            print(f"   • Clusters created: {ai_stats.get('clusters_created', 0)}")
            print(f"   • Similarities found: {ai_stats.get('similarities_found', 0)}")
            print(f"   • Tags suggested: {ai_stats.get('tags_suggested', 0)}")
        
        # Show performance metrics
        if performance_metrics and 'performance_metrics' in progress_summary:
            perf_metrics = progress_summary['performance_metrics']
            print("\n📈 Performance Metrics:")
            for metric_name, metric_value in perf_metrics.items():
                print(f"   • {metric_name}: {metric_value}")
        
        # Show warnings
        if progress_summary.get('warnings'):
            print(f"\n⚠️ Warnings ({len(progress_summary['warnings'])}):")
            for warning in progress_summary['warnings']:
                print(f"   • {warning}")
        
        # Show comprehensive learning summary if available
        if learning_feedback and self.progress_tracker:
            self.progress_tracker.show_learning_summary()
    
    def _display_process_errors(self, progress_summary):
        """Display processing errors"""
        
        print("❌ Processing failed!")
        
        if progress_summary.get('errors'):
            print("Errors:")
            for error in progress_summary['errors']:
                print(f"  • {error}")
        
        if progress_summary.get('warnings'):
            print("Warnings:")
            for warning in progress_summary['warnings']:
                print(f"  • {warning}")
    
    def _collect_performance_metrics(self):
        """Collect detailed performance metrics"""
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            return {
                'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'threads_count': process.num_threads(),
                'open_files': len(process.open_files()),
                'system_memory_percent': psutil.virtual_memory().percent,
                'system_cpu_percent': psutil.cpu_percent()
            }
        except ImportError:
            return {'note': 'psutil not available for detailed metrics'}
        except Exception as e:
            return {'error': f'Failed to collect metrics: {e}'}
    
    def _generate_enhanced_processing_report(self, args, progress_summary):
        """Generate enhanced processing report with learning data"""
        
        try:
            from ..reporting.report_generator import ReportGenerator, ReportFormat
            
            report_generator = ReportGenerator(self.config)
            
            # Prepare comprehensive report data
            report_data = {
                'operation': 'enhanced_process',
                'timestamp': time.time(),
                'input_file': args.input_file,
                'output_file': args.output_file,
                'execution_summary': progress_summary,
                'configuration': {
                    'dry_run': self.dry_run,
                    'verbose': self.verbose,
                    'interactive': self.interactive_mode
                }
            }
            
            # Add phase details if available
            if self.progress_tracker:
                report_data['phase_summary'] = self.progress_tracker.get_phase_summary()
            
            # Generate report in requested format
            report_format = getattr(args, 'report_format', 'json')
            if report_format == 'json':
                format_enum = ReportFormat.JSON
            elif report_format == 'html':
                format_enum = ReportFormat.HTML
            elif report_format == 'csv':
                format_enum = ReportFormat.CSV
            else:
                format_enum = ReportFormat.MARKDOWN
            
            report_path = report_generator.generate_operation_report(
                before_data={},
                after_data=progress_summary.get('result', {}),
                operation_name='enhanced_process',
                format=format_enum,
                additional_data=report_data
            )
            
            print(f"📋 Enhanced processing report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate enhanced report: {e}")
            print(f"⚠️ Report generation failed: {e}")
    
    def _execute_import_command(self, args) -> int:
        """Execute import command"""
        
        try:
            print("📥 Importing bookmarks from sources...")
            
            from ..importers.universal_importer import UniversalImporter, ImportConfig
            
            # Create import configuration
            import_config = ImportConfig(
                linkwarden_backup_path=args.linkwarden_backup,
                github_token=args.github_token,
                github_username=args.github_username,
                import_github_starred=args.github_starred if args.github else False,
                import_github_owned=args.github_owned if args.github else False,
                max_github_repos=args.max_repos,
                browser_bookmarks_path=args.browser,
                dry_run=self.dry_run,
                verbose=self.verbose
            )
            
            # Initialize importer
            importer = UniversalImporter(self.config)
            
            # Validate configuration
            validation_errors = importer.validate_import_config(import_config)
            if validation_errors:
                print("❌ Import configuration errors:")
                for error in validation_errors:
                    print(f"  - {error}")
                return 1
            
            # Execute import
            result = importer.import_all_sources(import_config)
            
            # Save results
            if result.total_bookmarks > 0:
                all_bookmarks = result.get_all_bookmarks()
                
                # Merge with existing file if specified
                if args.merge_with and Path(args.merge_with).exists():
                    print(f"🔄 Merging with existing file: {args.merge_with}")
                    with open(args.merge_with, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Simple merge - add to existing bookmarks
                    existing_data.setdefault('bookmarks', []).extend(all_bookmarks)
                    
                    with open(args.output, 'w') as f:
                        json.dump(existing_data, f, indent=2, default=str)
                else:
                    # Create new file
                    output_data = {
                        'bookmarks': all_bookmarks,
                        'collections': [],
                        'tags': [],
                        'imported_at': datetime.now().isoformat(),
                        'import_summary': result.get_summary()
                    }
                    
                    with open(args.output, 'w') as f:
                        json.dump(output_data, f, indent=2, default=str)
                
                print(f"✅ Import completed: {result.total_bookmarks} bookmarks saved to {args.output}")
                
                # Show summary
                summary = result.get_summary()
                for source, count in summary['source_breakdown'].items():
                    print(f"  📊 {source}: {count} bookmarks")
                
                if result.warnings:
                    print(f"⚠️ Warnings: {len(result.warnings)}")
                    if self.verbose:
                        for warning in result.warnings:
                            print(f"  - {warning}")
                
                return 0
            else:
                print("❌ No bookmarks imported")
                if result.errors:
                    for error in result.errors:
                        print(f"  - {error}")
                return 1
                
        except Exception as e:
            logger.error(f"Import command failed: {e}")
            print(f"❌ Import failed: {e}")
            return 1
    
    def _execute_validate_command(self, args) -> int:
        """Execute validate command"""
        
        try:
            print(f"🔍 Validating data file: {args.input_file}")
            
            # Use safety manager's validation
            validation_result = self.safety_manager.validate_data_file(args.input_file)
            
            if validation_result.get('overall_valid', False):
                print("✅ Validation passed!")
                
                # Show inventory
                inventory = validation_result.get('inventory', {})
                if inventory:
                    print("📊 Data inventory:")
                    print(f"  📖 Bookmarks: {inventory.get('total_bookmarks', 0)}")
                    print(f"  📁 Collections: {inventory.get('total_collections', 0)}")
                    print(f"  🏷️ Tags: {inventory.get('total_tags', 0)}")
                
                if validation_result.get('warnings'):
                    print(f"⚠️ Warnings: {len(validation_result['warnings'])}")
                    if self.verbose:
                        for warning in validation_result['warnings']:
                            print(f"  - {warning}")
                
                return 0
            else:
                print("❌ Validation failed!")
                
                errors = validation_result.get('errors', [])
                if errors:
                    print("Errors found:")
                    for error in errors:
                        print(f"  - {error}")
                
                if args.fix_issues:
                    print("🔧 Attempting to fix issues...")
                    # TODO: Implement issue fixing
                    print("⚠️ Issue fixing not yet implemented")
                
                return 1
                
        except Exception as e:
            logger.error(f"Validate command failed: {e}")
            print(f"❌ Validation failed: {e}")
            return 1
    
    def _execute_report_command(self, args) -> int:
        """Execute report command"""
        
        try:
            report_generator = ReportGenerator(self.config)
            
            # Determine output formats
            formats = []
            if args.format:
                formats = [ReportFormat(fmt) for fmt in args.format]
            else:
                formats = [ReportFormat.JSON]
            
            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            if args.report_type == 'operation':
                print(f"📊 Generating operation report: {args.before_file} vs {args.after_file}")
                
                # Load data files
                with open(args.before_file, 'r') as f:
                    before_data = json.load(f)
                
                with open(args.after_file, 'r') as f:
                    after_data = json.load(f)
                
                # Generate report
                generated_files = report_generator.generate_operation_report(
                    operation_name=args.operation_name,
                    before_data=before_data,
                    after_data=after_data,
                    formats=formats
                )
                
                if generated_files:
                    print("✅ Operation report generated:")
                    for format_type, file_path in generated_files.items():
                        print(f"  📄 {format_type.upper()}: {file_path}")
                    return 0
                else:
                    print("❌ Failed to generate operation report")
                    return 1
            
            elif args.report_type == 'period':
                print(f"📊 Generating period report for last {args.hours} hours")
                
                from datetime import timedelta
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=args.hours)
                
                generated_files = report_generator.generate_period_report(
                    start_date=start_time,
                    end_date=end_time,
                    formats=formats
                )
                
                if generated_files:
                    print("✅ Period report generated:")
                    for format_type, file_path in generated_files.items():
                        print(f"  📄 {format_type.upper()}: {file_path}")
                    return 0
                else:
                    print("❌ Failed to generate period report")
                    return 1
            
            elif args.report_type == 'performance':
                print("📊 Generating performance report")
                
                # Get performance summary
                performance_summary = self.metrics_collector.get_performance_summary(24)
                
                # Create performance report data
                report_data = {
                    'report_type': 'performance',
                    'generated_at': datetime.now().isoformat(),
                    'performance_summary': performance_summary,
                    'system_health': self.metrics_collector.get_system_health(),
                    'operation_statistics': self.metrics_collector.get_operation_statistics()
                }
                
                # Save report
                report_file = output_dir / f"performance_report_{int(time.time())}.json"
                with open(report_file, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                
                print(f"✅ Performance report generated: {report_file}")
                
                # Export metrics if requested
                if args.export_metrics:
                    metrics_file = self.metrics_collector.export_metrics('json', 24)
                    if metrics_file:
                        print(f"📊 Metrics exported: {metrics_file}")
                
                return 0
            
            else:
                print(f"❌ Unknown report type: {args.report_type}")
                return 1
                
        except Exception as e:
            logger.error(f"Report command failed: {e}")
            print(f"❌ Report generation failed: {e}")
            return 1
    
    def _execute_stats_command(self, args) -> int:
        """Execute stats command"""
        
        try:
            print("📊 System Statistics")
            print("=" * 50)
            
            if args.all or args.safety:
                print("\n🛡️ Safety System Statistics:")
                safety_stats = self.safety_manager.get_safety_statistics()
                self._print_nested_dict(safety_stats, indent=2)
            
            if args.all or args.intelligence:
                print("\n🧠 Intelligence System Statistics:")
                dictionary_manager = SmartDictionaryManager(self.config)
                intel_stats = dictionary_manager.get_intelligence_stats()
                self._print_nested_dict(intel_stats, indent=2)
            
            if args.all or args.learning:
                print("\n📚 Learning System Statistics:")
                continuous_learner = ContinuousLearner()
                learning_stats = continuous_learner.get_learning_statistics()
                self._print_nested_dict(learning_stats, indent=2)
                
                # Show adaptive intelligence stats
                adaptive_intel = AdaptiveIntelligence()
                adaptation_stats = adaptive_intel.get_adaptation_statistics()
                print("\n🎯 Adaptive Intelligence Statistics:")
                self._print_nested_dict(adaptation_stats, indent=2)
            
            if args.all or args.performance:
                print("\n⚡ Performance Statistics:")
                perf_stats = self.metrics_collector.get_performance_summary(24)
                self._print_nested_dict(perf_stats, indent=2)
            
            # Export statistics if requested
            if args.export:
                all_stats = {
                    'timestamp': datetime.now().isoformat(),
                    'safety': self.safety_manager.get_safety_statistics(),
                    'intelligence': SmartDictionaryManager(self.config).get_intelligence_stats(),
                    'learning': ContinuousLearner().get_learning_statistics(),
                    'adaptation': AdaptiveIntelligence().get_adaptation_statistics(),
                    'performance': self.metrics_collector.get_performance_summary(24)
                }
                
                with open(args.export, 'w') as f:
                    json.dump(all_stats, f, indent=2, default=str)
                
                print(f"\n💾 Statistics exported to: {args.export}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Stats command failed: {e}")
            print(f"❌ Failed to get statistics: {e}")
            return 1
    
    def _execute_backup_command(self, args) -> int:
        """Execute backup command"""
        
        try:
            if args.backup_action == 'create':
                print(f"💾 Creating backup of: {args.input_file}")
                
                backup_path = self.safety_manager.backup_system.create_backup(
                    args.input_file,
                    "manual",
                    {"description": args.description or "Manual backup via CLI"}
                )
                
                if backup_path:
                    print(f"✅ Backup created: {backup_path}")
                    return 0
                else:
                    print("❌ Failed to create backup")
                    return 1
            
            elif args.backup_action == 'list':
                print("💾 Available Backups:")
                print("-" * 50)
                
                backups = self.safety_manager.list_available_backups(args.operation)
                
                if backups:
                    for backup in backups:
                        print(f"📄 {backup['path']}")
                        print(f"   📅 Created: {backup['timestamp']}")
                        print(f"   📊 Size: {backup['file_size_mb']} MB")
                        print(f"   🏷️ Operation: {backup['operation_name']}")
                        print(f"   ⏰ Age: {backup['age_hours']:.1f} hours")
                        print()
                else:
                    print("No backups found")
                
                return 0
            
            elif args.backup_action == 'restore':
                print(f"🔄 Restoring backup: {args.backup_file} -> {args.target_file}")
                
                result = self.safety_manager.rollback_to_backup(args.backup_file, args.target_file)
                
                if result['success']:
                    print("✅ Backup restored successfully!")
                    print(f"⏱️ Recovery time: {result['recovery_time']:.2f} seconds")
                    return 0
                else:
                    print("❌ Backup restoration failed!")
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    return 1
            
            elif args.backup_action == 'cleanup':
                print(f"🧹 Cleaning up backups older than {args.days} days...")
                
                deleted_count = self.safety_manager.cleanup_old_backups()
                
                print(f"✅ Cleaned up {deleted_count} old backups")
                return 0
            
            else:
                print(f"❌ Unknown backup action: {args.backup_action}")
                return 1
                
        except Exception as e:
            logger.error(f"Backup command failed: {e}")
            print(f"❌ Backup operation failed: {e}")
            return 1
    
    def _execute_intelligence_command(self, args) -> int:
        """Execute intelligence command"""
        
        try:
            intelligence_manager = IntelligenceManager(self.config)
            
            if args.intel_action == 'export':
                print("🧠 Exporting intelligence data...")
                
                if args.components:
                    export_file = intelligence_manager.create_selective_export(
                        components=args.components,
                        description=args.description or "CLI selective export"
                    )
                else:
                    export_file = intelligence_manager.create_full_export(
                        description=args.description or "CLI full export"
                    )
                
                if export_file:
                    print(f"✅ Intelligence data exported: {export_file}")
                    return 0
                else:
                    print("❌ Failed to export intelligence data")
                    return 1
            
            elif args.intel_action == 'import':
                print(f"🧠 Importing intelligence data from: {args.input_file}")
                
                result = intelligence_manager.import_intelligence_data(
                    args.input_file,
                    components=args.components
                )
                
                if result['success']:
                    print("✅ Intelligence data imported successfully!")
                    print(f"📊 Components imported: {', '.join(result['components_imported'])}")
                    
                    if result['warnings']:
                        print(f"⚠️ Warnings: {len(result['warnings'])}")
                        if self.verbose:
                            for warning in result['warnings']:
                                print(f"  - {warning}")
                    
                    return 0
                else:
                    print("❌ Intelligence import failed!")
                    for error in result.get('errors', []):
                        print(f"  - {error}")
                    return 1
            
            elif args.intel_action == 'train':
                print(f"🧠 Training intelligence from: {args.training_data}")
                
                # Load training data
                with open(args.training_data, 'r') as f:
                    training_data = json.load(f)
                
                # Initialize continuous learner
                continuous_learner = ContinuousLearner()
                
                # Train from data
                if args.incremental:
                    continuous_learner.start_learning_session("incremental")
                else:
                    continuous_learner.start_learning_session("batch")
                
                bookmarks = training_data.get('bookmarks', [])
                result = continuous_learner.learn_from_new_bookmarks(bookmarks)
                
                continuous_learner.end_learning_session()
                
                if result.get('bookmarks_processed', 0) > 0:
                    print("✅ Intelligence training completed!")
                    print(f"📊 Bookmarks processed: {result['bookmarks_processed']}")
                    print(f"🆕 New patterns learned: {result['new_patterns_learned']}")
                    print(f"🔄 Patterns updated: {result['patterns_updated']}")
                    return 0
                else:
                    print("❌ Intelligence training failed!")
                    for error in result.get('errors', []):
                        print(f"  - {error}")
                    return 1
            
            else:
                print(f"❌ Unknown intelligence action: {args.intel_action}")
                return 1
                
        except Exception as e:
            logger.error(f"Intelligence command failed: {e}")
            print(f"❌ Intelligence operation failed: {e}")
            return 1
    
    def _execute_menu_command(self, args) -> int:
        """Execute interactive menu command"""
        
        try:
            if not self.interactive_menu:
                self.interactive_menu = InteractiveMenu()
                self.interactive_reviewer = InteractiveReviewer(self.config)
            
            print("🔖 Welcome to Linkwarden Enhancer Interactive Menu!")
            
            while True:
                action = self.interactive_menu.show_main_menu()
                
                if action == "quit":
                    print("👋 Goodbye!")
                    return 0
                elif action == "process":
                    self._interactive_process_bookmarks()
                elif action == "import":
                    self._interactive_import_sources()
                elif action == "stats":
                    self._interactive_show_stats()
                elif action == "config":
                    self._interactive_configuration()
                elif action == "reports":
                    self._interactive_generate_reports()
                elif action == "validate":
                    self._interactive_validate_data()
                elif action == "backup":
                    self._interactive_backup_operations()
                elif action == "help":
                    self._show_interactive_help()
                else:
                    print(f"❌ Unknown action: {action}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return 0
        except Exception as e:
            logger.error(f"Menu command failed: {e}")
            print(f"❌ Interactive menu failed: {e}")
            return 1
    
    def _execute_help_command(self, args) -> int:
        """Execute help command"""
        
        try:
            from .help_system import show_help_command
            
            if args.topics:
                show_help_command('topics')
            elif args.search:
                show_help_command(f'search:{args.search}')
            elif args.quick:
                show_help_command('quick')
            else:
                show_help_command(args.topic)
            
            return 0
            
        except Exception as e:
            logger.error(f"Help command failed: {e}")
            print(f"❌ Help system failed: {e}")
            return 1
    
    def _interactive_process_bookmarks(self) -> None:
        """Interactive bookmark processing"""
        
        print("\n🔄 Interactive Bookmark Processing")
        
        input_file = self.interactive_menu.get_file_path("Enter input file path", must_exist=True)
        if not input_file:
            return
        
        output_file = self.interactive_menu.get_file_path("Enter output file path", must_exist=False)
        if not output_file:
            return
        
        # Get processing options
        enable_github = self.interactive_menu.get_yes_no("Import from GitHub?", False)
        enable_scraping = self.interactive_menu.get_yes_no("Enable web scraping?", True)
        enable_learning = self.interactive_menu.get_yes_no("Enable learning from data?", True)
        
        print(f"\n🔄 Processing: {input_file} -> {output_file}")
        if enable_scraping:
            print("🔧 Web scraping enabled")
        if enable_learning:
            print("🧠 Learning from data enabled")
        
        # Execute processing
        result = self.safety_manager.execute_safe_cleanup(
            input_file=input_file,
            output_file=output_file,
            import_github=enable_github
        )
        
        if result.success:
            print("✅ Processing completed successfully!")
            self._show_processing_results(result)
        else:
            print("❌ Processing failed!")
            for error in result.errors:
                print(f"  - {error}")
    
    def _show_processing_results(self, result) -> None:
        """Show processing results"""
        
        print(f"⏱️ Execution time: {result.execution_time:.2f} seconds")
        
        if result.backups_created:
            print(f"💾 Backups created: {len(result.backups_created)}")
        
        if result.warnings:
            print(f"⚠️ Warnings: {len(result.warnings)}")
            if self.verbose:
                for warning in result.warnings:
                    print(f"  - {warning}")
        
        if result.enhancement_report:
            report = result.enhancement_report
            print(f"✨ Enhanced bookmarks: {report.bookmarks_enhanced}")
            print(f"📊 Metadata fields added: {report.metadata_fields_added}")
    
    def _print_nested_dict(self, data: Dict[str, Any], indent: int = 0) -> None:
        """Print nested dictionary with indentation"""
        
        for key, value in data.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_nested_dict(value, indent + 1)
            elif isinstance(value, list):
                print("  " * indent + f"{key}: [{len(value)} items]")
            else:
                print("  " * indent + f"{key}: {value}")
    
    def _generate_processing_report(self, args, result) -> None:
        """Generate processing report"""
        
        try:
            # Create report data
            report_data = {
                'operation_name': 'bookmark_processing',
                'input_file': args.input_file,
                'output_file': args.output_file,
                'execution_time': result.execution_time,
                'success': result.success,
                'backups_created': result.backups_created,
                'warnings_count': len(result.warnings),
                'errors_count': len(result.errors),
                'enhancement_report': result.enhancement_report.__dict__ if result.enhancement_report else None
            }
            
            # Save report
            report_format = ReportFormat(args.report_format)
            report_file = f"processing_report_{int(time.time())}.{args.report_format}"
            
            # Generate report file
            if report_format == ReportFormat.JSON:
                with open(report_file, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
            
            print(f"📊 Processing report generated: {report_file}")
            
        except Exception as e:
            logger.warning(f"Failed to generate processing report: {e}")

    def _execute_cache_command(self, args) -> int:
        """Execute cache management command"""
        
        try:
            if not args.cache_action:
                print("❌ No cache action specified. Use --help for usage information.")
                return 1
            
            if args.cache_action == 'info':
                return self._execute_cache_info(args)
            elif args.cache_action == 'clear':
                return self._execute_cache_clear(args)
            elif args.cache_action == 'refresh':
                return self._execute_cache_refresh(args)
            else:
                print(f"❌ Unknown cache action: {args.cache_action}")
                return 1
                
        except Exception as e:
            logger.error(f"Cache command failed: {e}")
            print(f"❌ Cache operation failed: {e}")
            return 1
    
    def _execute_cache_info(self, args) -> int:
        """Show cache information"""
        
        try:
            source = getattr(args, 'source', 'all')
            
            print("📊 Cache Information")
            print("=" * 50)
            
            if source in ['github', 'all']:
                print("\n🐙 GitHub Cache:")
                
                try:
                    from ..importers.github_importer import GitHubImporter
                    github_importer = GitHubImporter(self.config)
                    cache_info = github_importer.get_cache_info()
                    
                    print(f"  Status: {'✅ Enabled' if cache_info['enabled'] else '❌ Disabled'}")
                    print(f"  TTL: {cache_info['ttl_hours']} hours")
                    print(f"  Directory: {cache_info['cache_dir']}")
                    
                    for name, info in cache_info['files'].items():
                        if info['exists']:
                            status = "✅ Valid" if info.get('valid', True) else "⚠️ Expired"
                            size_kb = info['size_bytes'] / 1024
                            print(f"  {name.title()}: {status} ({size_kb:.1f} KB, {info['modified']})")
                        else:
                            print(f"  {name.title()}: ❌ Not found")
                    
                except Exception as e:
                    print(f"  ❌ Error getting GitHub cache info: {e}")
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache info command failed: {e}")
            print(f"❌ Failed to get cache information: {e}")
            return 1
    
    def _execute_cache_clear(self, args) -> int:
        """Clear cached data"""
        
        try:
            source = getattr(args, 'source', 'all')
            confirm = getattr(args, 'confirm', False)
            
            if not confirm:
                response = input(f"⚠️  Are you sure you want to clear {source} cache? (y/N): ")
                if response.lower() not in ['y', 'yes']:
                    print("❌ Cache clear cancelled")
                    return 0
            
            print(f"🧹 Clearing {source} cache...")
            
            if source in ['github', 'all']:
                try:
                    from ..importers.github_importer import GitHubImporter
                    github_importer = GitHubImporter(self.config)
                    github_importer.clear_cache()
                    print("  ✅ GitHub cache cleared")
                except Exception as e:
                    print(f"  ❌ Failed to clear GitHub cache: {e}")
            
            print("✅ Cache clearing completed")
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear command failed: {e}")
            print(f"❌ Failed to clear cache: {e}")
            return 1
    
    def _execute_cache_refresh(self, args) -> int:
        """Force refresh cached data"""
        
        try:
            source = getattr(args, 'source', None)
            
            if not source:
                print("❌ Source is required for cache refresh")
                return 1
            
            print(f"🔄 Force refreshing {source} cache...")
            
            if source == 'github':
                try:
                    from ..importers.github_importer import GitHubImporter
                    
                    # Override config for force refresh
                    refresh_config = self.config.copy()
                    refresh_config['github']['cache']['force_refresh'] = True
                    
                    github_importer = GitHubImporter(refresh_config)
                    
                    result = github_importer.import_data(
                        import_starred=True,
                        import_owned=False,
                        max_repos=10,  # Limit for refresh test
                        force_refresh=True
                    )
                    
                    print(f"  ✅ Refreshed {result.total_imported} repositories")
                    
                except Exception as e:
                    print(f"  ❌ Failed to refresh GitHub cache: {e}")
                    return 1
            else:
                print(f"❌ Unknown source for refresh: {source}")
                return 1
            
            print("✅ Cache refresh completed")
            return 0
            
        except Exception as e:
            logger.error(f"Cache refresh command failed: {e}")
            print(f"❌ Failed to refresh cache: {e}")
            return 1


def main():
    """Main entry point for CLI"""
    cli = MainCLI()
    return cli.run()


if __name__ == "__main__":
    import sys
    sys.exit(main())