"""Comprehensive help system for CLI"""

from typing import Dict, List, Any


class HelpSystem:
    """Comprehensive help and documentation system"""
    
    def __init__(self):
        """Initialize help system"""
        self.help_topics = self._load_help_topics()
    
    def _load_help_topics(self) -> Dict[str, Dict[str, Any]]:
        """Load help topics and documentation"""
        
        return {
            'overview': {
                'title': '🔖 Linkwarden Enhancer Overview',
                'content': """
Linkwarden Enhancer is an AI-powered bookmark management system that provides:

🛡️ SAFETY FEATURES:
  • Comprehensive data validation and integrity checking
  • Automatic backup creation before any operations
  • Progress monitoring with safety thresholds
  • Rollback and recovery capabilities

🧠 AI INTELLIGENCE:
  • Smart categorization using domain and content analysis
  • Intelligent tag suggestions based on content
  • Continuous learning from user behavior
  • Similarity detection and duplicate prevention

📥 IMPORT CAPABILITIES:
  • GitHub starred repositories and owned repos
  • Browser bookmarks (Chrome, Firefox, Safari)
  • Linkwarden backup files
  • Universal import with conflict resolution

✨ ENHANCEMENT FEATURES:
  • Web scraping for metadata enrichment
  • Content analysis and summarization
  • Network analysis for relationship discovery
  • Adaptive intelligence based on user preferences

📊 REPORTING & METRICS:
  • Comprehensive change tracking
  • Performance monitoring and statistics
  • Detailed operation reports in multiple formats
  • System health monitoring
                """
            },
            
            'getting_started': {
                'title': '🚀 Getting Started',
                'content': """
QUICK START:

1. BASIC PROCESSING:
   python cli.py process input.json output.json
   
2. INTERACTIVE MODE:
   python cli.py process input.json output.json --interactive
   
3. IMPORT FROM GITHUB:
   python cli.py import --github --github-token YOUR_TOKEN --github-username YOUR_USERNAME -o bookmarks.json
   
4. INTERACTIVE MENU:
   python cli.py menu

FIRST TIME SETUP:

1. Install dependencies:
   pip install -r requirements.txt

2. Configure GitHub (optional):
   export GITHUB_TOKEN=your_token_here
   export GITHUB_USERNAME=your_username

3. Run validation on your data:
   python cli.py validate your_bookmarks.json

4. Create your first backup:
   python cli.py backup create your_bookmarks.json
                """
            },
            
            'safety_features': {
                'title': '🛡️ Safety Features',
                'content': """
COMPREHENSIVE SAFETY SYSTEM:

VALIDATION ENGINE:
  • JSON schema validation
  • Data consistency checking
  • Relationship integrity verification
  • Field requirement validation

BACKUP SYSTEM:
  • Automatic backup before operations
  • Timestamped backup files
  • Compressed storage option
  • Retention policy management

PROGRESS MONITORING:
  • Real-time progress tracking
  • Safety threshold monitoring
  • User confirmation for risky operations
  • Detailed error and warning reporting

INTEGRITY CHECKING:
  • URL preservation verification
  • Collection relationship validation
  • Orphaned reference detection
  • Before/after comparison

RECOVERY SYSTEM:
  • Automatic rollback capabilities
  • Manual recovery procedures
  • Recovery plan generation
  • Backup verification

SAFETY COMMANDS:
  python cli.py validate data.json                    # Validate data
  python cli.py backup create data.json               # Create backup
  python cli.py backup restore backup.json data.json  # Restore backup
  python cli.py backup list                           # List backups
                """
            },
            
            'ai_features': {
                'title': '🧠 AI Features',
                'content': """
INTELLIGENT BOOKMARK MANAGEMENT:

SMART CATEGORIZATION:
  • Domain-based classification
  • Content analysis using TF-IDF and LDA
  • Pattern recognition from existing data
  • Confidence scoring for suggestions

TAG PREDICTION:
  • Machine learning-based tag suggestions
  • Content keyword extraction
  • Technology and framework detection
  • Gaming and entertainment classification

CONTINUOUS LEARNING:
  • Learn from user feedback
  • Adapt to user preferences
  • Pattern strength tracking
  • Incremental model updates

SIMILARITY DETECTION:
  • Near-duplicate bookmark detection
  • Content similarity analysis
  • Recommendation system
  • Clustering for organization

NETWORK ANALYSIS:
  • Bookmark relationship graphs
  • Community detection
  • Hub identification
  • Collection optimization

AI COMMANDS:
  python cli.py stats --learning                      # Learning statistics
  python cli.py intelligence train data.json          # Train from data
  python cli.py intelligence export --output ai.json  # Export AI data
                """
            },
            
            'import_system': {
                'title': '📥 Import System',
                'content': """
UNIVERSAL IMPORT CAPABILITIES:

GITHUB INTEGRATION:
  • Import starred repositories
  • Import owned repositories
  • Intelligent tag generation
  • Language and framework detection
  • Repository metadata extraction

BROWSER BOOKMARKS:
  • Chrome bookmarks support
  • Firefox bookmarks support
  • Safari bookmarks support
  • Bookmark folder preservation

LINKWARDEN BACKUPS:
  • Native Linkwarden JSON format
  • Collection and tag preservation
  • Metadata retention
  • Relationship maintenance

IMPORT COMMANDS:
  # GitHub import
  python cli.py import --github --github-token TOKEN --github-username USER -o output.json
  
  # Browser bookmarks
  python cli.py import --browser bookmarks.html -o output.json
  
  # Linkwarden backup
  python cli.py import --linkwarden-backup backup.json -o output.json
  
  # Combined import
  python cli.py import --github --browser bookmarks.html --linkwarden-backup backup.json -o combined.json
                """
            },
            
            'interactive_mode': {
                'title': '🎯 Interactive Mode',
                'content': """
INTERACTIVE FEATURES:

SUGGESTION REVIEW:
  • Review AI-generated categories
  • Approve or modify tag suggestions
  • Provide feedback for learning
  • Custom category/tag input

ENHANCEMENT REVIEW:
  • Review scraped metadata
  • Edit enhanced descriptions
  • Approve or reject enhancements
  • Manual content editing

LEARNING FEEDBACK:
  • Track suggestion acceptance/rejection
  • Learn user preferences
  • Adapt future suggestions
  • Show learning progress

INTERACTIVE COMMANDS:
  python cli.py process data.json output.json --interactive
  python cli.py menu                                   # Interactive menu
  
INTERACTIVE MENU OPTIONS:
  1. Process bookmarks with review
  2. Import from sources
  3. View learning statistics
  4. Configuration management
  5. Generate reports
  6. Validate data
  7. Backup & recovery
                """
            },
            
            'reporting': {
                'title': '📊 Reporting & Analytics',
                'content': """
COMPREHENSIVE REPORTING SYSTEM:

OPERATION REPORTS:
  • Before/after comparison
  • Detailed change tracking
  • Statistics and metrics
  • Multiple output formats

PERFORMANCE REPORTS:
  • System performance metrics
  • Operation timing analysis
  • Resource usage tracking
  • Trend analysis

PERIOD REPORTS:
  • Activity over time periods
  • Change frequency analysis
  • Pattern identification
  • Historical trends

REPORT FORMATS:
  • JSON (structured data)
  • HTML (web viewing)
  • CSV (spreadsheet import)
  • Markdown (documentation)

REPORTING COMMANDS:
  # Operation report
  python cli.py report operation before.json after.json --format html
  
  # Performance report
  python cli.py report performance --export-metrics
  
  # Period report
  python cli.py report period --hours 168 --format html
  
  # Statistics
  python cli.py stats --all --export stats.json
                """
            },
            
            'configuration': {
                'title': '⚙️ Configuration',
                'content': """
CONFIGURATION OPTIONS:

SAFETY SETTINGS:
  • max_deletion_percentage: Maximum deletion threshold
  • backup_enabled: Automatic backup creation
  • dry_run_mode: Test mode without changes
  • integrity_checks: Enable integrity validation

AI SETTINGS:
  • enable_ollama: Local LLM integration
  • max_clusters: Maximum clustering groups
  • learning_rate: AI learning speed
  • confidence_threshold: Suggestion confidence

IMPORT SETTINGS:
  • github_token: GitHub API token
  • github_username: GitHub username
  • max_repos: Repository import limit
  • rate_limiting: API rate limiting

CONFIGURATION FILES:
  • config/settings.py: Main configuration
  • config/defaults.py: Default values
  • .env: Environment variables

ENVIRONMENT VARIABLES:
  export GITHUB_TOKEN=your_token
  export GITHUB_USERNAME=your_username
  export OLLAMA_HOST=localhost:11434
                """
            },
            
            'troubleshooting': {
                'title': '🔧 Troubleshooting',
                'content': """
COMMON ISSUES AND SOLUTIONS:

IMPORT ISSUES:
  Problem: GitHub import fails
  Solution: Check token permissions and rate limits
  Command: python cli.py import --github --verbose

  Problem: Browser bookmarks not importing
  Solution: Verify file format and encoding
  Command: python cli.py validate bookmarks.html

PROCESSING ISSUES:
  Problem: Processing takes too long
  Solution: Enable progress monitoring and check thresholds
  Command: python cli.py process data.json output.json --verbose

  Problem: Too many items being deleted
  Solution: Adjust deletion threshold or use dry-run
  Command: python cli.py process data.json output.json --dry-run --max-deletion-percent 5

AI ISSUES:
  Problem: Poor categorization suggestions
  Solution: Train AI with more data or provide feedback
  Command: python cli.py intelligence train training_data.json

  Problem: Ollama not working
  Solution: Check Ollama installation and server status
  Command: python cli.py stats --performance

SAFETY ISSUES:
  Problem: Validation failures
  Solution: Check data integrity and fix issues
  Command: python cli.py validate data.json --fix-issues

  Problem: Backup restoration fails
  Solution: Verify backup integrity and permissions
  Command: python cli.py backup list

DEBUGGING:
  • Use --verbose flag for detailed logging
  • Check log files in logs/ directory
  • Use --dry-run for testing
  • Validate data before processing
                """
            }
        }
    
    def show_help(self, topic: str = 'overview') -> None:
        """Show help for a specific topic"""
        
        if topic not in self.help_topics:
            print(f"❌ Unknown help topic: {topic}")
            print("\nAvailable topics:")
            for topic_name in self.help_topics.keys():
                print(f"  • {topic_name}")
            return
        
        help_info = self.help_topics[topic]
        print(f"\n{help_info['title']}")
        print("=" * len(help_info['title']))
        print(help_info['content'])
    
    def show_all_topics(self) -> None:
        """Show all available help topics"""
        
        print("\n📚 Available Help Topics:")
        print("=" * 30)
        
        for topic_name, topic_info in self.help_topics.items():
            print(f"\n{topic_info['title']}")
            print(f"  Command: python cli.py help {topic_name}")
    
    def search_help(self, search_term: str) -> None:
        """Search help content for a term"""
        
        search_term = search_term.lower()
        matches = []
        
        for topic_name, topic_info in self.help_topics.items():
            content = topic_info['content'].lower()
            title = topic_info['title'].lower()
            
            if search_term in content or search_term in title:
                matches.append((topic_name, topic_info['title']))
        
        if matches:
            print(f"\n🔍 Help topics matching '{search_term}':")
            print("-" * 40)
            
            for topic_name, title in matches:
                print(f"  • {title}")
                print(f"    Command: python cli.py help {topic_name}")
        else:
            print(f"❌ No help topics found matching '{search_term}'")
    
    def show_quick_reference(self) -> None:
        """Show quick reference of common commands"""
        
        print("""
🔖 LINKWARDEN ENHANCER - Quick Reference

BASIC COMMANDS:
  python cli.py process input.json output.json         # Process bookmarks
  python cli.py import --github -o bookmarks.json     # Import from GitHub
  python cli.py validate data.json                    # Validate data
  python cli.py menu                                  # Interactive menu

SAFETY COMMANDS:
  python cli.py backup create data.json               # Create backup
  python cli.py backup restore backup.json data.json  # Restore backup
  python cli.py validate data.json --fix-issues       # Fix validation issues

REPORTING COMMANDS:
  python cli.py stats --all                           # Show all statistics
  python cli.py report operation before.json after.json  # Operation report
  python cli.py report performance                    # Performance report

INTELLIGENCE COMMANDS:
  python cli.py intelligence export --output ai.json  # Export AI data
  python cli.py intelligence train data.json          # Train from data
  python cli.py stats --learning                      # Learning statistics

INTERACTIVE MODE:
  python cli.py process data.json output.json --interactive  # Interactive processing
  python cli.py menu                                         # Interactive menu

HELP COMMANDS:
  python cli.py --help                                # General help
  python cli.py help overview                         # System overview
  python cli.py help getting_started                  # Getting started guide
  python cli.py help troubleshooting                  # Troubleshooting guide

For detailed help on any topic, use: python cli.py help <topic>
        """)


def show_help_command(topic: str = 'overview') -> None:
    """Show help for CLI usage"""
    
    help_system = HelpSystem()
    
    if topic == 'topics':
        help_system.show_all_topics()
    elif topic == 'quick':
        help_system.show_quick_reference()
    elif topic.startswith('search:'):
        search_term = topic[7:]  # Remove 'search:' prefix
        help_system.search_help(search_term)
    else:
        help_system.show_help(topic)