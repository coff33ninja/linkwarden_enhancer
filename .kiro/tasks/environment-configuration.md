# Environment Configuration System

## Task Overview
Implement comprehensive environment configuration management for multi-platform bookmark manager integration with secure credential storage and validation.

## Environment Variables Structure

### 1. Core Platform Configurations

#### Linkwarden Configuration
```bash
# Linkwarden API Settings
LINKWARDEN_URL=http://localhost:3000
LINKWARDEN_API_KEY=your_linkwarden_api_key_here
LINKWARDEN_USERNAME=your_username
LINKWARDEN_TIMEOUT=30
LINKWARDEN_RATE_LIMIT=100

# Linkwarden Features
LINKWARDEN_IMPORT_COLLECTIONS=true
LINKWARDEN_IMPORT_TAGS=true
LINKWARDEN_IMPORT_ARCHIVES=false
LINKWARDEN_PRESERVE_STRUCTURE=true
```

#### Raindrop.io Configuration
```bash
# Raindrop.io API Settings
RAINDROP_API_TOKEN=your_raindrop_token_here
RAINDROP_API_URL=https://api.raindrop.io/rest/v1
RAINDROP_TIMEOUT=30
RAINDROP_RATE_LIMIT=120

# Raindrop.io Features
RAINDROP_IMPORT_COLLECTIONS=true
RAINDROP_IMPORT_TAGS=true
RAINDROP_IMPORT_HIGHLIGHTS=true
RAINDROP_PRESERVE_HIERARCHY=true
```

#### Pocket Configuration
```bash
# Pocket API Settings
POCKET_CONSUMER_KEY=your_pocket_consumer_key
POCKET_ACCESS_TOKEN=your_pocket_access_token
POCKET_API_URL=https://getpocket.com/v3
POCKET_TIMEOUT=30
POCKET_RATE_LIMIT=320

# Pocket Features
POCKET_IMPORT_TAGS=true
POCKET_IMPORT_FAVORITES=true
POCKET_IMPORT_ARCHIVED=true
POCKET_RESOLVE_REDIRECTS=true
```

#### Pinboard Configuration
```bash
# Pinboard API Settings
PINBOARD_API_TOKEN=your_pinboard_token_here
PINBOARD_API_URL=https://api.pinboard.in/v1
PINBOARD_TIMEOUT=30
PINBOARD_RATE_LIMIT=60

# Pinboard Features
PINBOARD_IMPORT_TAGS=true
PINBOARD_IMPORT_DESCRIPTIONS=true
PINBOARD_IMPORT_PRIVATE=true
PINBOARD_IMPORT_UNREAD=true
```

#### Wallabag Configuration
```bash
# Wallabag API Settings
WALLABAG_URL=https://your-wallabag-instance.com
WALLABAG_CLIENT_ID=your_client_id
WALLABAG_CLIENT_SECRET=your_client_secret
WALLABAG_USERNAME=your_username
WALLABAG_PASSWORD=your_password
WALLABAG_TIMEOUT=30

# Wallabag Features
WALLABAG_IMPORT_TAGS=true
WALLABAG_IMPORT_ANNOTATIONS=true
WALLABAG_IMPORT_ARCHIVED=true
WALLABAG_IMPORT_STARRED=true
```

### 2. Enhancement Configuration

#### Data Enhancement Settings
```bash
# Title Enhancement
ENABLE_TITLE_ENHANCEMENT=true
TITLE_QUALITY_THRESHOLD=0.3
TITLE_MAX_LENGTH=100
TITLE_SCRAPING_TIMEOUT=10
TITLE_FALLBACK_TO_URL=true

# Auto-Tagging
ENABLE_AUTO_TAGGING=true
MAX_TAGS_PER_BOOKMARK=10
TAG_CONFIDENCE_THRESHOLD=0.7
ENABLE_DOMAIN_SPECIFIC_TAGGING=true
TAG_SIMILARITY_THRESHOLD=0.8

# Description Generation
ENABLE_DESCRIPTION_GENERATION=true
DESCRIPTION_MAX_LENGTH=200
DESCRIPTION_MIN_LENGTH=50
PREFER_META_DESCRIPTIONS=true
ENABLE_AI_SUMMARIZATION=true
AI_SUMMARY_MODEL=llama2

# Duplicate Detection
ENABLE_DUPLICATE_DETECTION=true
SIMILARITY_THRESHOLD=0.85
URL_NORMALIZATION=true
FUZZY_MATCHING_THRESHOLD=0.9
DUPLICATE_RESOLUTION_STRATEGY=merge
```

#### AI and ML Configuration
```bash
# Ollama Settings
OLLAMA_HOST=localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TIMEOUT=60
OLLAMA_MAX_TOKENS=500

# Content Analysis
ENABLE_CONTENT_ANALYSIS=true
CONTENT_ANALYSIS_TIMEOUT=30
ENABLE_SENTIMENT_ANALYSIS=true
ENABLE_TOPIC_MODELING=true
TOPIC_MODEL_CLUSTERS=20

# Machine Learning
ML_BATCH_SIZE=100
ML_CONFIDENCE_THRESHOLD=0.7
ENABLE_CONTINUOUS_LEARNING=true
LEARNING_RATE=0.1
```

### 3. Sync and Merge Configuration

#### Sync Settings
```bash
# Sync Behavior
DEFAULT_SYNC_STRATEGY=smart-merge
ENABLE_INCREMENTAL_SYNC=true
SYNC_BATCH_SIZE=100
MAX_SYNC_CONFLICTS=50
SYNC_TIMEOUT_MINUTES=30

# Conflict Resolution
DEFAULT_CONFLICT_RESOLUTION=interactive
AUTO_RESOLVE_LOW_RISK_CONFLICTS=true
CONFLICT_CONFIDENCE_THRESHOLD=0.8
ENABLE_CONFLICT_LOGGING=true

# Merge Strategies
MERGE_TITLE_STRATEGY=longest
MERGE_DESCRIPTION_STRATEGY=most_informative
MERGE_TAGS_STRATEGY=union
MERGE_COLLECTIONS_STRATEGY=preserve_hierarchy
```

#### Safety and Backup Settings
```bash
# Safety Settings
BACKUP_BEFORE_SYNC=true
MAX_DELETION_PERCENTAGE=10.0
VALIDATE_SYNC_RESULTS=true
ENABLE_SYNC_ROLLBACK=true
SAFETY_PAUSE_THRESHOLD=100

# Backup Configuration
BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
BACKUP_ENCRYPTION=false
BACKUP_DIRECTORY=./backups
AUTO_BACKUP_FREQUENCY=daily
```

### 4. Performance and Logging

#### Performance Settings
```bash
# Performance Tuning
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=30
RETRY_ATTEMPTS=3
RETRY_DELAY=1.0
ENABLE_CACHING=true
CACHE_TTL_HOURS=24

# Resource Limits
MAX_MEMORY_MB=1024
MAX_WORKERS=4
CHUNK_SIZE=1000
ENABLE_PARALLEL_PROCESSING=true
```

#### Logging Configuration
```bash
# Logging Settings
LOG_LEVEL=INFO
LOG_FILE=linkwarden_enhancer.log
LOG_MAX_SIZE_MB=10
LOG_BACKUP_COUNT=5
ENABLE_CONSOLE_LOGGING=true
ENABLE_FILE_LOGGING=true

# Debug Settings
DEBUG_MODE=false
VERBOSE_LOGGING=false
ENABLE_API_LOGGING=false
LOG_SENSITIVE_DATA=false
```

## Implementation Tasks

### 1. Configuration Manager (`config/config_manager.py`)
```python
class ConfigManager:
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or '.env'
        self.config = {}
        self.validators = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from environment and files"""
        
    def validate_config(self) -> List[ValidationError]:
        """Validate all configuration values"""
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with type conversion"""
        
    def get_platform_config(self, platform: str) -> Dict[str, Any]:
        """Get all configuration for a specific platform"""
        
    def is_platform_configured(self, platform: str) -> bool:
        """Check if platform is properly configured"""
```

### 2. Configuration Validation (`config/validators.py`)
```python
class ConfigValidator:
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def validate_url(self, url: str) -> bool:
        """Validate URL format and accessibility"""
        
    def validate_api_key(self, platform: str, api_key: str) -> bool:
        """Validate API key format and test connection"""
        
    def validate_numeric_range(self, value: Any, min_val: float, max_val: float) -> bool:
        """Validate numeric values are within range"""
        
    def validate_enum_choice(self, value: str, choices: List[str]) -> bool:
        """Validate value is one of allowed choices"""
```

### 3. Configuration Wizard (`config/setup_wizard.py`)
```python
class ConfigurationWizard:
    def __init__(self):
        self.config_manager = ConfigManager()
    
    def run_interactive_setup(self) -> None:
        """Run interactive configuration setup"""
        
    def setup_platform(self, platform: str) -> None:
        """Setup configuration for specific platform"""
        
    def test_platform_connection(self, platform: str) -> bool:
        """Test connection to platform with provided credentials"""
        
    def generate_config_file(self, platforms: List[str]) -> None:
        """Generate .env file with platform configurations"""
```

### 4. Enhanced .env.example
```bash
# =============================================================================
# LINKWARDEN ENHANCER CONFIGURATION
# =============================================================================
# Copy this file to .env and fill in your actual values
# All settings are optional unless marked as REQUIRED

# =============================================================================
# PLATFORM CONFIGURATIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Linkwarden (REQUIRED for Linkwarden integration)
# -----------------------------------------------------------------------------
LINKWARDEN_URL=http://localhost:3000
LINKWARDEN_API_KEY=your_linkwarden_api_key_here
LINKWARDEN_USERNAME=your_username
LINKWARDEN_TIMEOUT=30
LINKWARDEN_RATE_LIMIT=100
LINKWARDEN_IMPORT_COLLECTIONS=true
LINKWARDEN_IMPORT_TAGS=true
LINKWARDEN_IMPORT_ARCHIVES=false
LINKWARDEN_PRESERVE_STRUCTURE=true

# -----------------------------------------------------------------------------
# Raindrop.io (Optional - for Raindrop.io integration)
# -----------------------------------------------------------------------------
# RAINDROP_API_TOKEN=your_raindrop_token_here
# RAINDROP_API_URL=https://api.raindrop.io/rest/v1
# RAINDROP_TIMEOUT=30
# RAINDROP_RATE_LIMIT=120
# RAINDROP_IMPORT_COLLECTIONS=true
# RAINDROP_IMPORT_TAGS=true
# RAINDROP_IMPORT_HIGHLIGHTS=true

# -----------------------------------------------------------------------------
# Pocket (Optional - for Pocket integration)
# -----------------------------------------------------------------------------
# POCKET_CONSUMER_KEY=your_pocket_consumer_key
# POCKET_ACCESS_TOKEN=your_pocket_access_token
# POCKET_API_URL=https://getpocket.com/v3
# POCKET_TIMEOUT=30
# POCKET_RATE_LIMIT=320
# POCKET_IMPORT_TAGS=true
# POCKET_IMPORT_FAVORITES=true

# -----------------------------------------------------------------------------
# Pinboard (Optional - for Pinboard integration)
# -----------------------------------------------------------------------------
# PINBOARD_API_TOKEN=your_pinboard_token_here
# PINBOARD_API_URL=https://api.pinboard.in/v1
# PINBOARD_TIMEOUT=30
# PINBOARD_RATE_LIMIT=60
# PINBOARD_IMPORT_TAGS=true
# PINBOARD_IMPORT_DESCRIPTIONS=true

# -----------------------------------------------------------------------------
# Wallabag (Optional - for Wallabag integration)
# -----------------------------------------------------------------------------
# WALLABAG_URL=https://your-wallabag-instance.com
# WALLABAG_CLIENT_ID=your_client_id
# WALLABAG_CLIENT_SECRET=your_client_secret
# WALLABAG_USERNAME=your_username
# WALLABAG_PASSWORD=your_password
# WALLABAG_TIMEOUT=30

# =============================================================================
# ENHANCEMENT SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# Data Enhancement
# -----------------------------------------------------------------------------
ENABLE_TITLE_ENHANCEMENT=true
TITLE_QUALITY_THRESHOLD=0.3
TITLE_MAX_LENGTH=100
TITLE_SCRAPING_TIMEOUT=10

ENABLE_AUTO_TAGGING=true
MAX_TAGS_PER_BOOKMARK=10
TAG_CONFIDENCE_THRESHOLD=0.7
ENABLE_DOMAIN_SPECIFIC_TAGGING=true

ENABLE_DESCRIPTION_GENERATION=true
DESCRIPTION_MAX_LENGTH=200
PREFER_META_DESCRIPTIONS=true
ENABLE_AI_SUMMARIZATION=true

ENABLE_DUPLICATE_DETECTION=true
SIMILARITY_THRESHOLD=0.85
DUPLICATE_RESOLUTION_STRATEGY=merge

# -----------------------------------------------------------------------------
# AI and Machine Learning
# -----------------------------------------------------------------------------
OLLAMA_HOST=localhost:11434
OLLAMA_MODEL=llama2
OLLAMA_TIMEOUT=60
OLLAMA_MAX_TOKENS=500

ENABLE_CONTENT_ANALYSIS=true
ENABLE_CONTINUOUS_LEARNING=true
ML_CONFIDENCE_THRESHOLD=0.7
LEARNING_RATE=0.1

# =============================================================================
# SYNC AND MERGE SETTINGS
# =============================================================================

# -----------------------------------------------------------------------------
# Sync Configuration
# -----------------------------------------------------------------------------
DEFAULT_SYNC_STRATEGY=smart-merge
ENABLE_INCREMENTAL_SYNC=true
SYNC_BATCH_SIZE=100
MAX_SYNC_CONFLICTS=50
SYNC_TIMEOUT_MINUTES=30

DEFAULT_CONFLICT_RESOLUTION=interactive
AUTO_RESOLVE_LOW_RISK_CONFLICTS=true
CONFLICT_CONFIDENCE_THRESHOLD=0.8

# -----------------------------------------------------------------------------
# Safety and Backup
# -----------------------------------------------------------------------------
BACKUP_BEFORE_SYNC=true
MAX_DELETION_PERCENTAGE=10.0
VALIDATE_SYNC_RESULTS=true
ENABLE_SYNC_ROLLBACK=true

BACKUP_RETENTION_DAYS=30
BACKUP_COMPRESSION=true
BACKUP_DIRECTORY=./backups
AUTO_BACKUP_FREQUENCY=daily

# =============================================================================
# PERFORMANCE AND LOGGING
# =============================================================================

# -----------------------------------------------------------------------------
# Performance Settings
# -----------------------------------------------------------------------------
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=30
RETRY_ATTEMPTS=3
RETRY_DELAY=1.0
ENABLE_CACHING=true
CACHE_TTL_HOURS=24

MAX_MEMORY_MB=1024
MAX_WORKERS=4
CHUNK_SIZE=1000
ENABLE_PARALLEL_PROCESSING=true

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_LEVEL=INFO
LOG_FILE=linkwarden_enhancer.log
LOG_MAX_SIZE_MB=10
LOG_BACKUP_COUNT=5
ENABLE_CONSOLE_LOGGING=true
ENABLE_FILE_LOGGING=true

DEBUG_MODE=false
VERBOSE_LOGGING=false
ENABLE_API_LOGGING=false
LOG_SENSITIVE_DATA=false

# =============================================================================
# DATA DIRECTORIES
# =============================================================================
DATA_DIR=data
BACKUP_DIR=backups
CACHE_DIR=cache
MODELS_DIR=models
REPORTS_DIR=reports
LOGS_DIR=logs
```

### 5. CLI Configuration Commands
```bash
# Configuration setup wizard
linkwarden-enhancer config setup
linkwarden-enhancer config setup --platform linkwarden
linkwarden-enhancer config setup --platform raindrop

# Configuration validation
linkwarden-enhancer config validate
linkwarden-enhancer config validate --platform linkwarden
linkwarden-enhancer config test-connections

# Configuration management
linkwarden-enhancer config show
linkwarden-enhancer config show --platform linkwarden
linkwarden-enhancer config export --output config_backup.json
linkwarden-enhancer config import --input config_backup.json

# Platform-specific configuration
linkwarden-enhancer config platforms --list
linkwarden-enhancer config platforms --available
linkwarden-enhancer config platforms --configured
```

## Security Considerations

### 1. Credential Security
```python
class CredentialManager:
    def __init__(self):
        self.encryption_key = self._get_or_create_key()
    
    def encrypt_credential(self, credential: str) -> str:
        """Encrypt sensitive credential"""
        
    def decrypt_credential(self, encrypted_credential: str) -> str:
        """Decrypt credential for use"""
        
    def store_credential_securely(self, platform: str, credential_type: str, value: str) -> None:
        """Store credential with encryption"""
```

### 2. Environment Validation
```python
class SecurityValidator:
    def validate_environment_security(self) -> List[SecurityIssue]:
        """Check for security issues in configuration"""
        issues = []
        
        # Check for credentials in plain text
        if self._has_plaintext_credentials():
            issues.append(SecurityIssue("Credentials stored in plain text"))
        
        # Check file permissions
        if not self._check_file_permissions():
            issues.append(SecurityIssue("Insecure file permissions on .env"))
        
        # Check for credential leakage
        if self._check_credential_leakage():
            issues.append(SecurityIssue("Potential credential leakage detected"))
        
        return issues
```

## Testing Strategy

### 1. Configuration Tests
```python
class TestConfigManager:
    def test_load_config_from_env(self):
        # Test loading from environment variables
        
    def test_load_config_from_file(self):
        # Test loading from .env file
        
    def test_config_validation(self):
        # Test configuration validation
        
    def test_platform_config_isolation(self):
        # Test platform configurations don't interfere
```

### 2. Integration Tests
```python
class TestConfigIntegration:
    def test_api_connection_with_config(self):
        # Test API connections using configuration
        
    def test_config_wizard_flow(self):
        # Test interactive configuration setup
        
    def test_config_migration(self):
        # Test configuration format migration
```

## Success Criteria
1. **Easy Setup**: New users can configure in <5 minutes
2. **Validation**: 100% of invalid configurations caught before use
3. **Security**: No credentials stored in plain text by default
4. **Flexibility**: Support for multiple configuration sources
5. **Documentation**: Clear documentation for all settings
6. **Testing**: Automated validation of all configuration options
7. **Migration**: Seamless upgrade path for configuration changes

## Future Enhancements
- GUI configuration interface
- Cloud-based configuration sync
- Team/organization configuration templates
- Advanced credential management with external vaults
- Configuration versioning and rollback