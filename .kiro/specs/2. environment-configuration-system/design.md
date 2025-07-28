# Design Document

## Overview

The environment configuration system provides comprehensive configuration management for multi-platform bookmark integration with secure credential storage, validation, and management. The system supports multiple bookmark platforms, enhancement settings, sync configurations, and performance tuning through a unified configuration interface with interactive setup wizards and robust validation.

## Architecture

### Enhanced Configuration System (Building on Existing)

**Current Architecture Integration:**
- **Extends existing `config/` package** with advanced configuration management
- **Leverages existing `cli/` system** for interactive setup commands
- **Integrates with existing safety features** for secure credential storage

```
# EXISTING MODULES (✅ Already implemented)
config/
├── __init__.py                  # ✅ Basic configuration package
├── settings.py                  # ✅ Configuration loading and management
└── defaults.py                  # ✅ Default configuration values

cli/
├── main_cli.py                  # ✅ Main CLI with extensive commands
├── interactive.py               # ✅ Interactive components
└── help_system.py               # ✅ Help system

# NEW ADDITIONS (❌ To be added)
config/
├── config_manager.py            # ❌ Enhanced configuration management
├── validators.py                # ❌ Configuration validation
├── setup_wizard.py              # ❌ Interactive configuration setup
├── credential_manager.py        # ❌ Secure credential storage
├── platform_configs.py         # ❌ Platform-specific configurations
├── environment_detector.py      # ❌ Environment detection
└── templates/
    ├── config_templates.py      # ❌ Configuration templates
    └── platform_templates.py    # ❌ Platform-specific templates

# SECURITY ENHANCEMENTS (❌ New package)
security/
├── __init__.py                  # ❌ Security package
├── encryption.py               # ❌ Credential encryption
├── security_validator.py       # ❌ Security validation
└── permissions.py              # ❌ File permission management
```

### Configuration Flow

```
Environment Variables → Configuration Loader → Validation Engine → Platform Clients
         ↓                      ↓                     ↓                    ↓
    .env Files          Config Manager         Validators         API Connections
         ↓                      ↓                     ↓                    ↓
  Config Templates      Credential Manager    Security Checks    Connection Tests
```

## Components and Interfaces

### Configuration Manager

```python
class ConfigManager:
    def __init__(self, config_file: Optional[str] = None, 
                 environment: Optional[str] = None):
        self.config_file = config_file or '.env'
        self.environment = environment or self._detect_environment()
        self.config = {}
        self.validators = ConfigValidators()
        self.credential_manager = CredentialManager()
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from multiple sources with precedence"""
        
    def get(self, key: str, default: Any = None, 
           config_type: Type = str) -> Any:
        """Get configuration value with type conversion and validation"""
        
    def get_platform_config(self, platform: str) -> PlatformConfig:
        """Get complete configuration for a specific platform"""
        
    def is_platform_configured(self, platform: str) -> bool:
        """Check if platform is properly configured"""
        
    def validate_all(self) -> ValidationResult:
        """Validate all configuration values"""
        
    def export_config(self, exclude_sensitive: bool = True) -> Dict[str, Any]:
        """Export configuration for backup or sharing"""
        
    def import_config(self, config_data: Dict[str, Any], 
                     merge: bool = True) -> None:
        """Import configuration from external source"""

class PlatformConfig:
    def __init__(self, platform: str, config_data: Dict[str, Any]):
        self.platform = platform
        self.config_data = config_data
        
    @property
    def is_configured(self) -> bool:
        """Check if platform has all required configuration"""
        
    @property
    def api_client_config(self) -> Dict[str, Any]:
        """Get configuration formatted for API client"""
        
    def test_connection(self) -> ConnectionTestResult:
        """Test API connection with current configuration"""
```

### Configuration Validators

```python
class ConfigValidators:
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.platform_validators = self._load_platform_validators()
    
    def validate_url(self, url: str) -> ValidationResult:
        """Validate URL format and accessibility"""
        
    def validate_api_key(self, platform: str, api_key: str) -> ValidationResult:
        """Validate API key format and test connection"""
        
    def validate_numeric_range(self, value: Any, min_val: float, 
                              max_val: float) -> ValidationResult:
        """Validate numeric values are within acceptable range"""
        
    def validate_enum_choice(self, value: str, 
                           choices: List[str]) -> ValidationResult:
        """Validate value is one of allowed choices"""
        
    def validate_platform_config(self, platform: str, 
                                config: Dict[str, Any]) -> ValidationResult:
        """Validate complete platform configuration"""

class ValidationResult:
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.suggestions = []
    
    def add_error(self, field: str, message: str, 
                  suggestion: Optional[str] = None) -> None:
        """Add validation error with optional suggestion"""
        
    def add_warning(self, field: str, message: str) -> None:
        """Add validation warning"""
        
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge validation results"""
```

### Interactive Setup Wizard

```python
class ConfigurationWizard:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.platform_configs = {}
        self.enhancement_config = {}
        self.performance_config = {}
    
    def run_interactive_setup(self) -> SetupResult:
        """Run complete interactive configuration setup"""
        
    def setup_platforms(self) -> Dict[str, PlatformConfig]:
        """Interactive setup for all bookmark platforms"""
        
    def setup_platform(self, platform: str) -> PlatformConfig:
        """Setup configuration for specific platform"""
        
    def setup_enhancement_options(self) -> EnhancementConfig:
        """Setup enhancement and AI configuration"""
        
    def setup_performance_options(self) -> PerformanceConfig:
        """Setup performance and resource configuration"""
        
    def test_all_connections(self) -> Dict[str, ConnectionTestResult]:
        """Test connections to all configured platforms"""
        
    def generate_config_file(self) -> None:
        """Generate final .env file with all configurations"""

class InteractivePrompt:
    @staticmethod
    def prompt_text(message: str, default: Optional[str] = None, 
                   required: bool = True) -> str:
        """Prompt for text input with validation"""
        
    @staticmethod
    def prompt_choice(message: str, choices: List[str], 
                     default: Optional[str] = None) -> str:
        """Prompt for choice from list of options"""
        
    @staticmethod
    def prompt_boolean(message: str, default: bool = False) -> bool:
        """Prompt for yes/no input"""
        
    @staticmethod
    def prompt_numeric(message: str, min_val: Optional[float] = None,
                      max_val: Optional[float] = None,
                      default: Optional[float] = None) -> float:
        """Prompt for numeric input with range validation"""
```

### Credential Management

```python
class CredentialManager:
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_credential(self, credential: str) -> str:
        """Encrypt sensitive credential for storage"""
        
    def decrypt_credential(self, encrypted_credential: str) -> str:
        """Decrypt credential for use"""
        
    def store_credential_securely(self, platform: str, 
                                 credential_type: str, 
                                 value: str) -> None:
        """Store credential with encryption"""
        
    def get_credential(self, platform: str, 
                      credential_type: str) -> Optional[str]:
        """Retrieve and decrypt credential"""
        
    def rotate_encryption_key(self) -> None:
        """Rotate encryption key and re-encrypt all credentials"""
        
    def export_credentials(self, include_encrypted: bool = False) -> Dict[str, Any]:
        """Export credentials for backup (encrypted or placeholder)"""

class SecurityValidator:
    def validate_environment_security(self) -> List[SecurityIssue]:
        """Check for security issues in configuration"""
        
    def check_file_permissions(self, file_path: str) -> bool:
        """Validate file permissions are secure"""
        
    def detect_credential_leakage(self) -> List[SecurityIssue]:
        """Detect potential credential exposure"""
        
    def recommend_security_improvements(self) -> List[SecurityRecommendation]:
        """Provide security improvement recommendations"""

class SecurityIssue:
    def __init__(self, severity: str, message: str, 
                 recommendation: str, affected_files: List[str] = None):
        self.severity = severity  # 'low', 'medium', 'high', 'critical'
        self.message = message
        self.recommendation = recommendation
        self.affected_files = affected_files or []
```

### Platform-Specific Configurations

```python
class PlatformConfigurations:
    @staticmethod
    def get_linkwarden_config() -> PlatformConfigSchema:
        """Get Linkwarden configuration schema"""
        return PlatformConfigSchema(
            platform="linkwarden",
            required_fields={
                "LINKWARDEN_URL": {"type": "url", "description": "Linkwarden instance URL"},
                "LINKWARDEN_API_KEY": {"type": "api_key", "description": "API key for authentication", "sensitive": True}
            },
            optional_fields={
                "LINKWARDEN_USERNAME": {"type": "string", "description": "Username for reference"},
                "LINKWARDEN_TIMEOUT": {"type": "integer", "default": 30, "min": 5, "max": 300},
                "LINKWARDEN_RATE_LIMIT": {"type": "integer", "default": 100, "min": 1, "max": 1000}
            },
            feature_flags={
                "LINKWARDEN_IMPORT_COLLECTIONS": {"type": "boolean", "default": True},
                "LINKWARDEN_IMPORT_TAGS": {"type": "boolean", "default": True},
                "LINKWARDEN_PRESERVE_STRUCTURE": {"type": "boolean", "default": True}
            }
        )
    
    @staticmethod
    def get_raindrop_config() -> PlatformConfigSchema:
        """Get Raindrop.io configuration schema"""
        
    @staticmethod
    def get_pocket_config() -> PlatformConfigSchema:
        """Get Pocket configuration schema"""
        
    @staticmethod
    def get_pinboard_config() -> PlatformConfigSchema:
        """Get Pinboard configuration schema"""
        
    @staticmethod
    def get_wallabag_config() -> PlatformConfigSchema:
        """Get Wallabag configuration schema"""

class PlatformConfigSchema:
    def __init__(self, platform: str, required_fields: Dict[str, Any],
                 optional_fields: Dict[str, Any] = None,
                 feature_flags: Dict[str, Any] = None):
        self.platform = platform
        self.required_fields = required_fields
        self.optional_fields = optional_fields or {}
        self.feature_flags = feature_flags or {}
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against schema"""
        
    def get_connection_test_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration needed for connection testing"""
```

### Environment Detection and Optimization

```python
class EnvironmentDetector:
    def __init__(self):
        self.environment_indicators = self._load_environment_indicators()
        self.resource_detector = ResourceDetector()
    
    def detect_environment(self) -> EnvironmentInfo:
        """Detect current environment (development, testing, production)"""
        
    def detect_resources(self) -> ResourceInfo:
        """Detect available system resources"""
        
    def recommend_configuration(self, environment: EnvironmentInfo,
                              resources: ResourceInfo) -> ConfigRecommendations:
        """Recommend optimal configuration for environment and resources"""

class EnvironmentInfo:
    def __init__(self):
        self.environment_type = "development"  # development, testing, production
        self.container_environment = False
        self.cloud_environment = None
        self.network_quality = "good"  # poor, fair, good, excellent
        self.security_requirements = "standard"  # minimal, standard, high, critical

class ResourceInfo:
    def __init__(self):
        self.cpu_cores = 1
        self.memory_gb = 1.0
        self.disk_space_gb = 10.0
        self.network_bandwidth_mbps = 10.0

class ConfigRecommendations:
    def __init__(self):
        self.performance_settings = {}
        self.security_settings = {}
        self.logging_settings = {}
        self.resource_limits = {}
        self.reasoning = []
    
    def apply_to_config(self, config_manager: ConfigManager) -> None:
        """Apply recommendations to configuration"""
```

## Data Models

### Configuration Schemas

```python
@dataclass
class EnhancementConfig:
    # Title Enhancement
    enable_title_enhancement: bool = True
    title_quality_threshold: float = 0.3
    title_max_length: int = 100
    title_scraping_timeout: int = 10
    title_fallback_to_url: bool = True
    
    # Auto-Tagging
    enable_auto_tagging: bool = True
    max_tags_per_bookmark: int = 10
    tag_confidence_threshold: float = 0.7
    enable_domain_specific_tagging: bool = True
    tag_similarity_threshold: float = 0.8
    
    # Description Generation
    enable_description_generation: bool = True
    description_max_length: int = 200
    description_min_length: int = 50
    prefer_meta_descriptions: bool = True
    enable_ai_summarization: bool = True
    ai_summary_model: str = "llama2"
    
    # Duplicate Detection
    enable_duplicate_detection: bool = True
    similarity_threshold: float = 0.85
    url_normalization: bool = True
    fuzzy_matching_threshold: float = 0.9
    duplicate_resolution_strategy: str = "merge"

@dataclass
class SyncConfig:
    # Sync Behavior
    default_sync_strategy: str = "smart-merge"
    enable_incremental_sync: bool = True
    sync_batch_size: int = 100
    max_sync_conflicts: int = 50
    sync_timeout_minutes: int = 30
    
    # Conflict Resolution
    default_conflict_resolution: str = "interactive"
    auto_resolve_low_risk_conflicts: bool = True
    conflict_confidence_threshold: float = 0.8
    enable_conflict_logging: bool = True
    
    # Safety Settings
    backup_before_sync: bool = True
    max_deletion_percentage: float = 10.0
    validate_sync_results: bool = True
    enable_sync_rollback: bool = True
    safety_pause_threshold: int = 100

@dataclass
class PerformanceConfig:
    # Concurrency
    max_concurrent_requests: int = 5
    max_workers: int = 4
    enable_parallel_processing: bool = True
    
    # Timeouts and Retries
    request_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    cache_max_size_mb: int = 100
    
    # Resource Limits
    max_memory_mb: int = 1024
    chunk_size: int = 1000
    batch_size: int = 100

@dataclass
class LoggingConfig:
    # Logging Levels
    log_level: str = "INFO"
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    
    # File Logging
    log_file: str = "linkwarden_enhancer.log"
    log_max_size_mb: int = 10
    log_backup_count: int = 5
    
    # Debug Settings
    debug_mode: bool = False
    verbose_logging: bool = False
    enable_api_logging: bool = False
    log_sensitive_data: bool = False
```

### Configuration Templates

```python
class ConfigTemplates:
    @staticmethod
    def get_development_template() -> Dict[str, Any]:
        """Get configuration template optimized for development"""
        return {
            "environment": "development",
            "performance": {
                "max_concurrent_requests": 2,
                "enable_caching": True,
                "debug_mode": True
            },
            "logging": {
                "log_level": "DEBUG",
                "verbose_logging": True,
                "enable_console_logging": True
            },
            "security": {
                "encryption_enabled": False,  # For easier development
                "validate_certificates": False
            }
        }
    
    @staticmethod
    def get_production_template() -> Dict[str, Any]:
        """Get configuration template optimized for production"""
        return {
            "environment": "production",
            "performance": {
                "max_concurrent_requests": 10,
                "enable_caching": True,
                "max_memory_mb": 2048
            },
            "logging": {
                "log_level": "INFO",
                "enable_file_logging": True,
                "log_sensitive_data": False
            },
            "security": {
                "encryption_enabled": True,
                "validate_certificates": True,
                "secure_file_permissions": True
            }
        }
    
    @staticmethod
    def get_minimal_template() -> Dict[str, Any]:
        """Get minimal configuration template"""
        
    @staticmethod
    def get_comprehensive_template() -> Dict[str, Any]:
        """Get comprehensive configuration template with all options"""
```

## CLI Integration

### Configuration Commands

```python
class ConfigCommands:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.wizard = ConfigurationWizard(config_manager)
    
    def setup_command(self, platform: Optional[str] = None) -> None:
        """Run interactive configuration setup"""
        
    def validate_command(self, platform: Optional[str] = None) -> None:
        """Validate configuration"""
        
    def test_connections_command(self) -> None:
        """Test all platform connections"""
        
    def show_command(self, platform: Optional[str] = None) -> None:
        """Show current configuration"""
        
    def export_command(self, output_file: str, 
                      exclude_sensitive: bool = True) -> None:
        """Export configuration to file"""
        
    def import_command(self, input_file: str, merge: bool = True) -> None:
        """Import configuration from file"""
        
    def platforms_command(self, action: str) -> None:
        """Manage platform configurations"""
```

### Example CLI Usage

```bash
# Interactive setup wizard
linkwarden-enhancer config setup
linkwarden-enhancer config setup --platform linkwarden

# Configuration validation
linkwarden-enhancer config validate
linkwarden-enhancer config test-connections

# Configuration management
linkwarden-enhancer config show
linkwarden-enhancer config export --output config_backup.json
linkwarden-enhancer config import --input config_backup.json

# Platform management
linkwarden-enhancer config platforms --list
linkwarden-enhancer config platforms --available
linkwarden-enhancer config platforms --configured
```

## Security Implementation

### Encryption and Security

```python
class EncryptionManager:
    def __init__(self, key_file: str = ".encryption_key"):
        self.key_file = key_file
        self.key = self._load_or_create_key()
        self.cipher = Fernet(self.key)
    
    def _load_or_create_key(self) -> bytes:
        """Load existing key or create new one"""
        
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        
    def rotate_key(self, old_key: bytes) -> None:
        """Rotate encryption key and re-encrypt data"""

class FilePermissionManager:
    @staticmethod
    def set_secure_permissions(file_path: str) -> None:
        """Set secure file permissions (600 on Unix, restricted on Windows)"""
        
    @staticmethod
    def validate_permissions(file_path: str) -> bool:
        """Validate file has secure permissions"""
        
    @staticmethod
    def fix_permissions(file_path: str) -> None:
        """Fix insecure file permissions"""
```

## Testing Strategy

### Unit Tests

```python
class TestConfigManager:
    def test_load_config_from_env(self):
        """Test loading configuration from environment variables"""
        
    def test_load_config_from_file(self):
        """Test loading configuration from .env file"""
        
    def test_config_validation(self):
        """Test configuration validation"""
        
    def test_platform_config_isolation(self):
        """Test platform configurations don't interfere"""

class TestConfigValidators:
    def test_url_validation(self):
        """Test URL format validation"""
        
    def test_api_key_validation(self):
        """Test API key format validation"""
        
    def test_numeric_range_validation(self):
        """Test numeric range validation"""

class TestCredentialManager:
    def test_credential_encryption(self):
        """Test credential encryption/decryption"""
        
    def test_secure_storage(self):
        """Test secure credential storage"""
        
    def test_key_rotation(self):
        """Test encryption key rotation"""
```

### Integration Tests

```python
class TestConfigIntegration:
    def test_wizard_flow(self):
        """Test complete configuration wizard flow"""
        
    def test_api_connections(self):
        """Test API connections with configuration"""
        
    def test_config_migration(self):
        """Test configuration format migration"""
        
    def test_security_validation(self):
        """Test security validation and recommendations"""
```

## Success Criteria

1. **Easy Setup**: New users can configure system in <5 minutes using wizard
2. **Validation**: 100% of invalid configurations caught before use
3. **Security**: No credentials stored in plain text by default
4. **Flexibility**: Support for multiple configuration sources and environments
5. **Platform Support**: Full configuration support for 5+ bookmark platforms
6. **Performance**: Configuration loading and validation in <1 second
7. **User Experience**: Clear error messages and helpful suggestions for all validation failures
8. **Security**: Comprehensive security validation with actionable recommendations