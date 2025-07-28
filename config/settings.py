"""Configuration loading and management"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from config.defaults import DEFAULT_CONFIG


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from multiple sources with precedence:
    1. Command line arguments (handled in main.py)
    2. Custom config file
    3. Environment variables
    4. Default configuration
    """
    # Load environment variables
    load_dotenv()
    
    # Start with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    config = _apply_env_overrides(config)
    
    # Override with config file if provided
    if config_file:
        config = _apply_config_file(config, config_file)
    
    # Ensure directories exist
    _ensure_directories(config)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get the default configuration"""
    return DEFAULT_CONFIG.copy()


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides"""
    
    # GitHub configuration
    if os.getenv('GITHUB_TOKEN'):
        config['github']['token'] = os.getenv('GITHUB_TOKEN')
    if os.getenv('GITHUB_USERNAME'):
        config['github']['username'] = os.getenv('GITHUB_USERNAME')
    
    # Ollama configuration
    if os.getenv('OLLAMA_HOST'):
        config['ai']['ollama_host'] = os.getenv('OLLAMA_HOST')
    if os.getenv('OLLAMA_MODEL'):
        config['ai']['ollama_model'] = os.getenv('OLLAMA_MODEL')
    
    # AI configuration
    if os.getenv('ENABLE_AI_ANALYSIS'):
        config['ai']['enable_ai_analysis'] = os.getenv('ENABLE_AI_ANALYSIS').lower() == 'true'
    if os.getenv('SIMILARITY_THRESHOLD'):
        config['ai']['similarity_threshold'] = float(os.getenv('SIMILARITY_THRESHOLD'))
    if os.getenv('MAX_CLUSTERS'):
        config['ai']['max_clusters'] = int(os.getenv('MAX_CLUSTERS'))
    
    # Safety configuration
    if os.getenv('DRY_RUN_MODE'):
        config['safety']['dry_run_mode'] = os.getenv('DRY_RUN_MODE').lower() == 'true'
    if os.getenv('MAX_DELETION_PERCENTAGE'):
        config['safety']['max_deletion_percentage'] = float(os.getenv('MAX_DELETION_PERCENTAGE'))
    if os.getenv('BACKUP_RETENTION_COUNT'):
        config['safety']['backup_retention_count'] = int(os.getenv('BACKUP_RETENTION_COUNT'))
    
    # Logging configuration
    if os.getenv('LOG_LEVEL'):
        config['logging']['level'] = os.getenv('LOG_LEVEL')
    if os.getenv('LOG_FILE'):
        config['logging']['file'] = os.getenv('LOG_FILE')
    
    # Directory configuration
    if os.getenv('DATA_DIR'):
        config['directories']['data_dir'] = os.getenv('DATA_DIR')
    if os.getenv('BACKUP_DIR'):
        config['directories']['backup_dir'] = os.getenv('BACKUP_DIR')
    if os.getenv('CACHE_DIR'):
        config['directories']['cache_dir'] = os.getenv('CACHE_DIR')
    if os.getenv('MODELS_DIR'):
        config['directories']['models_dir'] = os.getenv('MODELS_DIR')
    
    return config


def _apply_config_file(config: Dict[str, Any], config_file: str) -> Dict[str, Any]:
    """Apply configuration from JSON file"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            file_config = json.load(f)
        
        # Deep merge configuration
        config = _deep_merge(config, file_config)
        
    except FileNotFoundError:
        print(f"Warning: Config file {config_file} not found, using defaults")
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in config file {config_file}: {e}")
    except Exception as e:
        print(f"Warning: Error loading config file {config_file}: {e}")
    
    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def _ensure_directories(config: Dict[str, Any]) -> None:
    """Ensure all configured directories exist"""
    directories = config.get('directories', {})
    
    for dir_key, dir_path in directories.items():
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_config(config: Dict[str, Any], config_file: str) -> None:
    """Save configuration to JSON file"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise Exception(f"Failed to save config to {config_file}: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration values"""
    errors = []
    
    # Validate safety configuration
    safety = config.get('safety', {})
    if safety.get('max_deletion_percentage', 0) > 100:
        errors.append("max_deletion_percentage cannot exceed 100")
    if safety.get('backup_retention_count', 0) < 1:
        errors.append("backup_retention_count must be at least 1")
    
    # Validate AI configuration
    ai = config.get('ai', {})
    if not 0 < ai.get('similarity_threshold', 0.85) <= 1:
        errors.append("similarity_threshold must be between 0 and 1")
    if ai.get('max_clusters', 50) < 1:
        errors.append("max_clusters must be at least 1")
    
    # Validate GitHub configuration
    github = config.get('github', {})
    if github.get('import_starred') or github.get('import_owned_repos'):
        if not github.get('token'):
            errors.append("GitHub token required for GitHub import")
        if not github.get('username'):
            errors.append("GitHub username required for GitHub import")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True