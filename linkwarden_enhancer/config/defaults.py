"""Default configuration values for Linkwarden Enhancer"""

import os

DEFAULT_CONFIG = {
    'safety': {
        'dry_run_mode': False,
        'max_deletion_percentage': 10.0,
        'max_items_deleted': 100,
        'backup_retention_count': 5,
        'require_confirmation_threshold': 50,
        'enable_real_time_monitoring': True,
        'checkpoint_frequency': 1000,
    },
    
    'ai': {
        'enable_ai_analysis': True,
        'ollama_model': 'llama2',
        'ollama_host': 'localhost:11434',
        'similarity_threshold': 0.85,
        'max_clusters': 50,
        'enable_content_analysis': True,
        'enable_smart_tagging': True,
        'enable_duplicate_detection': True,
        'enable_clustering': True,
        'batch_size': 100,
        'use_gpu': False,
    },
    
    'github': {
        'token': os.getenv('GITHUB_TOKEN'),
        'username': os.getenv('GITHUB_USERNAME'),
        'import_starred': True,
        'import_owned_repos': True,
        'rate_limit_requests_per_hour': 5000,
        'cache': {
            'enabled': True,
            'ttl_hours': 24,  # Cache for 24 hours
            'force_refresh': False,
        },
    },
    
    'scraping': {
        'enable_scraping': True,
        'timeout_seconds': 30,
        'max_retries': 3,
        'retry_delay': 1.0,
        'user_agent': 'Linkwarden-Enhancer/0.1.0',
        'enable_javascript': False,
        'cache_enabled': True,
        'cache_ttl_hours': 24,
    },
    
    'intelligence': {
        'enable_smart_dictionaries': True,
        'enable_continuous_learning': True,
        'pattern_strength_threshold': 0.7,
        'learning_rate': 0.1,
        'max_learned_patterns': 10000,
    },
    
    'directories': {
        'data_dir': 'data',
        'backup_dir': 'backups',
        'cache_dir': 'cache',
        'models_dir': 'models',
        'logs_dir': 'logs',
    },
    
    'logging': {
        'level': 'INFO',
        'file': 'linkwarden_enhancer.log',
        'max_file_size_mb': 10,
        'backup_count': 5,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    },
    
    'performance': {
        'max_workers': 4,
        'chunk_size': 1000,
        'memory_limit_mb': 1024,
        'enable_parallel_processing': True,
    }
}