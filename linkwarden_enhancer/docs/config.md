# Config Module

The `config` module is responsible for managing the configuration of the Linkwarden Enhancer application. It provides a way to load, validate, and access configuration settings from various sources.

## Modules

### Defaults (`defaults.py`)

The `defaults.py` script contains the default configuration values for the application. This ensures that the application has a working configuration out of the box, even if no custom configuration is provided.

**Key Features:**

- **Centralized Defaults:** Provides a single source of truth for all default configuration values.
- **Structured Configuration:** The default configuration is organized into sections (e.g., `safety`, `ai`, `github`) for better readability and maintainability.

### Settings (`settings.py`)

The `settings.py` script provides the main logic for loading and managing the application's configuration. It supports loading configuration from multiple sources with a clear precedence order.

**Functions:**

- **`load_config(config_file)`**: Loads the configuration from the specified sources. It follows a specific precedence order:
    1. Command-line arguments (handled in `main_cli.py`)
    2. Custom config file (if provided)
    3. Environment variables
    4. Default configuration (`defaults.py`)
- **`get_default_config()`**: Returns a copy of the default configuration dictionary.
- **`_apply_env_overrides(config)`**: A helper function that applies configuration overrides from environment variables.
- **`_apply_config_file(config, config_file)`**: A helper function that applies configuration overrides from a JSON config file.
- **`_deep_merge(base, override)`**: A helper function that recursively merges two dictionaries.
- **`_ensure_directories(config)`**: A helper function that ensures all the directories specified in the configuration exist, and creates them if they don't.
- **`save_config(config, config_file)`**: Saves the given configuration to a JSON file.
- **`validate_config(config)`**: Validates the given configuration to ensure that the values are within acceptable ranges and that all required settings are present.
