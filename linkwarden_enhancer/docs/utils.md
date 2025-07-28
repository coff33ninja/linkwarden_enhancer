# Utils Module

The `utils` module contains a collection of utility functions and classes that are used throughout the Linkwarden Enhancer application. These utilities provide common functionality for tasks such as file operations, JSON handling, logging, and more.

## Modules

### File Utils (`file_utils.py`)

The `FileUtils` class provides a set of static methods for performing common file operations.

**Classes:**

- **`FileUtils`**: A collection of static methods for file operations.
    - **`ensure_directory(directory_path)`**: Ensures that a directory exists, and creates it if it doesn't.
    - **`get_file_hash(file_path)`**: Calculates the SHA256 hash of a file.
    - **`backup_file(source_path, backup_dir, prefix)`**: Creates a timestamped backup of a file.
    - **`cleanup_old_backups(backup_dir, pattern, keep_count)`**: Cleans up old backup files based on a retention policy.
    - **`get_file_size(file_path)`**: Gets the size of a file in bytes.
    - **`is_file_readable(file_path)`**: Checks if a file is readable.
    - **`get_available_disk_space(directory)`**: Gets the available disk space in a directory.
    - **`safe_filename(filename)`**: Makes a filename safe for the filesystem.
    - **`find_files(directory, pattern, recursive)`**: Finds files in a directory that match a given pattern.

### JSON Handler (`json_handler.py`)

The `JsonHandler` class provides utilities for working with JSON files.

**Classes:**

- **`JsonHandler`**: A class for handling JSON file operations.
    - **`load_json(file_path)`**: Loads a JSON file with robust encoding handling.
    - **`save_json(data, file_path, indent)`**: Saves data to a JSON file.
    - **`validate_json_structure(data, required_keys)`**: Validates the structure of a JSON object.
    - **`get_json_stats(data)`**: Calculates statistics about the content of a JSON object.

### Logging Utils (`logging_utils.py`)

The `logging_utils.py` module provides utilities for setting up and managing logging in the application.

**Functions:**

- **`setup_logging(level, log_file, max_file_size_mb, backup_count)`**: Sets up the basic logging configuration.
- **`setup_verbose_logging(enable_debug, component_filters, log_file)`**: Sets up verbose logging for debugging.
- **`get_logger(name)`**: Gets a logger instance for a specific module.
- **`get_component_logger(component_name, verbose)`**: Gets an enhanced component logger.

**Classes:**

- **`ComponentLogger`**: An enhanced logger for specific components that provides methods for structured logging of operations, data flow, performance, and learning events.

### Progress Utils (`progress_utils.py`)

The `progress_utils.py` module provides utilities for tracking the progress of long-running operations.

**Classes:**

- **`ProgressStats`**: A data class for storing progress statistics.
- **`ProgressIndicator`**: A class for displaying a progress bar and other progress information in the console.
    - **`update(current, operation, phase)`**: Updates the progress indicator with the current progress.
    - **`finish(message)`**: Finishes the progress tracking and displays a final message.
- **`DetailedProgressTracker`**: A class for tracking the progress of multi-phase operations.
    - **`start_phase(phase_name, total_items)`**: Starts a new phase of the operation.
    - **`finish_phase(phase_name, items_processed, learning_data)`**: Finishes the current phase.
    - **`show_overall_progress()`**: Displays the overall progress across all phases.
    - **`show_learning_summary()`**: Displays a summary of the learning statistics.

### Text Utils (`text_utils.py`)

The `TextUtils` class provides a set of static methods for performing common text processing and analysis tasks.

**Classes:**

- **`TextUtils`**: A collection of static methods for text processing.
    - **`clean_text(text)`**: Cleans and normalizes text.
    - **`extract_keywords(text, min_length, max_keywords)`**: Extracts keywords from a block of text.
    - **`calculate_text_similarity(text1, text2)`**: Calculates the similarity between two blocks of text.
    - **`truncate_text(text, max_length, suffix)`**: Truncates text to a specified length.
    - **`normalize_tag_name(tag_name)`**: Normalizes a tag name for consistency.
    - **`extract_sentences(text, max_sentences)`**: Extracts the first few sentences from a block of text.
    - **`detect_language_hints(text)`**: Detects programming language hints in a block of text.

### URL Utils (`url_utils.py`)

The `UrlUtils` class provides a set of static methods for performing common URL processing and analysis tasks.

**Classes:**

- **`UrlUtils`**: A collection of static methods for URL processing.
    - **`extract_domain(url)`**: Extracts the domain from a URL.
    - **`extract_path_segments(url)`**: Extracts the path segments from a URL.
    - **`is_valid_url(url)`**: Checks if a URL is valid.
    - **`normalize_url(url)`**: Normalizes a URL for comparison.
    - **`get_base_url(url)`**: Gets the base URL (scheme + netloc).
    - **`resolve_relative_url(base_url, relative_url)`**: Resolves a relative URL against a base URL.
    - **`extract_url_keywords(url)`**: Extracts keywords from a URL.
    - **`classify_url_type(url)`**: Classifies a URL into different types (e.g., "github_repository", "documentation", "article").

### Version (`version.py`)

The `version.py` module provides functions for getting the version information of the application.

**Functions:**

- **`get_version_info()`**: Returns a comprehensive version information string.
- **`get_short_version()`**: Returns a short version string.
