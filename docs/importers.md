# Importers Module

The `importers` module is responsible for importing bookmarks from various sources into the Linkwarden Enhancer. It provides a flexible and extensible framework for adding new importers.

## Modules

### Universal Importer (`universal_importer.py`)

The `UniversalImporter` is the main entry point for all import operations. It can coordinate multiple importers to import bookmarks from various sources in a single operation.

**Classes:**

- **`ImportConfig`**: A data class for configuring the universal import process.
- **`CombinedImportResult`**: A data class for storing the results of a multi-source import operation.
- **`UniversalImporter`**: The main class that orchestrates the import process.
    - **`__init__(config)`**: Initializes the universal importer with the application configuration.
    - **`import_all_sources(import_config)`**: Imports bookmarks from all the sources specified in the `ImportConfig`.
    - **`preview_all_sources(import_config)`**: Previews the import operation without actually importing any data.
    - **`validate_import_config(import_config)`**: Validates the import configuration.

### Base Importer (`base_importer.py`)

The `base_importer.py` module provides the base interface for all importers. It defines the common methods and properties that all importers must implement.

**Classes:**

- **`BaseImporter`**: An abstract base class that defines the common interface for all importers.
    - **`__init__(config)`**: Initializes the importer with the application configuration.
    - **`import_data(**kwargs)`**: An abstract method that must be implemented by subclasses to import data from a source.
    - **`validate_config()`**: An abstract method that can be implemented by subclasses to validate their configuration.
    - **`get_import_stats()`**: Gets statistics about the import process.
    - **`add_error(error)`**: Adds an error message to the importer.
    - **`add_warning(warning)`**: Adds a warning message to the importer.

### GitHub Importer (`github_importer.py`)

The `GitHubImporter` is a specialized importer for importing bookmarks from GitHub. It can import a user's starred repositories and their own repositories.

**Classes:**

- **`GitHubImporter`**: The main class that performs the import from GitHub.
    - **`__init__(config)`**: Initializes the GitHub importer with the application configuration.
    - **`import_data(import_starred, import_owned, max_repos, force_refresh)`**: Imports the user's starred and/or owned repositories from GitHub.
    - **`_import_starred_repositories(max_repos)`**: Imports the user's starred repositories.
    - **`_import_user_repositories(max_repos)`**: Imports the user's own public repositories.
    - **`_convert_repo_to_bookmark(repo, bookmark_type)`**: Converts a GitHub repository object to a bookmark.
    - **`_generate_repo_tags(repo, languages, topics, bookmark_type)`**: Generates intelligent tags for a repository.
    - **`_suggest_repo_collection(repo, languages, topics)`**: Suggests an appropriate collection for a repository.
    - **`_import_*_cached(...)`**: A series of helper methods for caching the results of API requests.

### Linkwarden Importer (`linkwarden_importer.py`)

The `LinkwardenImporter` is a specialized importer for importing bookmarks from a Linkwarden backup JSON file.

**Classes:**

- **`LinkwardenImporter`**: The main class that performs the import from a Linkwarden backup.
    - **`__init__(config)`**: Initializes the Linkwarden importer with the application configuration.
    - **`import_data(backup_file_path)`**: Imports bookmarks from a Linkwarden backup JSON file.
    - **`_validate_linkwarden_structure(data)`**: Validates the structure of the Linkwarden backup data.
    - **`_extract_collection_bookmarks(collection)`**: Extracts bookmarks from a collection.
    - **`_convert_linkwarden_bookmark(link, collection)`**: Converts a Linkwarden link to the internal bookmark format.
    - **`preview_import(backup_file_path, max_items)`**: Previews the import operation without actually importing any data.
