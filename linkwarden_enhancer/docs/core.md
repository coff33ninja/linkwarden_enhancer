# Core Module

The `core` module contains the essential components and business logic of the Linkwarden Enhancer application. It is responsible for ensuring the safety, integrity, and reliability of all operations.

## Modules

### Safety Manager (`safety_manager.py`)

The `SafetyManager` is the central orchestrator for all safety-related operations. It coordinates the actions of the other core components to ensure that all operations are performed in a safe and reliable manner.

**Classes:**

- **`SafetyManager`**: The main class that orchestrates all safety operations.
    - **`__init__(config)`**: Initializes the safety manager with the application configuration and all the core safety components.
    - **`execute_safe_cleanup(...)`**: Executes the entire bookmark enhancement process with all safety checks enabled, including validation, backup, import, enhancement, learning, and output generation.
    - **`import_from_github()`**: Imports bookmarks from GitHub with safety checks.
    - **`rollback_to_backup(backup_path, target_file)`**: Rolls back to a specific backup using the recovery system.
    - **`create_recovery_plan(target_file, backup_path)`**: Creates a recovery plan for manual recovery.
    - **`generate_recovery_documentation(target_file, backup_path)`**: Generates manual recovery documentation.
    - **`validate_data_file(file_path)`**: Validates a data file using the validation engine.
    - **`get_safety_statistics()`**: Gets comprehensive statistics about the safety system.
    - **`list_available_backups(operation_name)`**: Lists the available backups for recovery.
    - **`cleanup_old_backups()`**: Cleans up old backups based on the retention policy.

### Validation Engine (`validation_engine.py`)

The `ValidationEngine` is responsible for validating the integrity and consistency of the bookmark data. It uses JSON schemas and custom validation logic to ensure that the data is well-formed and consistent.

**Classes:**

- **`ValidationEngine`**: The main class that performs all validation checks.
    - **`__init__(config)`**: Initializes the validation engine and loads the JSON schemas.
    - **`validate_json_schema(data, schema_name)`**: Validates the given data against a predefined JSON schema.
    - **`validate_data_consistency(data)`**: Checks for inconsistencies in the data, such as orphaned references and duplicate IDs.
    - **`validate_field_requirements(data)`**: Validates that all required fields are present and have valid values.
    - **`create_data_inventory(data)`**: Creates a detailed inventory of the data, including counts of bookmarks, collections, and tags.

### Backup System (`backup_system.py`)

The `BackupSystem` provides a multi-tier backup and retention management system. It is used to create and manage backups of the bookmark data.

**Classes:**

- **`BackupInfo`**: A data class that represents information about a backup file.
- **`BackupSystem`**: The main class that manages backups.
    - **`__init__(config)`**: Initializes the backup system with the application configuration.
    - **`create_backup(source_file, operation_name, metadata)`**: Creates a timestamped backup of the source file.
    - **`create_incremental_backup(source_file, operation_name, metadata)`**: Creates an incremental backup if changes are detected.
    - **`restore_backup(backup_path, target_file)`**: Restores a backup from a file.
    - **`verify_backup_integrity(backup_path)`**: Verifies the integrity of a backup file using its checksum.
    - **`list_backups(operation_name)`**: Lists all the available backups.
    - **`cleanup_old_backups()`**: Cleans up old backups based on the retention policy.

### Progress Monitor (`progress_monitor.py`)

The `ProgressMonitor` provides real-time progress tracking for long-running operations. It also includes safety thresholds to prevent runaway operations.

**Classes:**

- **`OperationStatus`**: An enumeration of the possible statuses of an operation.
- **`ProgressInfo`**: A data class that represents the progress information for an operation.
- **`SafetyThreshold`**: A data class that represents a safety threshold configuration.
- **`ProgressMonitor`**: The main class that monitors the progress of operations.
    - **`__init__(config)`**: Initializes the progress monitor with the application configuration.
    - **`start_operation(operation_name, total_items, description)`**: Starts tracking a new operation.
    - **`update_progress(operation_id, current_item, current_task, additional_data)`**: Updates the progress of an operation.
    - **`complete_operation(operation_id, success)`**: Marks an operation as completed.
    - **`cancel_operation(operation_id)`**: Cancels a running operation.

### Integrity Checker (`integrity_checker.py`)

The `IntegrityChecker` is responsible for performing comprehensive data integrity validation and comparison. It can be used to check the integrity of a single data set or compare two data sets to find the differences.

**Classes:**

- **`DataDiff`**: A data class that represents the differences between two data sets.
- **`RelationshipIssue`**: A data class that represents a relationship integrity issue.
- **`IntegrityChecker`**: The main class that performs integrity checks and comparisons.
    - **`__init__(config)`**: Initializes the integrity checker with the application configuration.
    - **`check_data_integrity(data)`**: Performs a comprehensive integrity check on the given data.
    - **`compare_data_sets(before_data, after_data)`**: Compares two data sets and generates a detailed report of the differences.
    - **`validate_before_after_consistency(before_data, after_data)`**: Validates that critical data is preserved between two states of the data.

### Recovery System (`recovery_system.py`)

The `RecoverySystem` provides automated rollback and manual recovery procedures. It is used to recover from errors and restore the data to a previous state.

**Classes:**

- **`RecoveryPlan`**: A data class that represents a detailed recovery plan.
- **`RecoveryResult`**: A data class that represents the result of a recovery operation.
- **`RecoverySystem`**: The main class that manages recovery operations.
    - **`__init__(config, backup_system)`**: Initializes the recovery system with the application configuration and a `BackupSystem` instance.
    - **`rollback_to_latest_backup(target_file, operation_name)`**: Rolls back to the most recent backup.
    - **`rollback_to_backup(backup_path, target_file)`**: Rolls back to a specific backup.
    - **`create_recovery_plan(target_file, backup_path)`**: Creates a detailed recovery plan.
    - **`generate_recovery_script(recovery_plan)`**: Generates a manual recovery script.
    - **`generate_recovery_documentation(recovery_plan)`**: Generates manual recovery documentation.

### Dead Link Detector (`dead_link_detector.py`)

The `DeadLinkDetector` is responsible for detecting and managing dead links in the bookmark collection.

**Classes:**

- **`DeadLinkDetector`**: The main class that detects dead links.
    - **`__init__(config)`**: Initializes the dead link detector with the application configuration.
    - **`check_bookmarks(bookmarks)`**: Checks a list of bookmarks for dead links.
    - **`_check_single_bookmark(semaphore, bookmark)`**: Checks a single bookmark with retry logic.
- **`DeadLinkManager`**: A class for managing dead link collections and cleanup operations.
    - **`__init__(config)`**: Initializes the dead link manager with the application configuration.
    - **`organize_dead_links(bookmarks, dead_link_results)`**: Organizes bookmarks by moving dead links to appropriate collections.
