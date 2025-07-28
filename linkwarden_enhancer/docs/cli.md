# Command-Line Interface (CLI)

The `cli` module provides a powerful and flexible command-line interface for the Linkwarden Enhancer. It allows users to perform all the major operations of the application, such as importing bookmarks, enhancing them with AI, and generating reports, directly from the terminal.

## Main Usage

The main entry point for the CLI is the `linkwarden-enhancer` command (or `python -m linkwarden_enhancer.cli.main_cli` if not installed as a package). The basic syntax is:

```bash
linkwarden-enhancer <command> [options]
```

## Global Options

These options can be used with any command:

- `-v`, `--verbose`: Enable verbose logging and debugging output.
- `--debug`: Enable debug mode with detailed component logging.
- `-c`, `--config FILE`: Specify a custom configuration file.
- `--dry-run`: Perform a dry run without making any changes to the data.
- `--interactive`: Enable interactive mode for reviewing suggestions.
- `--log-file FILE`: Specify a path to a log file for detailed logging.

## Commands

### `process`

The `process` command is the main command for processing and enhancing a bookmark file.

**Usage:**

```bash
linkwarden-enhancer process INPUT_FILE OUTPUT_FILE [OPTIONS]
```

**Arguments:**

- `INPUT_FILE`: The path to the input JSON file containing the bookmarks to process.
- `OUTPUT_FILE`: The path to the output JSON file where the enhanced bookmarks will be saved.

**Options:**

- `--import-github`: Import GitHub starred repositories before processing.
- `--import-browser FILE`: Import browser bookmarks from an HTML file before processing.
- `--enable-scraping`: Enable web scraping to enhance bookmarks with metadata.
- `--enable-ai-analysis`: Enable AI content analysis and suggestions.
- `--enable-learning`: Enable continuous learning from the processed data.
- `--generate-report`: Generate a detailed report of the processing operation.

### `import`

The `import` command is used to import bookmarks from various sources.

**Usage:**

```bash
linkwarden-enhancer import [OPTIONS] --output OUTPUT_FILE
```

**Options:**

- `--github`: Import from GitHub.
- `--github-token TOKEN`: Your GitHub personal access token.
- `--github-username USER`: Your GitHub username.
- `--browser FILE`: The path to a browser bookmarks HTML file.
- `--linkwarden-backup FILE`: The path to a Linkwarden backup JSON file.
- `-o`, `--output FILE`: The path to the output file where the imported bookmarks will be saved.
- `--merge-with FILE`: An existing bookmark file to merge the imported bookmarks with.

### `validate`

The `validate` command is used to validate the integrity of a bookmark data file.

**Usage:**

```bash
linkwarden-enhancer validate INPUT_FILE [OPTIONS]
```

**Options:**

- `--fix-issues`: Attempt to automatically fix any validation issues found.
- `--detailed-report`: Generate a detailed report of the validation results.

### `report`

The `report` command is used to generate various reports.

**Usage:**

```bash
linkwarden-enhancer report <report_type> [OPTIONS]
```

**Report Types:**

- `operation`: Generates a report that compares the state of the data before and after an operation.
- `period`: Generates a report that summarizes the changes that occurred during a specific time period.
- `performance`: Generates a report on the performance metrics of the application.

### `stats`

The `stats` command is used to display various statistics about the system.

**Usage:**

```bash
linkwarden-enhancer stats [OPTIONS]
```

**Options:**

- `--learning`: Show learning statistics.
- `--intelligence`: Show intelligence system statistics.
- `--performance`: Show performance statistics.
- `--safety`: Show safety system statistics.
- `--all`: Show all statistics.
- `--export FILE`: Export the statistics to a JSON file.

### `backup`

The `backup` command is used for backup and recovery operations.

**Usage:**

```bash
linkwarden-enhancer backup <action> [OPTIONS]
```

**Actions:**

- `create`: Creates a backup of a file.
- `list`: Lists the available backups.
- `restore`: Restores a backup from a file.
- `cleanup`: Cleans up old backups based on the retention policy.

### `check-dead-links`

The `check-dead-links` command is used to check for and manage dead links in a bookmark file.

**Usage:**

```bash
linkwarden-enhancer check-dead-links INPUT_FILE [OPTIONS]
```

### `intelligence`

The `intelligence` command is used for managing the intelligence system.

**Usage:**

```bash
linkwarden-enhancer intelligence <action> [OPTIONS]
```

**Actions:**

- `export`: Exports the intelligence data to a file.
- `import`: Imports intelligence data from a file.
- `train`: Trains the intelligence system from a data file.

### `cache`

The `cache` command is used for managing the application's cache.

**Usage:**

```bash
linkwarden-enhancer cache <action> [OPTIONS]
```

**Actions:**

- `info`: Shows information about the cache.
- `clear`: Clears the cache.
- `refresh`: Force-refreshes the cache.

### `menu`

The `menu` command launches the interactive menu interface.

**Usage:**

```bash
linkwarden-enhancer menu
```

### `help`

The `help` command displays help and documentation.

**Usage:**

```bash
linkwarden-enhancer help [topic]
```
