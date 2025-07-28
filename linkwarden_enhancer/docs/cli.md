# CLI Module

The `cli` module provides the command-line interface for the Linkwarden Enhancer. It allows users to interact with the application, run different commands, and manage their bookmarks from the terminal.

## Modules

### Main CLI (`main_cli.py`)

The `main_cli.py` script is the main entry point for the CLI application. It uses `argparse` to create a comprehensive command-line interface with support for various commands and options.

**Classes:**

- **`MainCLI`**: The main class that encapsulates the entire CLI application.
    - **`__init__()`**: Initializes the CLI application, setting up the configuration, safety manager, interactive reviewer, and other components.
    - **`run(args)`**: The main entry point for the CLI application. It parses the command-line arguments, sets up logging, loads the configuration, and executes the appropriate command.
    - **`_create_argument_parser()`**: Creates the `ArgumentParser` object and defines all the available commands, subcommands, and options.
    - **`_add_*_command(subparsers)`**: A series of helper methods that add the different subcommands (e.g., `process`, `import`, `validate`) to the argument parser.
    - **`_apply_cli_overrides(args)`**: Applies the command-line arguments as overrides to the application's configuration.
    - **`_initialize_components()`**: Initializes all the necessary components of the application, such as the `SafetyManager`, `MetricsCollector`, and `InteractiveReviewer`.
    - **`_execute_command(args)`**: Executes the appropriate command based on the parsed command-line arguments.
    - **`_execute_*_command(args)`**: A series of helper methods that implement the logic for each of the available commands.
    - **`_interactive_*()`**: A series of helper methods that implement the logic for the different options in the interactive menu.
    - **`_print_nested_dict(data, indent)`**: A helper method for printing nested dictionaries in a human-readable format.

### Interactive Mode (`interactive.py`)

The `interactive.py` module provides the components for the interactive mode of the CLI. This mode allows users to review and provide feedback on the suggestions made by the AI.

**Classes:**

- **`InteractiveReviewer`**: A class for reviewing AI suggestions and providing feedback.
    - **`__init__(config)`**: Initializes the interactive reviewer with the application configuration.
    - **`review_category_suggestions(url, title, content, suggestions)`**: Prompts the user to review and select a category from a list of suggestions.
    - **`review_tag_suggestions(url, title, content, suggestions, existing_tags)`**: Prompts the user to review and select tags from a list of suggestions.
    - **`review_enhancement_results(url, original_data, enhanced_data)`**: Prompts the user to review the results of the content enhancement process.
    - **`_edit_enhancement_data(data)`**: Allows the user to manually edit the enhanced data.
    - **`_track_*_feedback(...)`**: A series of helper methods for tracking user feedback and sending it to the `AdaptiveIntelligence` system.
    - **`show_learning_progress()`**: Displays the learning progress and statistics of the `AdaptiveIntelligence` system.
    - **`get_user_confirmation(message, default)`**: Prompts the user for a yes/no confirmation.
- **`InteractiveMenu`**: A class for creating and managing the interactive menu.
    - **`__init__()`**: Initializes the interactive menu.
    - **`show_main_menu()`**: Displays the main menu and gets the user's choice.
    - **`show_submenu(title, options)`**: Displays a submenu with the given options.
    - **`get_file_path(prompt, must_exist)`**: Prompts the user for a file path and validates it.
    - **`get_yes_no(prompt, default)`**: Prompts the user for a yes/no answer.
    - **`show_list_selection(title, items, multi_select)`**: Displays a list of items and prompts the user to select one or more.

### Help System (`help_system.py`)

The `help_system.py` module provides a comprehensive help and documentation system for the CLI. It allows users to get help on different topics and commands.

**Classes:**

- **`HelpSystem`**: The main class that manages the help topics and documentation.
    - **`__init__()`**: Initializes the help system and loads the help topics.
    - **`_load_help_topics()`**: Loads the help topics from a dictionary.
    - **`show_help(topic)`**: Displays the help for a specific topic.
    - **`show_all_topics()`**: Displays a list of all available help topics.
    - **`search_help(search_term)`**: Searches the help content for a specific term.
    - **`show_quick_reference()`**: Displays a quick reference guide with the most common commands.

**Functions:**

- **`show_help_command(topic)`**: A convenience function for showing help from the command line.
