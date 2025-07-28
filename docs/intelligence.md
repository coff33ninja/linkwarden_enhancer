# Intelligence Module

The `intelligence` module is the brain of the Linkwarden Enhancer. It contains all the components responsible for learning from user behavior, making intelligent suggestions, and personalizing the user experience.

## Modules

### Intelligence Manager (`intelligence_manager.py`)

The `IntelligenceManager` is the central orchestrator for all intelligence-related operations. It provides a unified interface for exporting, importing, and managing the different components of the intelligence system.

**Classes:**

- **`IntelligenceSnapshot`**: A data class that represents a snapshot of intelligence data.
- **`IntelligenceExport`**: A data class that represents a complete intelligence export package.
- **`IntelligenceManager`**: The main class that manages all intelligence operations.
    - **`__init__(config, data_dir)`**: Initializes the intelligence manager with the application configuration and data directory.
    - **`create_full_export(description, include_user_data)`**: Creates a complete export of all intelligence data.
    - **`create_selective_export(components, description)`**: Creates an export of selected intelligence components.
    - **`import_intelligence_data(import_file, components, merge_strategy)`**: Imports intelligence data from an export file.
    - **`create_intelligence_backup()`**: Creates an automatic backup of the intelligence data.
    - **`restore_from_backup(backup_file)`**: Restores intelligence data from a backup.

### Adaptive Intelligence (`adaptive_intelligence.py`)

The `AdaptiveIntelligence` system is responsible for learning from user behavior and feedback to personalize the user experience.

**Classes:**

- **`FeedbackType`**: An enumeration of the possible types of user feedback.
- **`UserFeedback`**: A data class that represents a user feedback event.
- **`UserPreference`**: A data class that represents a learned user preference.
- **`PersonalizationProfile`**: A data class that represents a user's personalization profile.
- **`AdaptiveIntelligence`**: The main class that manages adaptive intelligence.
    - **`__init__(data_dir, user_id)`**: Initializes the adaptive intelligence system.
    - **`track_user_feedback(...)`**: Tracks user feedback on system suggestions.
    - **`get_personalized_suggestions(...)`**: Gets personalized suggestions based on user preferences.
    - **`decay_preferences()`**: Applies decay to unused preferences and removes weak ones.
    - **`get_adaptation_statistics()`**: Gets comprehensive statistics about the adaptation system.

### Continuous Learner (`continuous_learner.py`)

The `ContinuousLearner` is responsible for continuously improving the intelligence of the system by learning from new data and user interactions.

**Classes:**

- **`LearningMetrics`**: A data class for tracking learning performance.
- **`PatternStrength`**: A data class that represents the strength and reliability of a learned pattern.
- **`LearningSession`**: A data class that represents a single learning session.
- **`ContinuousLearner`**: The main class that manages continuous learning.
    - **`__init__(data_dir)`**: Initializes the continuous learner.
    - **`start_learning_session(session_type)`**: Starts a new learning session.
    - **`learn_from_new_bookmarks(bookmarks)`**: Learns from newly added bookmarks.
    - **`predict_category(url, title, content)`**: Predicts a category for a bookmark using learned patterns.
    - **`predict_tags(url, title, content)`**: Predicts tags for a bookmark using learned patterns.
    - **`track_prediction_feedback(...)`**: Tracks feedback on predictions to improve accuracy.
    - **`end_learning_session()`**: Ends the current learning session.

### Dictionary Manager (`dictionary_manager.py`)

The `DictionaryManager` is the central manager for all intelligent categorization and tagging. It orchestrates the actions of the `CategoryDictionary`, `TagDictionary`, `PatternLearner`, and `DomainClassifier`.

**Classes:**

- **`SmartDictionaryManager`**: The main class that manages the intelligent dictionaries.
    - **`__init__(config)`**: Initializes the dictionary manager with the application configuration.
    - **`learn_from_bookmark_data(bookmarks)`**: Learns from existing bookmark data to improve intelligence.
    - **`suggest_categories_for_bookmark(url, title, content)`**: Gets comprehensive category suggestions for a bookmark.
    - **`suggest_tags_for_bookmark(url, title, content, existing_tags)`**: Gets comprehensive tag suggestions for a bookmark.
    - **`get_related_tags(tag, limit)`**: Gets tags that are related to the given tag.

### Category Dictionary (`category_dictionary.py`)

The `CategoryDictionary` provides intelligent category suggestions based on domain and content patterns.

**Classes:**

- **`CategoryDictionary`**: The main class that manages the category dictionary.
    - **`__init__()`**: Initializes the category dictionary with comprehensive patterns.
    - **`suggest_categories_for_url(url, content)`**: Suggests categories for a URL with confidence scores.
    - **`learn_from_bookmark_data(bookmarks)`**: Learns category patterns from existing bookmark data.

### Tag Dictionary (`tag_dictionary.py`)

The `TagDictionary` provides intelligent tag suggestions based on content analysis and predefined patterns.

**Classes:**

- **`TagDictionary`**: The main class that manages the tag dictionary.
    - **`__init__()`**: Initializes the tag dictionary with comprehensive patterns.
    - **`suggest_tags_for_content(title, content, url, existing_tags)`**: Suggests tags based on content analysis.
    - **`learn_from_bookmark_tags(bookmarks)`**: Learns tag patterns from existing bookmark data.
    - **`get_related_tags(tag, limit)`**: Gets tags that frequently co-occur with the given tag.

### Pattern Learner (`pattern_learner.py`)

The `PatternLearner` is responsible for learning and adapting patterns from user behavior and bookmark data.

**Classes:**

- **`PatternLearner`**: The main class that learns and manages patterns.
    - **`__init__(data_dir)`**: Initializes the pattern learner.
    - **`learn_from_bookmark_history(bookmarks)`**: Learns patterns from existing bookmark categorization.
    - **`predict_category_for_url(url)`**: Predicts a category for a URL based on learned patterns.
    - **`predict_tags_for_content(content, limit)`**: Predicts tags for content based on learned patterns.
    - **`track_user_feedback(feedback_type, original, modified, context)`**: Tracks user feedback to improve predictions.

### Domain Classifier (`domain_classifier.py`)

The `DomainClassifier` classifies domains into categories based on a comprehensive set of predefined patterns.

**Classes:**

- **`DomainClassifier`**: The main class that classifies domains.
    - **`__init__()`**: Initializes the domain classifier with comprehensive patterns.
    - **`classify_domain(url)`**: Classifies a domain into categories with confidence scores.
    - **`learn_from_classifications(domain, category, confidence)`**: Learns new domain classifications.
