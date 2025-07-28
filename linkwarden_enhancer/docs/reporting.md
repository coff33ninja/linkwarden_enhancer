# Reporting Module

The `reporting` module is responsible for generating comprehensive reports for change tracking, performance monitoring, and system analysis.

## Modules

### Report Generator (`report_generator.py`)

The `ReportGenerator` is the main component for creating detailed reports in various formats. It can track changes in the system and generate reports that summarize these changes.

**Classes:**

- **`ReportFormat`**: An enumeration of the supported report formats (JSON, HTML, CSV, Markdown).
- **`ChangeRecord`**: A data class that represents a single change in the system.
- **`ReportSummary`**: A data class that represents a summary of a report.
- **`ReportGenerator`**: The main class that generates reports.
    - **`__init__(config, data_dir)`**: Initializes the report generator with the application configuration and data directory.
    - **`track_change(...)`**: Tracks a change in the system and saves it to a log.
    - **`generate_operation_report(...)`**: Generates a report that compares the state of the data before and after an operation.
    - **`generate_period_report(start_date, end_date, formats)`**: Generates a report that summarizes the changes that occurred during a specific time period.
    - **`generate_comparison_report(...)`**: Generates a report that compares two different data sets.
    - **`_analyze_data_changes(before_data, after_data)`**: Analyzes the changes between two data sets.
    - **`_save_report(report_data, format_type, report_id)`**: Saves a report in the specified format.

### Metrics Collector (`metrics_collector.py`)

The `MetricsCollector` is responsible for collecting and tracking performance metrics for all system components. This is useful for monitoring the health and performance of the application.

**Classes:**

- **`PerformanceMetric`**: A data class that represents a single performance metric measurement.
- **`OperationMetrics`**: A data class that represents the metrics for a specific operation.
- **`MetricsCollector`**: The main class that collects and tracks performance metrics.
    - **`__init__(config)`**: Initializes the metrics collector with the application configuration.
    - **`start_monitoring()`**: Starts a background thread to monitor system metrics.
    - **`stop_monitoring()`**: Stops the background monitoring thread.
    - **`record_metric(...)`**: Records a performance metric.
    - **`start_operation(operation_name, context)`**: Starts tracking metrics for an operation.
    - **`end_operation(...)`**: Ends tracking metrics for an operation.
    - **`track_operation(operation_name, context)`**: A context manager for tracking operation metrics.
    - **`get_performance_summary(time_window_hours)`**: Gets a performance summary for a specified time window.
    - **`get_system_health()`**: Gets the current system health metrics.
    - **`export_metrics(format_type, time_window_hours)`**: Exports metrics data in the specified format.
