#!/usr/bin/env python3
"""
Main CLI entry point for Linkwarden Enhancer
"""

import click
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any

from .config.settings import load_config
from .core.safety_manager import SafetyManager
from .utils.logging_utils import setup_logging
from .utils.version import get_version_info


@click.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(exists=True),
    help="Input Linkwarden backup JSON file",
)
@click.option(
    "--output", "-o", "output_file", type=click.Path(), help="Output enhanced JSON file"
)
@click.option("--dry-run", is_flag=True, help="Preview changes without applying them")
@click.option("--ai-enabled", is_flag=True, help="Enable AI analysis and enhancement")
@click.option(
    "--import-github",
    is_flag=True,
    help="Import GitHub starred repositories and owned repos",
)
@click.option("--github-username", help="GitHub username for import (overrides config)")
@click.option("--github-token", help="GitHub token for import (overrides config)")
@click.option(
    "--ollama-model", default="llama2", help="Ollama model to use for LLM features"
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.85,
    help="Similarity threshold for duplicate detection",
)
@click.option(
    "--max-clusters",
    type=int,
    default=50,
    help="Maximum number of clusters for bookmark organization",
)
@click.option("--enable-clustering", is_flag=True, help="Enable bookmark clustering")
@click.option(
    "--enable-smart-tagging", is_flag=True, help="Enable AI-powered smart tagging"
)
@click.option(
    "--import-browser",
    type=click.Path(exists=True),
    help="Import browser bookmarks from HTML file",
)
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Custom configuration file"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--version", is_flag=True, help="Show version information")
@click.option(
    "--validate-only", is_flag=True, help="Only validate input file without processing"
)
@click.option("--list-backups", is_flag=True, help="List available backups")
@click.option(
    "--rollback",
    type=click.Path(exists=True),
    help="Rollback from specified backup file",
)
@click.option(
    "--cleanup-backups",
    is_flag=True,
    help="Clean up old backups based on retention policy",
)
@click.option("--safety-stats", is_flag=True, help="Show safety system statistics")
def main(
    input_file: Optional[str],
    output_file: Optional[str],
    dry_run: bool,
    ai_enabled: bool,
    import_github: bool,
    github_username: Optional[str],
    github_token: Optional[str],
    ollama_model: str,
    similarity_threshold: float,
    max_clusters: int,
    enable_clustering: bool,
    enable_smart_tagging: bool,
    import_browser: Optional[str],
    config: Optional[str],
    verbose: bool,
    version: bool,
    validate_only: bool,
    list_backups: bool,
    rollback: Optional[str],
    cleanup_backups: bool,
    safety_stats: bool,
):
    """
    Linkwarden Enhancer - Intelligent bookmark management with AI

    Transform your Linkwarden bookmarks into a smart, continuously learning
    organization system with comprehensive safety checks and multi-source import.
    """

    if version:
        click.echo(get_version_info())
        return

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)

    # Load configuration
    config_data = load_config(config)

    # Override config with CLI arguments
    if github_username:
        config_data["github"]["username"] = github_username
    if github_token:
        config_data["github"]["token"] = github_token
    if ai_enabled:
        config_data["ai"]["enable_ai_analysis"] = True
    if dry_run:
        config_data["safety"]["dry_run_mode"] = True

    config_data["ai"]["ollama_model"] = ollama_model
    config_data["ai"]["similarity_threshold"] = similarity_threshold
    config_data["ai"]["max_clusters"] = max_clusters
    config_data["ai"]["enable_clustering"] = enable_clustering
    config_data["ai"]["enable_smart_tagging"] = enable_smart_tagging

    try:
        # Initialize safety manager
        safety_manager = SafetyManager(config_data)

        # Handle safety-only operations
        if safety_stats:
            click.echo("📊 Safety System Statistics")
            stats = safety_manager.get_safety_statistics()
            _display_safety_stats(stats)
            return

        if list_backups:
            click.echo("💾 Available Backups")
            backups = safety_manager.list_available_backups()
            _display_backups(backups)
            return

        if cleanup_backups:
            click.echo("🧹 Cleaning up old backups...")
            result = safety_manager.cleanup_old_backups()
            if result["success"]:
                click.echo(f"✅ {result['message']}")
            else:
                click.echo(f"❌ Cleanup failed: {result['error']}")
            return

        if rollback:
            if not input_file:
                click.echo("❌ Error: --input file required for rollback target")
                sys.exit(1)

            click.echo(f"🔄 Rolling back from {rollback} to {input_file}...")
            result = safety_manager.rollback_to_backup(rollback, input_file)

            if result["success"]:
                click.echo(
                    f"✅ Rollback completed successfully in {result['recovery_time']:.2f}s"
                )
                if not result["verification_passed"]:
                    click.echo("⚠️  Warning: Recovery verification had issues")
            else:
                click.echo(
                    f"❌ Rollback failed: {result.get('error', 'Unknown error')}"
                )
                sys.exit(1)
            return

        if validate_only:
            if not input_file:
                click.echo("❌ Error: --input file required for validation")
                sys.exit(1)

            click.echo(f"� Validating {input_file}...")
            validation_result = safety_manager.validate_data_file(input_file)
            _display_validation_results(validation_result)

            if validation_result["overall_valid"]:
                click.echo("✅ Validation passed!")
            else:
                click.echo("❌ Validation failed!")
                sys.exit(1)
            return

        # Determine operation mode
        if import_github and not input_file:
            # GitHub-only import mode
            click.echo("🐙 Importing from GitHub...")
            result = safety_manager.import_from_github()

        elif input_file:
            # Standard enhancement mode
            if not output_file:
                output_file = _generate_output_filename(input_file, dry_run)

            click.echo(f"🚀 Processing {input_file}...")
            if dry_run:
                click.echo("🔍 DRY RUN MODE - No changes will be applied")

            result = safety_manager.execute_safe_cleanup(
                input_file=input_file,
                output_file=output_file,
                import_github=import_github,
                import_browser=import_browser,
            )

        else:
            click.echo("❌ Error: Must specify --input file or --import-github")
            click.echo("Use --help for usage information")
            sys.exit(1)

        # Display results
        _display_results(result, verbose)

        if result.success:
            click.echo("✅ Operation completed successfully!")
        else:
            click.echo("❌ Operation completed with errors")
            sys.exit(1)

    except KeyboardInterrupt:
        click.echo("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"💥 Unexpected error: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def _generate_output_filename(input_file: str, dry_run: bool) -> str:
    """Generate output filename based on input file"""
    input_path = Path(input_file)
    stem = input_path.stem
    suffix = input_path.suffix

    if dry_run:
        return f"{stem}_dry_run_preview{suffix}"
    else:
        return f"{stem}_enhanced{suffix}"


def _display_results(result, verbose: bool):
    """Display operation results"""
    click.echo("\n📊 Results Summary:")
    click.echo(f"   • Execution time: {result.execution_time:.2f}s")

    if hasattr(result, "changes_applied"):
        changes = result.changes_applied
        click.echo(f"   • Bookmarks processed: {len(changes.bookmarks_modified)}")
        click.echo(f"   • Collections updated: {len(changes.collections_modified)}")
        click.echo(f"   • Tags enhanced: {len(changes.tags_modified)}")

    if hasattr(result, "ai_analysis_report"):
        ai_report = result.ai_analysis_report
        click.echo(f"   • AI tags suggested: {ai_report.ai_tags_suggested}")
        click.echo(f"   • Duplicates detected: {ai_report.duplicates_detected}")
        click.echo(f"   • Clusters created: {ai_report.clusters_created}")

    if hasattr(result, "backups_created"):
        click.echo(f"   • Backups created: {len(result.backups_created)}")

    if result.warnings and verbose:
        click.echo("\n⚠️  Warnings:")
        for warning in result.warnings[:5]:  # Show first 5 warnings
            click.echo(f"   • {warning}")
        if len(result.warnings) > 5:
            click.echo(f"   • ... and {len(result.warnings) - 5} more warnings")

    if result.errors:
        click.echo("\n❌ Errors:")
        for error in result.errors:
            click.echo(f"   • {error}")


if __name__ == "__main__":
    main()


def _display_safety_stats(stats: Dict[str, Any]):
    """Display safety system statistics"""

    if "error" in stats:
        click.echo(f"❌ Error getting stats: {stats['error']}")
        return

    # Validation stats
    if "validation_stats" in stats:
        v_stats = stats["validation_stats"]["validation_stats"]
        click.echo(
            f"   🔍 Validation: {v_stats['total_validations']} checks, {v_stats['schema_errors']} schema errors"
        )

    # Backup stats
    if "backup_stats" in stats:
        b_stats = stats["backup_stats"]
        click.echo(
            f"   💾 Backups: {b_stats['total_backups']} backups, {b_stats['total_size_mb']} MB total"
        )

    # Progress stats
    if "progress_stats" in stats:
        p_stats = stats["progress_stats"]
        click.echo(
            f"   📊 Operations: {p_stats['completed_operations']} completed, {p_stats['success_rate']:.1f}% success rate"
        )

    # Integrity stats
    if "integrity_stats" in stats:
        i_stats = stats["integrity_stats"]["integrity_stats"]
        click.echo(
            f"   🔒 Integrity: {i_stats['total_checks']} checks, {i_stats['issues_found']} issues found"
        )

    # Recovery stats
    if "recovery_stats" in stats:
        r_stats = stats["recovery_stats"]
        click.echo(
            f"   🔄 Recovery: {r_stats['total_recoveries']} recoveries, {r_stats['success_rate']:.1f}% success rate"
        )


def _display_backups(backups: List[Dict[str, Any]]):
    """Display available backups"""

    if not backups:
        click.echo("   No backups found")
        return

    click.echo(f"   Found {len(backups)} backups:")

    for backup in backups[:10]:  # Show first 10
        age_str = (
            f"{backup['age_hours']:.1f}h ago"
            if backup["age_hours"] < 24
            else f"{backup['age_hours']/24:.1f}d ago"
        )
        compressed_str = " (compressed)" if backup["compressed"] else ""

        click.echo(
            f"   • {backup['operation_name']}: {backup['file_size_mb']} MB, {age_str}{compressed_str}"
        )
        click.echo(f"     Path: {backup['path']}")

    if len(backups) > 10:
        click.echo(f"   ... and {len(backups) - 10} more backups")


def _display_validation_results(validation_result: Dict[str, Any]):
    """Display validation results"""

    if "error" in validation_result:
        click.echo(f"❌ Validation error: {validation_result['error']}")
        return

    # Overall status
    if validation_result["overall_valid"]:
        click.echo("✅ Overall validation: PASSED")
    else:
        click.echo("❌ Overall validation: FAILED")

    # Individual checks
    checks = [
        ("Schema", validation_result["schema_valid"]),
        ("Consistency", validation_result["consistency_valid"]),
        ("Fields", validation_result["fields_valid"]),
        ("Integrity", validation_result["integrity_valid"]),
    ]

    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        click.echo(f"   {check_name}: {status}")

    # Show inventory if available
    if (
        "inventory" in validation_result
        and "error" not in validation_result["inventory"]
    ):
        inventory = validation_result["inventory"]
        click.echo(f"\n📊 Data Inventory:")
        click.echo(f"   • Bookmarks: {inventory['total_bookmarks']}")
        click.echo(f"   • Collections: {inventory['total_collections']}")
        click.echo(f"   • Tags: {inventory['total_tags']}")

    # Show errors and warnings
    if validation_result.get("errors"):
        click.echo(f"\n❌ Errors ({len(validation_result['errors'])}):")
        for error in validation_result["errors"][:5]:  # Show first 5
            click.echo(f"   • {error}")
        if len(validation_result["errors"]) > 5:
            click.echo(f"   ... and {len(validation_result['errors']) - 5} more errors")

    if validation_result.get("warnings"):
        click.echo(f"\n⚠️  Warnings ({len(validation_result['warnings'])}):")
        for warning in validation_result["warnings"][:3]:  # Show first 3
            click.echo(f"   • {warning}")
        if len(validation_result["warnings"]) > 3:
            click.echo(
                f"   ... and {len(validation_result['warnings']) - 3} more warnings"
            )
