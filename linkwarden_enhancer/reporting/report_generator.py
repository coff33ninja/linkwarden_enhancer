"""Report generator for comprehensive change tracking and analysis"""

import json
import csv
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Supported report formats"""
    JSON = "json"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "md"


@dataclass
class ChangeRecord:
    """Represents a single change in the system"""
    change_id: str
    timestamp: datetime
    change_type: str  # 'added', 'modified', 'deleted', 'moved'
    item_type: str    # 'bookmark', 'collection', 'tag'
    item_id: str
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportSummary:
    """Summary statistics for a report"""
    total_changes: int
    changes_by_type: Dict[str, int]
    changes_by_item_type: Dict[str, int]
    time_range: Tuple[datetime, datetime]
    most_active_period: Optional[str] = None
    top_changed_items: List[Dict[str, Any]] = field(default_factory=list)


class ReportGenerator:
    """Generate comprehensive reports for change tracking and analysis"""
    
    def __init__(self, config: Dict[str, Any], data_dir: str = "data"):
        """Initialize report generator"""
        self.config = config
        self.data_dir = Path(data_dir)
        self.reports_dir = self.data_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Report settings
        self.max_report_history = config.get('reporting', {}).get('max_report_history', 100)
        self.default_format = ReportFormat(config.get('reporting', {}).get('default_format', 'json'))
        
        # Change tracking
        self.change_records = []
        self.report_history = []
        
        # Load existing data
        self._load_change_records()
        
        logger.info("Report generator initialized")
    
    def track_change(self, change_type: str, item_type: str, item_id: str,
                    before_state: Optional[Dict[str, Any]] = None,
                    after_state: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Track a change in the system"""
        
        try:
            change_id = f"change_{int(datetime.now().timestamp())}_{len(self.change_records)}"
            
            change_record = ChangeRecord(
                change_id=change_id,
                timestamp=datetime.now(),
                change_type=change_type,
                item_type=item_type,
                item_id=item_id,
                before_state=before_state,
                after_state=after_state,
                metadata=metadata or {}
            )
            
            self.change_records.append(change_record)
            
            # Save changes periodically
            if len(self.change_records) % 100 == 0:
                self._save_change_records()
            
            logger.debug(f"Tracked change: {change_type} {item_type} {item_id}")
            return change_id
            
        except Exception as e:
            logger.error(f"Failed to track change: {e}")
            return ""
    
    def generate_operation_report(self, operation_name: str, 
                                before_data: Dict[str, Any],
                                after_data: Dict[str, Any],
                                operation_metadata: Optional[Dict[str, Any]] = None,
                                formats: Optional[List[ReportFormat]] = None) -> Dict[str, str]:
        """Generate comprehensive report for an operation"""
        
        try:
            if formats is None:
                formats = [self.default_format]
            
            report_id = f"operation_report_{int(datetime.now().timestamp())}"
            
            logger.info(f"Generating operation report: {report_id} for {operation_name}")
            
            # Analyze changes
            changes_analysis = self._analyze_data_changes(before_data, after_data)
            
            # Create report data
            report_data = {
                'report_id': report_id,
                'report_type': 'operation',
                'operation_name': operation_name,
                'generated_at': datetime.now().isoformat(),
                'operation_metadata': operation_metadata or {},
                'summary': self._create_operation_summary(changes_analysis),
                'detailed_changes': changes_analysis,
                'statistics': self._calculate_operation_statistics(changes_analysis),
                'recommendations': self._generate_recommendations(changes_analysis)
            }
            
            # Generate reports in requested formats
            generated_files = {}
            for format_type in formats:
                file_path = self._save_report(report_data, format_type, report_id)
                if file_path:
                    generated_files[format_type.value] = file_path
            
            # Add to report history
            self._add_to_report_history(report_id, 'operation', operation_name, generated_files)
            
            logger.info(f"Operation report generated: {len(generated_files)} formats")
            return generated_files
            
        except Exception as e:
            logger.error(f"Failed to generate operation report: {e}")
            return {}
    
    def generate_period_report(self, start_date: datetime, end_date: datetime,
                             formats: Optional[List[ReportFormat]] = None) -> Dict[str, str]:
        """Generate report for a specific time period"""
        
        try:
            if formats is None:
                formats = [self.default_format]
            
            report_id = f"period_report_{int(datetime.now().timestamp())}"
            
            logger.info(f"Generating period report: {report_id} from {start_date} to {end_date}")
            
            # Filter changes by date range
            period_changes = [
                change for change in self.change_records
                if start_date <= change.timestamp <= end_date
            ]
            
            # Analyze period changes
            period_analysis = self._analyze_period_changes(period_changes, start_date, end_date)
            
            # Create report data
            report_data = {
                'report_id': report_id,
                'report_type': 'period',
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'generated_at': datetime.now().isoformat(),
                'summary': self._create_period_summary(period_analysis),
                'detailed_analysis': period_analysis,
                'trends': self._analyze_trends(period_changes),
                'statistics': self._calculate_period_statistics(period_changes)
            }
            
            # Generate reports in requested formats
            generated_files = {}
            for format_type in formats:
                file_path = self._save_report(report_data, format_type, report_id)
                if file_path:
                    generated_files[format_type.value] = file_path
            
            # Add to report history
            period_name = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
            self._add_to_report_history(report_id, 'period', period_name, generated_files)
            
            logger.info(f"Period report generated: {len(generated_files)} formats")
            return generated_files
            
        except Exception as e:
            logger.error(f"Failed to generate period report: {e}")
            return {}
    
    def generate_comparison_report(self, dataset1: Dict[str, Any], dataset2: Dict[str, Any],
                                 dataset1_name: str = "Dataset 1", dataset2_name: str = "Dataset 2",
                                 formats: Optional[List[ReportFormat]] = None) -> Dict[str, str]:
        """Generate comparison report between two datasets"""
        
        try:
            if formats is None:
                formats = [self.default_format]
            
            report_id = f"comparison_report_{int(datetime.now().timestamp())}"
            
            logger.info(f"Generating comparison report: {report_id}")
            
            # Perform detailed comparison
            comparison_analysis = self._perform_detailed_comparison(dataset1, dataset2)
            
            # Create report data
            report_data = {
                'report_id': report_id,
                'report_type': 'comparison',
                'dataset1_name': dataset1_name,
                'dataset2_name': dataset2_name,
                'generated_at': datetime.now().isoformat(),
                'summary': self._create_comparison_summary(comparison_analysis),
                'detailed_comparison': comparison_analysis,
                'statistics': self._calculate_comparison_statistics(comparison_analysis),
                'insights': self._generate_comparison_insights(comparison_analysis)
            }
            
            # Generate reports in requested formats
            generated_files = {}
            for format_type in formats:
                file_path = self._save_report(report_data, format_type, report_id)
                if file_path:
                    generated_files[format_type.value] = file_path
            
            # Add to report history
            comparison_name = f"{dataset1_name} vs {dataset2_name}"
            self._add_to_report_history(report_id, 'comparison', comparison_name, generated_files)
            
            logger.info(f"Comparison report generated: {len(generated_files)} formats")
            return generated_files
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            return {}   
 
    def _analyze_data_changes(self, before_data: Dict[str, Any], after_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze changes between before and after data"""
        
        try:
            changes_analysis = {
                'bookmarks': self._analyze_item_changes(
                    before_data.get('bookmarks', []),
                    after_data.get('bookmarks', []),
                    'id'
                ),
                'collections': self._analyze_item_changes(
                    before_data.get('collections', []),
                    after_data.get('collections', []),
                    'id'
                ),
                'tags': self._analyze_item_changes(
                    before_data.get('tags', []),
                    after_data.get('tags', []),
                    'id'
                )
            }
            
            return changes_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze data changes: {e}")
            return {}
    
    def _analyze_item_changes(self, before_items: List[Dict[str, Any]], 
                             after_items: List[Dict[str, Any]], 
                             id_field: str) -> Dict[str, Any]:
        """Analyze changes for a specific item type"""
        
        try:
            # Create ID-based lookups
            before_lookup = {item[id_field]: item for item in before_items}
            after_lookup = {item[id_field]: item for item in after_items}
            
            before_ids = set(before_lookup.keys())
            after_ids = set(after_lookup.keys())
            
            # Find changes
            added_ids = after_ids - before_ids
            removed_ids = before_ids - after_ids
            common_ids = before_ids & after_ids
            
            added_items = [after_lookup[item_id] for item_id in added_ids]
            removed_items = [before_lookup[item_id] for item_id in removed_ids]
            
            modified_items = []
            unchanged_items = []
            
            for item_id in common_ids:
                before_item = before_lookup[item_id]
                after_item = after_lookup[item_id]
                
                if self._items_are_different(before_item, after_item):
                    modified_items.append({
                        'id': item_id,
                        'before': before_item,
                        'after': after_item,
                        'changes': self._identify_field_changes(before_item, after_item)
                    })
                else:
                    unchanged_items.append(after_item)
            
            return {
                'added': added_items,
                'removed': removed_items,
                'modified': modified_items,
                'unchanged': unchanged_items,
                'summary': {
                    'added_count': len(added_items),
                    'removed_count': len(removed_items),
                    'modified_count': len(modified_items),
                    'unchanged_count': len(unchanged_items),
                    'total_before': len(before_items),
                    'total_after': len(after_items)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze item changes: {e}")
            return {}
    
    def _items_are_different(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """Check if two items are different (excluding timestamps)"""
        
        # Fields to ignore in comparison
        ignore_fields = {'updated_at', 'created_at', 'last_modified'}
        
        # Create copies without ignored fields
        item1_filtered = {k: v for k, v in item1.items() if k not in ignore_fields}
        item2_filtered = {k: v for k, v in item2.items() if k not in ignore_fields}
        
        return item1_filtered != item2_filtered
    
    def _identify_field_changes(self, before_item: Dict[str, Any], after_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific field changes between items"""
        
        changes = []
        
        try:
            all_fields = set(before_item.keys()) | set(after_item.keys())
            ignore_fields = {'updated_at', 'created_at', 'last_modified'}
            
            for field in all_fields:
                if field in ignore_fields:
                    continue
                
                before_value = before_item.get(field)
                after_value = after_item.get(field)
                
                if before_value != after_value:
                    changes.append({
                        'field': field,
                        'before': before_value,
                        'after': after_value,
                        'change_type': self._classify_field_change(before_value, after_value)
                    })
            
        except Exception as e:
            logger.warning(f"Failed to identify field changes: {e}")
        
        return changes
    
    def _classify_field_change(self, before_value: Any, after_value: Any) -> str:
        """Classify the type of field change"""
        
        if before_value is None and after_value is not None:
            return 'added'
        elif before_value is not None and after_value is None:
            return 'removed'
        elif before_value != after_value:
            return 'modified'
        else:
            return 'unchanged'
    
    def _create_operation_summary(self, changes_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary for operation report"""
        
        try:
            total_changes = 0
            changes_by_type = {'added': 0, 'removed': 0, 'modified': 0}
            changes_by_item = {}
            
            for item_type, analysis in changes_analysis.items():
                if 'summary' in analysis:
                    summary = analysis['summary']
                    changes_by_item[item_type] = summary
                    
                    total_changes += summary['added_count'] + summary['removed_count'] + summary['modified_count']
                    changes_by_type['added'] += summary['added_count']
                    changes_by_type['removed'] += summary['removed_count']
                    changes_by_type['modified'] += summary['modified_count']
            
            return {
                'total_changes': total_changes,
                'changes_by_type': changes_by_type,
                'changes_by_item': changes_by_item,
                'impact_level': self._assess_impact_level(total_changes, changes_by_type)
            }
            
        except Exception as e:
            logger.error(f"Failed to create operation summary: {e}")
            return {}
    
    def _assess_impact_level(self, total_changes: int, changes_by_type: Dict[str, int]) -> str:
        """Assess the impact level of changes"""
        
        if total_changes == 0:
            return 'none'
        elif total_changes < 10:
            return 'low'
        elif total_changes < 100:
            return 'medium'
        elif changes_by_type['removed'] > total_changes * 0.1:  # More than 10% deletions
            return 'critical'
        else:
            return 'high'
    
    def _calculate_operation_statistics(self, changes_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed statistics for operation"""
        
        try:
            statistics = {
                'processing_efficiency': {},
                'data_quality_impact': {},
                'relationship_changes': {}
            }
            
            # Calculate processing efficiency
            for item_type, analysis in changes_analysis.items():
                if 'summary' in analysis:
                    summary = analysis['summary']
                    total_processed = summary['total_before']
                    
                    if total_processed > 0:
                        statistics['processing_efficiency'][item_type] = {
                            'change_rate': (summary['added_count'] + summary['removed_count'] + summary['modified_count']) / total_processed,
                            'retention_rate': summary['unchanged_count'] / total_processed,
                            'growth_rate': (summary['total_after'] - summary['total_before']) / summary['total_before']
                        }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Failed to calculate operation statistics: {e}")
            return {}
    
    def _generate_recommendations(self, changes_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on changes analysis"""
        
        recommendations = []
        
        try:
            # Analyze patterns and generate recommendations
            for item_type, analysis in changes_analysis.items():
                if 'summary' in analysis:
                    summary = analysis['summary']
                    
                    # High deletion rate warning
                    if summary['total_before'] > 0:
                        deletion_rate = summary['removed_count'] / summary['total_before']
                        if deletion_rate > 0.1:
                            recommendations.append(f"High deletion rate detected for {item_type} ({deletion_rate:.1%}). Consider reviewing deletion criteria.")
                    
                    # Large number of modifications
                    if summary['modified_count'] > 50:
                        recommendations.append(f"Large number of {item_type} modifications ({summary['modified_count']}). Consider batch processing optimization.")
                    
                    # No changes detected
                    if summary['added_count'] + summary['removed_count'] + summary['modified_count'] == 0:
                        recommendations.append(f"No changes detected for {item_type}. Verify operation parameters.")
            
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    def _save_report(self, report_data: Dict[str, Any], format_type: ReportFormat, report_id: str) -> Optional[str]:
        """Save report in specified format"""
        
        try:
            if format_type == ReportFormat.JSON:
                return self._save_json_report(report_data, report_id)
            elif format_type == ReportFormat.HTML:
                return self._save_html_report(report_data, report_id)
            elif format_type == ReportFormat.CSV:
                return self._save_csv_report(report_data, report_id)
            elif format_type == ReportFormat.MARKDOWN:
                return self._save_markdown_report(report_data, report_id)
            else:
                logger.error(f"Unsupported report format: {format_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save report in {format_type.value} format: {e}")
            return None
    
    def _save_json_report(self, report_data: Dict[str, Any], report_id: str) -> str:
        """Save report as JSON"""
        
        file_path = self.reports_dir / f"{report_id}.json"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(file_path)
    
    def _save_html_report(self, report_data: Dict[str, Any], report_id: str) -> str:
        """Save report as HTML"""
        
        file_path = self.reports_dir / f"{report_id}.html"
        
        html_content = self._generate_html_content(report_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(file_path)
    
    def _save_csv_report(self, report_data: Dict[str, Any], report_id: str) -> str:
        """Save report as CSV"""
        
        file_path = self.reports_dir / f"{report_id}.csv"
        
        # Extract tabular data for CSV
        csv_data = self._extract_csv_data(report_data)
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            if csv_data:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
        
        return str(file_path)
    
    def _save_markdown_report(self, report_data: Dict[str, Any], report_id: str) -> str:
        """Save report as Markdown"""
        
        file_path = self.reports_dir / f"{report_id}.md"
        
        markdown_content = self._generate_markdown_content(report_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return str(file_path)
    
    def _generate_html_content(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML content for report"""
        
        # Basic HTML template - can be enhanced with CSS styling
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{report_data.get('report_type', 'Report').title()} Report - {report_data.get('report_id', '')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{report_data.get('report_type', 'Report').title()} Report</h1>
        <p><strong>Report ID:</strong> {report_data.get('report_id', 'N/A')}</p>
        <p><strong>Generated:</strong> {report_data.get('generated_at', 'N/A')}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="summary">
            {self._format_summary_html(report_data.get('summary', {}))}
        </div>
    </div>
    
    <div class="section">
        <h2>Statistics</h2>
        {self._format_statistics_html(report_data.get('statistics', {}))}
    </div>
</body>
</html>
"""
        return html
    
    def _format_summary_html(self, summary: Dict[str, Any]) -> str:
        """Format summary data as HTML"""
        
        html_parts = []
        
        for key, value in summary.items():
            if isinstance(value, dict):
                html_parts.append(f"<h4>{key.replace('_', ' ').title()}</h4>")
                for sub_key, sub_value in value.items():
                    html_parts.append(f"<div class='metric'><strong>{sub_key.replace('_', ' ').title()}:</strong> {sub_value}</div>")
            else:
                html_parts.append(f"<div class='metric'><strong>{key.replace('_', ' ').title()}:</strong> {value}</div>")
        
        return ''.join(html_parts)
    
    def _format_statistics_html(self, statistics: Dict[str, Any]) -> str:
        """Format statistics data as HTML"""
        
        html_parts = []
        
        for section, data in statistics.items():
            html_parts.append(f"<h3>{section.replace('_', ' ').title()}</h3>")
            
            if isinstance(data, dict):
                html_parts.append("<table>")
                html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
                
                for key, value in data.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            html_parts.append(f"<tr><td>{key} - {sub_key}</td><td>{sub_value}</td></tr>")
                    else:
                        html_parts.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
                
                html_parts.append("</table>")
        
        return ''.join(html_parts)
    
    def _extract_csv_data(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data suitable for CSV format"""
        
        csv_rows = []
        
        # Extract summary data
        if 'summary' in report_data:
            summary = report_data['summary']
            
            if isinstance(summary, dict):
                for key, value in summary.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            csv_rows.append({
                                'Category': key,
                                'Metric': sub_key,
                                'Value': sub_value,
                                'Type': 'Summary'
                            })
                    else:
                        csv_rows.append({
                            'Category': 'General',
                            'Metric': key,
                            'Value': value,
                            'Type': 'Summary'
                        })
        
        return csv_rows
    
    def _generate_markdown_content(self, report_data: Dict[str, Any]) -> str:
        """Generate Markdown content for report"""
        
        markdown_parts = [
            f"# {report_data.get('report_type', 'Report').title()} Report",
            "",
            f"**Report ID:** {report_data.get('report_id', 'N/A')}",
            f"**Generated:** {report_data.get('generated_at', 'N/A')}",
            ""
        ]
        
        # Add summary section
        if 'summary' in report_data:
            markdown_parts.extend([
                "## Summary",
                "",
                self._format_summary_markdown(report_data['summary']),
                ""
            ])
        
        # Add statistics section
        if 'statistics' in report_data:
            markdown_parts.extend([
                "## Statistics",
                "",
                self._format_statistics_markdown(report_data['statistics']),
                ""
            ])
        
        return '\n'.join(markdown_parts)
    
    def _format_summary_markdown(self, summary: Dict[str, Any]) -> str:
        """Format summary data as Markdown"""
        
        markdown_parts = []
        
        for key, value in summary.items():
            if isinstance(value, dict):
                markdown_parts.append(f"### {key.replace('_', ' ').title()}")
                markdown_parts.append("")
                
                for sub_key, sub_value in value.items():
                    markdown_parts.append(f"- **{sub_key.replace('_', ' ').title()}:** {sub_value}")
                
                markdown_parts.append("")
            else:
                markdown_parts.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        
        return '\n'.join(markdown_parts)
    
    def _format_statistics_markdown(self, statistics: Dict[str, Any]) -> str:
        """Format statistics data as Markdown"""
        
        markdown_parts = []
        
        for section, data in statistics.items():
            markdown_parts.append(f"### {section.replace('_', ' ').title()}")
            markdown_parts.append("")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        markdown_parts.append(f"#### {key.replace('_', ' ').title()}")
                        for sub_key, sub_value in value.items():
                            markdown_parts.append(f"- **{sub_key.replace('_', ' ').title()}:** {sub_value}")
                    else:
                        markdown_parts.append(f"- **{key.replace('_', ' ').title()}:** {value}")
            
            markdown_parts.append("")
        
        return '\n'.join(markdown_parts)
    
    def _add_to_report_history(self, report_id: str, report_type: str, 
                              subject: str, generated_files: Dict[str, str]) -> None:
        """Add report to history"""
        
        try:
            history_entry = {
                'report_id': report_id,
                'report_type': report_type,
                'subject': subject,
                'generated_at': datetime.now().isoformat(),
                'formats': list(generated_files.keys()),
                'files': generated_files
            }
            
            self.report_history.append(history_entry)
            
            # Limit history size
            if len(self.report_history) > self.max_report_history:
                self.report_history = self.report_history[-self.max_report_history:]
            
            self._save_report_history()
            
        except Exception as e:
            logger.warning(f"Failed to add report to history: {e}")
    
    def _save_change_records(self) -> None:
        """Save change records to file"""
        
        try:
            records_data = []
            
            for record in self.change_records:
                records_data.append({
                    'change_id': record.change_id,
                    'timestamp': record.timestamp.isoformat(),
                    'change_type': record.change_type,
                    'item_type': record.item_type,
                    'item_id': record.item_id,
                    'before_state': record.before_state,
                    'after_state': record.after_state,
                    'metadata': record.metadata
                })
            
            records_file = self.data_dir / 'change_records.json'
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(records_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(records_data)} change records")
            
        except Exception as e:
            logger.error(f"Failed to save change records: {e}")
    
    def _load_change_records(self) -> None:
        """Load change records from file"""
        
        try:
            records_file = self.data_dir / 'change_records.json'
            
            if not records_file.exists():
                logger.info("No existing change records found")
                return
            
            with open(records_file, 'r', encoding='utf-8') as f:
                records_data = json.load(f)
            
            for record_data in records_data:
                record = ChangeRecord(
                    change_id=record_data['change_id'],
                    timestamp=datetime.fromisoformat(record_data['timestamp']),
                    change_type=record_data['change_type'],
                    item_type=record_data['item_type'],
                    item_id=record_data['item_id'],
                    before_state=record_data.get('before_state'),
                    after_state=record_data.get('after_state'),
                    metadata=record_data.get('metadata', {})
                )
                self.change_records.append(record)
            
            logger.info(f"Loaded {len(self.change_records)} change records")
            
        except Exception as e:
            logger.warning(f"Failed to load change records: {e}")
    
    def _save_report_history(self) -> None:
        """Save report history to file"""
        
        try:
            history_file = self.data_dir / 'report_history.json'
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.report_history, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved report history: {len(self.report_history)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save report history: {e}")
    
    def get_report_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reporting statistics"""
        
        try:
            return {
                'total_reports_generated': len(self.report_history),
                'total_changes_tracked': len(self.change_records),
                'reports_by_type': self._count_reports_by_type(),
                'recent_activity': self._get_recent_activity(),
                'storage_usage': self._calculate_storage_usage()
            }
            
        except Exception as e:
            logger.error(f"Failed to get report statistics: {e}")
            return {'error': str(e)}
    
    def _count_reports_by_type(self) -> Dict[str, int]:
        """Count reports by type"""
        
        counts = {}
        for report in self.report_history:
            report_type = report.get('report_type', 'unknown')
            counts[report_type] = counts.get(report_type, 0) + 1
        
        return counts
    
    def _get_recent_activity(self) -> Dict[str, Any]:
        """Get recent reporting activity"""
        
        now = datetime.now()
        recent_reports = [
            report for report in self.report_history
            if (now - datetime.fromisoformat(report['generated_at'])).days < 7
        ]
        
        recent_changes = [
            change for change in self.change_records
            if (now - change.timestamp).days < 7
        ]
        
        return {
            'reports_last_7_days': len(recent_reports),
            'changes_last_7_days': len(recent_changes),
            'last_report_generated': self.report_history[-1]['generated_at'] if self.report_history else None
        }
    
    def _calculate_storage_usage(self) -> Dict[str, Any]:
        """Calculate storage usage for reports"""
        
        try:
            total_size = 0
            file_count = 0
            
            for report_file in self.reports_dir.glob('*'):
                if report_file.is_file():
                    total_size += report_file.stat().st_size
                    file_count += 1
            
            return {
                'total_files': file_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'reports_directory': str(self.reports_dir)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate storage usage: {e}")
            return {}
    
    def cleanup_old_reports(self, days_to_keep: int = 30) -> int:
        """Clean up old report files"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            removed_count = 0
            
            # Remove old report files
            for report_file in self.reports_dir.glob('*'):
                if report_file.is_file():
                    file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        report_file.unlink()
                        removed_count += 1
            
            # Clean up history
            self.report_history = [
                report for report in self.report_history
                if (datetime.now() - datetime.fromisoformat(report['generated_at'])).days <= days_to_keep
            ]
            
            self._save_report_history()
            
            logger.info(f"Cleaned up {removed_count} old report files")
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old reports: {e}")
            return 0