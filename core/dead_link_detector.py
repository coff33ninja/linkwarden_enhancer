"""
Dead Link Detection and Management System

This module provides comprehensive dead link detection, categorization, and management
capabilities for bookmark collections.
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
import logging
from datetime import datetime, timedelta
import json
import time

from ..utils.logging_utils import EnhancedLogger
from ..utils.progress_utils import EnhancedProgressTracker
from ..data_models import LinkwardenBookmark

logger = logging.getLogger(__name__)

class DeadLinkDetector:
    """Advanced dead link detection with intelligent categorization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = EnhancedLogger("DeadLinkDetector", verbose=self.config.get('verbose', False))
        self.progress = EnhancedProgressTracker(verbose=self.config.get('verbose', False))
        
        # Detection settings
        self.timeout = self.config.get('timeout', 10)
        self.max_retries = self.config.get('max_retries', 2)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.concurrent_requests = self.config.get('concurrent_requests', 10)
        self.user_agent = self.config.get('user_agent', 'Linkwarden-Enhancer-DeadLinkDetector/1.0')
        
        # Status categorization
        self.dead_status_codes = {
            404: "Not Found",
            403: "Forbidden", 
            410: "Gone",
            500: "Server Error",
            502: "Bad Gateway",
            503: "Service Unavailable",
            504: "Gateway Timeout"
        }
        
        self.suspicious_status_codes = {
            401: "Unauthorized",
            429: "Rate Limited",
            451: "Unavailable for Legal Reasons"
        }
        
        # Results storage
        self.dead_links = []
        self.suspicious_links = []
        self.working_links = []
        self.check_results = {}
        
    async def check_bookmarks(self, bookmarks: List[LinkwardenBookmark]) -> Dict[str, Any]:
        """Check all bookmarks for dead links with progress tracking"""
        
        self.logger.log_operation("DEAD_LINK_CHECK_START", {
            "total_bookmarks": len(bookmarks),
            "concurrent_requests": self.concurrent_requests,
            "timeout": self.timeout
        })
        
        self.progress.start_phase("Dead Link Detection", len(bookmarks))
        
        try:
            # Create semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.concurrent_requests)
            
            # Create tasks for all bookmark checks
            tasks = []
            for bookmark in bookmarks:
                if bookmark.url:
                    task = self._check_single_bookmark(semaphore, bookmark)
                    tasks.append(task)
            
            # Execute all checks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.log_error(f"Failed to check bookmark: {result}")
                    continue
                    
                bookmark_result, status_category = result
                self._categorize_result(bookmark_result, status_category)
                self.progress.update_progress(1)
            
            self.progress.finish_phase()
            
            # Generate summary
            summary = self._generate_summary()
            
            self.logger.log_operation("DEAD_LINK_CHECK_COMPLETE", summary)
            
            return {
                "summary": summary,
                "dead_links": self.dead_links,
                "suspicious_links": self.suspicious_links,
                "working_links": self.working_links,
                "detailed_results": self.check_results
            }
            
        except Exception as e:
            self.logger.log_error(f"Dead link check failed: {e}")
            raise
    
    async def _check_single_bookmark(self, semaphore: asyncio.Semaphore, 
                                   bookmark: LinkwardenBookmark) -> Tuple[Dict[str, Any], str]:
        """Check a single bookmark with retry logic"""
        
        async with semaphore:
            url = bookmark.url
            bookmark_result = {
                "bookmark_id": bookmark.id,
                "name": bookmark.name,
                "url": url,
                "collection_id": getattr(bookmark, 'collection_id', None),
                "check_timestamp": datetime.now().isoformat(),
                "attempts": []
            }
            
            for attempt in range(self.max_retries + 1):
                try:
                    async with aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        headers={'User-Agent': self.user_agent}
                    ) as session:
                        
                        start_time = time.time()
                        async with session.head(url, allow_redirects=True) as response:
                            response_time = time.time() - start_time
                            
                            attempt_result = {
                                "attempt": attempt + 1,
                                "status_code": response.status,
                                "response_time": response_time,
                                "final_url": str(response.url),
                                "headers": dict(response.headers)
                            }
                            
                            bookmark_result["attempts"].append(attempt_result)
                            
                            # Determine status category
                            if response.status == 200:
                                bookmark_result["final_status"] = "working"
                                return bookmark_result, "working"
                            elif response.status in self.dead_status_codes:
                                bookmark_result["final_status"] = "dead"
                                bookmark_result["reason"] = self.dead_status_codes[response.status]
                                return bookmark_result, "dead"
                            elif response.status in self.suspicious_status_codes:
                                bookmark_result["final_status"] = "suspicious"
                                bookmark_result["reason"] = self.suspicious_status_codes[response.status]
                                return bookmark_result, "suspicious"
                            else:
                                # Retry for other status codes
                                if attempt < self.max_retries:
                                    await asyncio.sleep(self.retry_delay)
                                    continue
                                else:
                                    bookmark_result["final_status"] = "suspicious"
                                    bookmark_result["reason"] = f"Unexpected status: {response.status}"
                                    return bookmark_result, "suspicious"
                
                except asyncio.TimeoutError:
                    attempt_result = {
                        "attempt": attempt + 1,
                        "error": "timeout",
                        "timeout_duration": self.timeout
                    }
                    bookmark_result["attempts"].append(attempt_result)
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    else:
                        bookmark_result["final_status"] = "dead"
                        bookmark_result["reason"] = "Connection timeout"
                        return bookmark_result, "dead"
                
                except Exception as e:
                    attempt_result = {
                        "attempt": attempt + 1,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    bookmark_result["attempts"].append(attempt_result)
                    
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    else:
                        bookmark_result["final_status"] = "dead"
                        bookmark_result["reason"] = f"Connection error: {str(e)}"
                        return bookmark_result, "dead"
            
            # Should not reach here, but fallback
            bookmark_result["final_status"] = "dead"
            bookmark_result["reason"] = "All attempts failed"
            return bookmark_result, "dead"
    
    def _categorize_result(self, bookmark_result: Dict[str, Any], status_category: str):
        """Categorize bookmark result into appropriate list"""
        
        self.check_results[bookmark_result["bookmark_id"]] = bookmark_result
        
        if status_category == "dead":
            self.dead_links.append(bookmark_result)
        elif status_category == "suspicious":
            self.suspicious_links.append(bookmark_result)
        else:
            self.working_links.append(bookmark_result)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        total_checked = len(self.dead_links) + len(self.suspicious_links) + len(self.working_links)
        
        return {
            "total_checked": total_checked,
            "working_count": len(self.working_links),
            "dead_count": len(self.dead_links),
            "suspicious_count": len(self.suspicious_links),
            "working_percentage": (len(self.working_links) / total_checked * 100) if total_checked > 0 else 0,
            "dead_percentage": (len(self.dead_links) / total_checked * 100) if total_checked > 0 else 0,
            "suspicious_percentage": (len(self.suspicious_links) / total_checked * 100) if total_checked > 0 else 0,
            "check_timestamp": datetime.now().isoformat()
        }
    
    def create_dead_links_collection(self, collection_name: str = "🔗 Dead Links") -> Dict[str, Any]:
        """Create a collection structure for dead links"""
        
        return {
            "name": collection_name,
            "description": f"Links that are no longer accessible. Detected on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "icon": "LinkSlash",
            "color": "#ef4444",  # Red color for dead links
            "links": self.dead_links
        }
    
    def create_suspicious_links_collection(self, collection_name: str = "⚠️ Suspicious Links") -> Dict[str, Any]:
        """Create a collection structure for suspicious links"""
        
        return {
            "name": collection_name,
            "description": f"Links that may have issues or require attention. Detected on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "icon": "AlertTriangle",
            "color": "#f59e0b",  # Amber color for suspicious links
            "links": self.suspicious_links
        }
    
    def generate_report(self, format: str = "json") -> str:
        """Generate detailed report of dead link detection"""
        
        report_data = {
            "report_type": "dead_link_detection",
            "generated_at": datetime.now().isoformat(),
            "summary": self._generate_summary(),
            "dead_links": self.dead_links,
            "suspicious_links": self.suspicious_links,
            "configuration": {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "concurrent_requests": self.concurrent_requests
            }
        }
        
        if format.lower() == "json":
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        elif format.lower() == "html":
            return self._generate_html_report(report_data)
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dead Link Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .dead {{ color: #ef4444; }}
                .suspicious {{ color: #f59e0b; }}
                .working {{ color: #10b981; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Dead Link Detection Report</h1>
            <p>Generated: {report_data['generated_at']}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Links Checked: {report_data['summary']['total_checked']}</p>
                <p class="working">Working Links: {report_data['summary']['working_count']} ({report_data['summary']['working_percentage']:.1f}%)</p>
                <p class="dead">Dead Links: {report_data['summary']['dead_count']} ({report_data['summary']['dead_percentage']:.1f}%)</p>
                <p class="suspicious">Suspicious Links: {report_data['summary']['suspicious_count']} ({report_data['summary']['suspicious_percentage']:.1f}%)</p>
            </div>
            
            <h2 class="dead">Dead Links ({len(report_data['dead_links'])})</h2>
            <table>
                <tr><th>Name</th><th>URL</th><th>Reason</th></tr>
        """
        
        for link in report_data['dead_links']:
            html += f"<tr><td>{link['name']}</td><td>{link['url']}</td><td>{link.get('reason', 'Unknown')}</td></tr>"
        
        html += """
            </table>
            
            <h2 class="suspicious">Suspicious Links ({len(report_data['suspicious_links'])})</h2>
            <table>
                <tr><th>Name</th><th>URL</th><th>Reason</th></tr>
        """
        
        for link in report_data['suspicious_links']:
            html += f"<tr><td>{link['name']}</td><td>{link['url']}</td><td>{link.get('reason', 'Unknown')}</td></tr>"
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html


class DeadLinkManager:
    """Manages dead link collections and cleanup operations"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = EnhancedLogger("DeadLinkManager", verbose=self.config.get('verbose', False))
    
    def organize_dead_links(self, bookmarks: List[LinkwardenBookmark], 
                          dead_link_results: Dict[str, Any]) -> Dict[str, Any]:
        """Organize bookmarks by moving dead links to appropriate collections"""
        
        self.logger.log_operation("ORGANIZE_DEAD_LINKS_START", {
            "total_bookmarks": len(bookmarks),
            "dead_count": len(dead_link_results['dead_links']),
            "suspicious_count": len(dead_link_results['suspicious_links'])
        })
        
        # Create collections for dead and suspicious links
        dead_collection = {
            "name": "🔗 Dead Links",
            "description": f"Links that are no longer accessible. Auto-detected on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "icon": "LinkSlash",
            "color": "#ef4444",
            "links": []
        }
        
        suspicious_collection = {
            "name": "⚠️ Suspicious Links", 
            "description": f"Links that may have issues or require manual review. Auto-detected on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "icon": "AlertTriangle",
            "color": "#f59e0b",
            "links": []
        }
        
        # Create lookup for dead/suspicious bookmark IDs
        dead_ids = {link['bookmark_id'] for link in dead_link_results['dead_links']}
        suspicious_ids = {link['bookmark_id'] for link in dead_link_results['suspicious_links']}
        
        # Organize bookmarks
        organized_bookmarks = []
        moved_count = 0
        
        for bookmark in bookmarks:
            if bookmark.id in dead_ids:
                # Add to dead collection
                dead_collection['links'].append(bookmark)
                moved_count += 1
            elif bookmark.id in suspicious_ids:
                # Add to suspicious collection
                suspicious_collection['links'].append(bookmark)
                moved_count += 1
            else:
                # Keep in original location
                organized_bookmarks.append(bookmark)
        
        result = {
            "organized_bookmarks": organized_bookmarks,
            "dead_collection": dead_collection,
            "suspicious_collection": suspicious_collection,
            "moved_count": moved_count,
            "organization_summary": {
                "total_processed": len(bookmarks),
                "kept_in_place": len(organized_bookmarks),
                "moved_to_dead": len(dead_collection['links']),
                "moved_to_suspicious": len(suspicious_collection['links']),
                "total_moved": moved_count
            }
        }
        
        self.logger.log_operation("ORGANIZE_DEAD_LINKS_COMPLETE", result["organization_summary"])
        
        return result