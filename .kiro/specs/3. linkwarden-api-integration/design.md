# Design Document

## Overview

The Linkwarden API integration provides comprehensive connectivity to Linkwarden instances through REST API endpoints, enabling seamless import/export and bidirectional sync capabilities. The system implements robust error handling, batch processing, and data integrity validation while supporting various sync strategies and conflict resolution mechanisms.

## Architecture

### API Integration Architecture (Building on Existing)

**Current Architecture Integration:**
- **Extends existing `importers/` package** with direct API capabilities
- **Leverages existing `core/` safety features** for backup and validation
- **Uses existing `reporting/` system** for monitoring and metrics

```
# EXISTING MODULES (✅ Already implemented)
importers/
├── linkwarden_importer.py       # ✅ Linkwarden backup JSON importer
├── github_importer.py           # ✅ GitHub API integration (reference)
├── universal_importer.py        # ✅ Multi-source import orchestrator
└── base_importer.py             # ✅ Base importer interface

core/
├── safety_manager.py            # ✅ Safety orchestration
├── backup_system.py             # ✅ Backup management
├── validation_engine.py         # ✅ Data validation
└── progress_monitor.py          # ✅ Progress tracking

reporting/
├── metrics_collector.py         # ✅ Performance metrics
└── report_generator.py          # ✅ Comprehensive reporting

# NEW ADDITIONS (❌ To be added)
api/                             # ❌ New package for direct API integration
├── __init__.py
├── base_client.py               # ❌ Base API client framework
├── linkwarden_client.py         # ❌ Direct Linkwarden API client
├── api_manager.py               # ❌ API client management
├── rate_limiter.py              # ❌ Rate limiting and throttling
├── retry_handler.py             # ❌ Retry logic and error recovery
└── response_validator.py        # ❌ API response validation

# ENHANCED IMPORTERS (🔄 Extend existing)
importers/
├── linkwarden_api_importer.py   # ❌ New API-based importer
└── enhanced_universal_importer.py # 🔄 Extend with API support
```

### API Integration Flow

```
CLI Commands → API Manager → Linkwarden Client → Rate Limiter → Linkwarden API
     ↓              ↓              ↓               ↓              ↓
Configuration   Client Factory   HTTP Client   Throttling    REST Endpoints
     ↓              ↓              ↓               ↓              ↓
Validation     Connection Pool   Request/Response  Retry Logic   JSON Data
     ↓              ↓              ↓               ↓              ↓
Data Mapping   Error Handling   Data Validation  Monitoring    Database
```

## Components and Interfaces

### Linkwarden API Client

```python
class LinkwardenClient(BaseAPIClient):
    def __init__(self, base_url: str, api_key: str, 
                 rate_limiter: RateLimiter = None,
                 retry_handler: RetryHandler = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = self._create_session()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.retry_handler = retry_handler or RetryHandler()
        self.response_validator = ResponseValidator()
    
    # Link operations
    async def get_all_links(self, limit: int = 1000, 
                           offset: int = 0, 
                           collection_id: Optional[int] = None) -> List[Dict]:
        """Get all links with pagination support"""
        
    async def get_link(self, link_id: int) -> Dict:
        """Get specific link by ID"""
        
    async def create_link(self, link_data: Dict) -> Dict:
        """Create new link"""
        
    async def update_link(self, link_id: int, link_data: Dict) -> Dict:
        """Update existing link"""
        
    async def delete_link(self, link_id: int) -> bool:
        """Delete link by ID"""
    
    # Collection operations
    async def get_all_collections(self) -> List[Dict]:
        """Get all collections with hierarchy"""
        
    async def get_collection(self, collection_id: int) -> Dict:
        """Get specific collection by ID"""
        
    async def create_collection(self, collection_data: Dict) -> Dict:
        """Create new collection"""
        
    async def update_collection(self, collection_id: int, 
                               collection_data: Dict) -> Dict:
        """Update existing collection"""
        
    async def delete_collection(self, collection_id: int) -> bool:
        """Delete collection by ID"""
    
    # Tag operations
    async def get_all_tags(self) -> List[Dict]:
        """Get all tags"""
        
    async def create_tag(self, tag_data: Dict) -> Dict:
        """Create new tag"""
        
    async def update_tag(self, tag_id: int, tag_data: Dict) -> Dict:
        """Update existing tag"""
    
    # Bulk operations
    async def bulk_import_links(self, links: List[Dict], 
                               batch_size: int = 50) -> BulkOperationResult:
        """Import multiple links in batches"""
        
    async def bulk_export_links(self, collection_ids: List[int] = None,
                               since: datetime = None) -> List[Dict]:
        """Export links with optional filtering"""
        
    async def bulk_update_links(self, updates: List[Dict],
                               batch_size: int = 50) -> BulkOperationResult:
        """Update multiple links in batches"""
    
    # Connection and health
    async def test_connection(self) -> ConnectionTestResult:
        """Test API connection and authentication"""
        
    async def get_api_info(self) -> Dict:
        """Get API version and instance information"""

class BaseAPIClient:
    def __init__(self):
        self.session = None
        self.last_request_time = None
        self.request_count = 0
        
    def _create_session(self) -> aiohttp.ClientSession:
        """Create HTTP session with appropriate headers and timeouts"""
        
    async def _make_request(self, method: str, endpoint: str, 
                           data: Dict = None, 
                           params: Dict = None) -> Dict:
        """Make HTTP request with error handling and validation"""
        
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Handle API response with error checking"""
        
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including authentication"""
```

### Data Mapping and Transformation

```python
class LinkwardenMapper:
    @staticmethod
    def to_internal_format(linkwarden_data: Dict) -> Bookmark:
        """Convert Linkwarden API response to internal Bookmark format"""
        return Bookmark(
            id=linkwarden_data.get('id'),
            url=linkwarden_data.get('url'),
            title=linkwarden_data.get('name', ''),
            description=linkwarden_data.get('description', ''),
            tags=[tag.get('name', '') for tag in linkwarden_data.get('tags', [])],
            collection_id=linkwarden_data.get('collectionId'),
            created_at=datetime.fromisoformat(linkwarden_data.get('createdAt', '')),
            updated_at=datetime.fromisoformat(linkwarden_data.get('updatedAt', '')),
            metadata=linkwarden_data.get('metadata', {})
        )
    
    @staticmethod
    def to_linkwarden_format(bookmark: Bookmark) -> Dict:
        """Convert internal Bookmark to Linkwarden API format"""
        return {
            'url': bookmark.url,
            'name': bookmark.title,
            'description': bookmark.description,
            'tags': [{'name': tag} for tag in bookmark.tags],
            'collectionId': bookmark.collection_id,
            'metadata': bookmark.metadata
        }
    
    @staticmethod
    def map_collection(collection_data: Dict) -> Collection:
        """Map Linkwarden collection to internal format"""
        return Collection(
            id=collection_data.get('id'),
            name=collection_data.get('name', ''),
            description=collection_data.get('description', ''),
            parent_id=collection_data.get('parentId'),
            created_at=datetime.fromisoformat(collection_data.get('createdAt', '')),
            updated_at=datetime.fromisoformat(collection_data.get('updatedAt', ''))
        )
    
    @staticmethod
    def map_tag(tag_data: Dict) -> Tag:
        """Map Linkwarden tag to internal format"""
        return Tag(
            id=tag_data.get('id'),
            name=tag_data.get('name', ''),
            created_at=datetime.fromisoformat(tag_data.get('createdAt', ''))
        )

class DataTransformer:
    def __init__(self):
        self.linkwarden_mapper = LinkwardenMapper()
        
    def transform_import_data(self, linkwarden_data: List[Dict]) -> List[Bookmark]:
        """Transform imported Linkwarden data to internal format"""
        
    def transform_export_data(self, bookmarks: List[Bookmark]) -> List[Dict]:
        """Transform internal bookmarks to Linkwarden format"""
        
    def merge_bookmark_data(self, original: Bookmark, 
                           enhanced: Bookmark, 
                           strategy: str = "smart") -> Bookmark:
        """Merge original and enhanced bookmark data"""
```

### Rate Limiting and Error Handling

```python
class RateLimiter:
    def __init__(self, requests_per_minute: int = 100, 
                 burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.request_times = deque()
        self.burst_count = 0
        self.last_reset = time.time()
    
    async def acquire(self) -> None:
        """Acquire rate limit token, blocking if necessary"""
        
    def _cleanup_old_requests(self) -> None:
        """Remove old request timestamps"""
        
    def _calculate_delay(self) -> float:
        """Calculate delay needed to respect rate limits"""

class RetryHandler:
    def __init__(self, max_retries: int = 3, 
                 base_delay: float = 1.0,
                 max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def execute_with_retry(self, operation: Callable, 
                                *args, **kwargs) -> Any:
        """Execute operation with exponential backoff retry"""
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        
    def _should_retry(self, exception: Exception) -> bool:
        """Determine if exception is retryable"""

class LinkwardenAPIError(Exception):
    def __init__(self, status_code: int, message: str, 
                 response: Dict = None, endpoint: str = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        self.endpoint = endpoint
        super().__init__(f"Linkwarden API Error {status_code}: {message}")

class APIErrorHandler:
    @staticmethod
    def handle_api_response(response: aiohttp.ClientResponse, 
                           content: Dict) -> None:
        """Handle API response and raise appropriate exceptions"""
        if response.status == 200:
            return
        elif response.status == 401:
            raise LinkwardenAPIError(401, "Invalid API key or unauthorized access")
        elif response.status == 403:
            raise LinkwardenAPIError(403, "Forbidden - insufficient permissions")
        elif response.status == 404:
            raise LinkwardenAPIError(404, "Resource not found")
        elif response.status == 429:
            raise LinkwardenAPIError(429, "Rate limit exceeded")
        elif response.status >= 500:
            raise LinkwardenAPIError(response.status, "Server error")
        else:
            raise LinkwardenAPIError(response.status, content.get('message', 'Unknown error'))
```

### Sync Engine and Conflict Resolution

```python
class SyncEngine:
    def __init__(self, client: LinkwardenClient, 
                 conflict_resolver: ConflictResolver,
                 merge_strategy: MergeStrategy):
        self.client = client
        self.conflict_resolver = conflict_resolver
        self.merge_strategy = merge_strategy
        self.sync_state = SyncState()
        self.change_detector = ChangeDetector()
    
    async def sync_bidirectional(self, local_data: List[Bookmark],
                                enhanced_data: List[Bookmark] = None) -> SyncResult:
        """Perform bidirectional sync between local and remote data"""
        
    async def import_from_linkwarden(self, collection_ids: List[int] = None,
                                   since: datetime = None) -> ImportResult:
        """Import data from Linkwarden with optional filtering"""
        
    async def export_to_linkwarden(self, bookmarks: List[Bookmark],
                                  update_existing: bool = True,
                                  create_missing_collections: bool = True) -> ExportResult:
        """Export enhanced data back to Linkwarden"""
        
    async def sync_incremental(self, since: datetime) -> SyncResult:
        """Perform incremental sync since specified timestamp"""

class ConflictResolver:
    def __init__(self, strategy: str = "interactive"):
        self.strategy = strategy
        self.resolution_history = []
    
    def detect_conflicts(self, local_data: List[Bookmark],
                        remote_data: List[Bookmark]) -> List[Conflict]:
        """Detect conflicts between local and remote data"""
        
    async def resolve_conflicts(self, conflicts: List[Conflict]) -> List[Resolution]:
        """Resolve conflicts using configured strategy"""
        
    def _resolve_interactive(self, conflict: Conflict) -> Resolution:
        """Interactive conflict resolution with user input"""
        
    def _resolve_automatic(self, conflict: Conflict) -> Resolution:
        """Automatic conflict resolution using rules"""

class MergeStrategy:
    def __init__(self, strategy_name: str = "smart"):
        self.strategy_name = strategy_name
        self.merge_functions = {
            "smart": self._smart_merge,
            "source_priority": self._source_priority_merge,
            "target_priority": self._target_priority_merge,
            "preserve_enhanced": self._preserve_enhanced_merge
        }
    
    def merge_bookmarks(self, source: Bookmark, target: Bookmark,
                       enhanced: Bookmark = None) -> Bookmark:
        """Merge bookmarks using configured strategy"""
        
    def _smart_merge(self, source: Bookmark, target: Bookmark,
                    enhanced: Bookmark = None) -> Bookmark:
        """Smart merge preserving best data from all sources"""
        
    def _preserve_enhanced_merge(self, source: Bookmark, target: Bookmark,
                               enhanced: Bookmark = None) -> Bookmark:
        """Merge preserving AI enhancements while respecting user changes"""
```

### Batch Processing and Progress Tracking

```python
class BatchProcessor:
    def __init__(self, client: LinkwardenClient, batch_size: int = 50):
        self.client = client
        self.batch_size = batch_size
        self.progress_tracker = ProgressTracker()
    
    async def process_in_batches(self, items: List[Any], 
                                operation: Callable,
                                operation_name: str) -> BatchResult:
        """Process items in batches with progress tracking"""
        
    async def import_bookmarks_batch(self, collection_ids: List[int] = None) -> ImportResult:
        """Import bookmarks in batches with progress tracking"""
        
    async def export_bookmarks_batch(self, bookmarks: List[Bookmark]) -> ExportResult:
        """Export bookmarks in batches with progress tracking"""
        
    async def update_bookmarks_batch(self, updates: List[Dict]) -> UpdateResult:
        """Update bookmarks in batches with progress tracking"""

class ProgressTracker:
    def __init__(self):
        self.current_operation = None
        self.total_items = 0
        self.completed_items = 0
        self.start_time = None
        self.errors = []
    
    def start_operation(self, operation_name: str, total_items: int) -> None:
        """Start tracking operation progress"""
        
    def update_progress(self, completed: int, errors: List[Exception] = None) -> None:
        """Update progress with completed items and errors"""
        
    def get_progress_info(self) -> ProgressInfo:
        """Get current progress information"""
        
    def complete_operation(self) -> OperationSummary:
        """Complete operation and return summary"""

class ProgressInfo:
    def __init__(self):
        self.operation_name = ""
        self.total_items = 0
        self.completed_items = 0
        self.percentage = 0.0
        self.eta_seconds = 0
        self.items_per_second = 0.0
        self.errors_count = 0
```

## Data Models

### API Response Models

```python
@dataclass
class ConnectionTestResult:
    success: bool
    response_time_ms: float
    api_version: str
    instance_info: Dict[str, Any]
    error_message: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)

@dataclass
class ImportResult:
    total_bookmarks: int
    imported_bookmarks: int
    total_collections: int
    imported_collections: int
    total_tags: int
    imported_tags: int
    processing_time: float
    errors: List[Exception] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class ExportResult:
    total_bookmarks: int
    exported_bookmarks: int
    updated_bookmarks: int
    created_bookmarks: int
    created_collections: int
    processing_time: float
    errors: List[Exception] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class SyncResult:
    import_result: ImportResult
    export_result: ExportResult
    conflicts_detected: int
    conflicts_resolved: int
    sync_strategy: str
    processing_time: float
    data_integrity_check: bool

@dataclass
class BulkOperationResult:
    total_items: int
    successful_items: int
    failed_items: int
    processing_time: float
    errors: List[Exception] = field(default_factory=list)
    item_results: List[Dict] = field(default_factory=list)
```

### Conflict Resolution Models

```python
@dataclass
class Conflict:
    conflict_type: str  # 'url', 'title', 'tags', 'collection', 'deletion'
    local_bookmark: Bookmark
    remote_bookmark: Bookmark
    enhanced_bookmark: Optional[Bookmark] = None
    confidence_score: float = 0.0
    auto_resolvable: bool = False

@dataclass
class Resolution:
    conflict: Conflict
    resolution_strategy: str
    resolved_bookmark: Bookmark
    reasoning: str
    user_confirmed: bool = False

class ConflictType(Enum):
    URL_MISMATCH = "url_mismatch"
    TITLE_DIFFERENCE = "title_difference"
    TAG_DIFFERENCE = "tag_difference"
    COLLECTION_DIFFERENCE = "collection_difference"
    DELETION_CONFLICT = "deletion_conflict"
    METADATA_DIFFERENCE = "metadata_difference"
```

## CLI Integration

### Command Implementation

```python
class LinkwardenCommands:
    def __init__(self, client: LinkwardenClient):
        self.client = client
        self.sync_engine = SyncEngine(client)
        self.batch_processor = BatchProcessor(client)
    
    async def import_command(self, output_file: str,
                           collections: List[str] = None,
                           enhance: bool = False,
                           since: str = None) -> None:
        """Import bookmarks from Linkwarden"""
        
    async def export_command(self, input_file: str,
                           update_existing: bool = True,
                           create_collections: bool = True) -> None:
        """Export enhanced bookmarks to Linkwarden"""
        
    async def sync_command(self, strategy: str = "smart",
                          conflict_resolution: str = "interactive",
                          backup: bool = True) -> None:
        """Bidirectional sync with Linkwarden"""
        
    async def test_connection_command(self) -> None:
        """Test Linkwarden API connection"""
```

### CLI Command Examples

```bash
# Import from Linkwarden
linkwarden-enhancer import --source linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --output bookmarks.json

# Import with enhancement
linkwarden-enhancer import --source linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --enhance-all \
    --output enhanced_bookmarks.json

# Import specific collections
linkwarden-enhancer import --source linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --collections "Development,Gaming,Research" \
    --output filtered_bookmarks.json

# Export enhanced data back
linkwarden-enhancer export --target linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --input enhanced_bookmarks.json \
    --update-existing \
    --create-missing-collections

# Bidirectional sync
linkwarden-enhancer sync --source linkwarden \
    --target linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --strategy smart-merge \
    --conflict-resolution interactive \
    --backup-before-sync
```

## Error Handling and Resilience

### Comprehensive Error Handling

```python
class APIErrorRecovery:
    def __init__(self, client: LinkwardenClient):
        self.client = client
        self.recovery_strategies = {
            401: self._handle_auth_error,
            403: self._handle_permission_error,
            404: self._handle_not_found_error,
            429: self._handle_rate_limit_error,
            500: self._handle_server_error
        }
    
    async def handle_error(self, error: LinkwardenAPIError, 
                          operation_context: Dict) -> RecoveryResult:
        """Handle API error with appropriate recovery strategy"""
        
    async def _handle_rate_limit_error(self, error: LinkwardenAPIError,
                                     context: Dict) -> RecoveryResult:
        """Handle rate limit errors with exponential backoff"""
        
    async def _handle_server_error(self, error: LinkwardenAPIError,
                                 context: Dict) -> RecoveryResult:
        """Handle server errors with retry logic"""

class DataIntegrityValidator:
    def validate_import_data(self, imported_data: List[Bookmark]) -> ValidationResult:
        """Validate imported data integrity"""
        
    def validate_export_data(self, export_data: List[Dict]) -> ValidationResult:
        """Validate data before export"""
        
    def validate_sync_result(self, sync_result: SyncResult) -> ValidationResult:
        """Validate sync operation results"""
```

## Performance Optimization

### Connection Pooling and Caching

```python
class ConnectionManager:
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.connection_pool = None
        self.session_cache = {}
    
    async def get_session(self, base_url: str) -> aiohttp.ClientSession:
        """Get or create HTTP session with connection pooling"""
        
    async def close_all_sessions(self) -> None:
        """Close all HTTP sessions and cleanup resources"""

class ResponseCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached response if still valid"""
        
    def set(self, key: str, value: Dict) -> None:
        """Cache response with TTL"""
        
    def invalidate(self, pattern: str = None) -> None:
        """Invalidate cache entries matching pattern"""
```

## Testing Strategy

### Unit Tests

```python
class TestLinkwardenClient:
    def test_authentication(self):
        """Test API key validation and authentication"""
        
    def test_get_all_links(self):
        """Test link retrieval with mocked responses"""
        
    def test_create_link(self):
        """Test link creation with validation"""
        
    def test_bulk_operations(self):
        """Test bulk import/export operations"""
        
    def test_error_handling(self):
        """Test various error scenarios and recovery"""

class TestSyncEngine:
    def test_conflict_detection(self):
        """Test conflict detection algorithms"""
        
    def test_merge_strategies(self):
        """Test different merge strategies"""
        
    def test_bidirectional_sync(self):
        """Test complete bidirectional sync flow"""
```

### Integration Tests

```python
class TestLinkwardenIntegration:
    def test_full_import_export_cycle(self):
        """Test complete import -> enhance -> export cycle"""
        
    def test_sync_with_conflicts(self):
        """Test conflict resolution during sync"""
        
    def test_large_dataset_handling(self):
        """Test performance with large bookmark collections"""
        
    def test_error_recovery(self):
        """Test recovery from various failure scenarios"""
```

## Success Criteria

1. **API Integration**: Successfully connect to Linkwarden API with 99%+ uptime
2. **Data Fidelity**: 100% accurate import/export of bookmark data with no data loss
3. **Performance**: Handle 10,000+ bookmarks with <10 minutes processing time
4. **Error Resilience**: Graceful handling of network failures, rate limits, and API errors
5. **Sync Reliability**: Successful bidirectional sync with intelligent conflict resolution
6. **User Experience**: Clear progress indicators and detailed error reporting
7. **Data Integrity**: Comprehensive validation ensuring referential integrity
8. **Scalability**: Efficient batch processing and resource management for large datasets