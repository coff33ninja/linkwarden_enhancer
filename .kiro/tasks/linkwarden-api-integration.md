# Linkwarden API Integration

## Task Overview
Implement comprehensive Linkwarden API integration for direct import/export and bidirectional sync capabilities.

## Linkwarden API Endpoints Analysis

### Authentication
- **Method**: Bearer Token Authentication
- **Header**: `Authorization: Bearer <api_key>`
- **Environment**: `LINKWARDEN_API_KEY`

### Core Endpoints
```
GET    /api/v1/links                    # Get all links
POST   /api/v1/links                    # Create new link
GET    /api/v1/links/{id}               # Get specific link
PUT    /api/v1/links/{id}               # Update link
DELETE /api/v1/links/{id}               # Delete link

GET    /api/v1/collections              # Get all collections
POST   /api/v1/collections              # Create collection
PUT    /api/v1/collections/{id}         # Update collection
DELETE /api/v1/collections/{id}         # Delete collection

GET    /api/v1/tags                     # Get all tags
POST   /api/v1/tags                     # Create tag
PUT    /api/v1/tags/{id}                # Update tag
DELETE /api/v1/tags/{id}                # Delete tag
```

## Implementation Tasks

### 1. Linkwarden API Client (`api/linkwarden_client.py`)
```python
class LinkwardenClient(BaseAPIClient):
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    # Link operations
    async def get_all_links(self, limit: int = 1000, offset: int = 0) -> List[Dict]
    async def create_link(self, link_data: Dict) -> Dict
    async def update_link(self, link_id: int, link_data: Dict) -> Dict
    async def delete_link(self, link_id: int) -> bool
    
    # Collection operations
    async def get_all_collections(self) -> List[Dict]
    async def create_collection(self, collection_data: Dict) -> Dict
    async def update_collection(self, collection_id: int, collection_data: Dict) -> Dict
    
    # Tag operations
    async def get_all_tags(self) -> List[Dict]
    async def create_tag(self, tag_data: Dict) -> Dict
    
    # Bulk operations
    async def bulk_import_links(self, links: List[Dict]) -> BulkResult
    async def bulk_export_links(self) -> List[Dict]
```

### 2. Data Mapping (`api/linkwarden_mapper.py`)
```python
class LinkwardenMapper:
    @staticmethod
    def to_internal_format(linkwarden_data: Dict) -> Bookmark:
        """Convert Linkwarden API response to internal Bookmark format"""
        
    @staticmethod
    def to_linkwarden_format(bookmark: Bookmark) -> Dict:
        """Convert internal Bookmark to Linkwarden API format"""
        
    @staticmethod
    def map_collection(collection_data: Dict) -> Collection:
        """Map Linkwarden collection to internal format"""
```

### 3. Enhanced Import Command
```bash
# Basic import
linkwarden-enhancer import --source linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --output enhanced_bookmarks.json

# Import with enhancement
linkwarden-enhancer import --source linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --enhance-titles \
    --auto-tag \
    --extract-descriptions \
    --remove-duplicates \
    --output enhanced_bookmarks.json

# Import specific collections
linkwarden-enhancer import --source linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --collections "Development,Gaming,Research" \
    --output filtered_bookmarks.json
```

### 4. Enhanced Export Command
```bash
# Export back to Linkwarden with enhancements
linkwarden-enhancer export --target linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --input enhanced_bookmarks.json \
    --update-existing \
    --create-missing-collections

# Sync enhanced data back
linkwarden-enhancer sync --source enhanced_bookmarks.json \
    --target linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --merge-strategy smart \
    --preserve-user-tags
```

### 5. Bidirectional Sync
```bash
# Full bidirectional sync with enhancement
linkwarden-enhancer sync --source linkwarden \
    --target linkwarden \
    --url $LINKWARDEN_URL \
    --api-key $LINKWARDEN_API_KEY \
    --enhance-during-sync \
    --conflict-resolution interactive \
    --backup-before-sync
```

## Data Enhancement Pipeline

### 1. Title Enhancement
- Scrape actual page titles if missing or generic
- Clean up titles (remove site names, extra characters)
- Standardize title formatting

### 2. Auto-Tagging
- Analyze URL patterns for automatic tags
- Content-based tagging using AI
- Domain-specific tag suggestions
- Merge with existing user tags

### 3. Description Generation
- Extract meta descriptions from pages
- Generate AI-powered summaries
- Preserve existing user descriptions
- Fallback to content snippets

### 4. Duplicate Detection
- URL normalization and comparison
- Title similarity matching
- Content fingerprinting
- User confirmation for ambiguous cases

## Environment Configuration

### Required Environment Variables
```bash
# Linkwarden Connection
LINKWARDEN_URL=http://localhost:3000
LINKWARDEN_API_KEY=your_api_key_here
LINKWARDEN_USERNAME=your_username

# Enhancement Settings
ENABLE_TITLE_ENHANCEMENT=true
ENABLE_AUTO_TAGGING=true
ENABLE_DESCRIPTION_GENERATION=true
ENABLE_DUPLICATE_DETECTION=true

# Sync Settings
DEFAULT_MERGE_STRATEGY=smart
BACKUP_BEFORE_SYNC=true
CONFLICT_RESOLUTION=interactive
```

### Configuration Validation
```python
def validate_linkwarden_config():
    required_vars = ['LINKWARDEN_URL', 'LINKWARDEN_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ConfigurationError(f"Missing required environment variables: {missing_vars}")
    
    # Test API connection
    client = LinkwardenClient(os.getenv('LINKWARDEN_URL'), os.getenv('LINKWARDEN_API_KEY'))
    if not client.test_connection():
        raise ConnectionError("Failed to connect to Linkwarden API")
```

## Error Handling & Resilience

### API Error Handling
```python
class LinkwardenAPIError(Exception):
    def __init__(self, status_code: int, message: str, response: Dict = None):
        self.status_code = status_code
        self.message = message
        self.response = response
        super().__init__(f"Linkwarden API Error {status_code}: {message}")

async def handle_api_response(response: requests.Response) -> Dict:
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 401:
        raise LinkwardenAPIError(401, "Invalid API key or unauthorized access")
    elif response.status_code == 429:
        # Rate limiting - implement exponential backoff
        await asyncio.sleep(2 ** attempt)
        return await retry_request(response.request)
    else:
        raise LinkwardenAPIError(response.status_code, response.text)
```

### Batch Processing
```python
async def process_in_batches(items: List[Any], batch_size: int = 50):
    """Process items in batches to avoid overwhelming the API"""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        try:
            await process_batch(batch)
            await asyncio.sleep(0.5)  # Rate limiting
        except Exception as e:
            logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
            # Continue with next batch
```

## Testing Strategy

### Unit Tests
```python
class TestLinkwardenClient:
    def test_authentication(self):
        # Test API key validation
        
    def test_get_all_links(self):
        # Test link retrieval with mocked responses
        
    def test_create_link(self):
        # Test link creation
        
    def test_bulk_operations(self):
        # Test bulk import/export
```

### Integration Tests
```python
class TestLinkwardenIntegration:
    def test_full_import_export_cycle(self):
        # Test complete import -> enhance -> export cycle
        
    def test_sync_with_conflicts(self):
        # Test conflict resolution during sync
        
    def test_error_recovery(self):
        # Test recovery from API failures
```

## Success Criteria
1. **API Integration**: Successfully connect to Linkwarden API
2. **Data Fidelity**: 100% accurate import/export of bookmark data
3. **Enhancement Quality**: Improved titles, tags, and descriptions for 90%+ of bookmarks
4. **Duplicate Detection**: 95%+ accuracy in identifying duplicates
5. **Sync Reliability**: Successful bidirectional sync with conflict resolution
6. **Performance**: Handle 10,000+ bookmarks within reasonable time limits
7. **Error Resilience**: Graceful handling of API failures and network issues

## Documentation
- API integration setup guide
- Environment configuration instructions
- CLI command examples
- Troubleshooting common issues
- Data mapping specifications