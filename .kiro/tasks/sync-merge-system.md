# Sync & Merge System

## Task Overview
Implement comprehensive bidirectional sync and intelligent merge system for bookmark data across multiple platforms with conflict resolution and data integrity guarantees.

## Core Sync Capabilities

### 1. Bidirectional Sync Engine
**Purpose**: Seamlessly sync bookmarks between any two supported platforms

#### Implementation Tasks
- [ ] **Sync Engine Core** (`sync/sync_engine.py`)
  ```python
  class SyncEngine:
      def __init__(self, source_client: BaseAPIClient, target_client: BaseAPIClient):
          self.source = source_client
          self.target = target_client
          self.conflict_resolver = ConflictResolver()
          self.merge_strategy = MergeStrategy()
      
      async def sync_bidirectional(self) -> SyncResult:
          """Perform full bidirectional sync"""
      
      async def sync_incremental(self, since: datetime) -> SyncResult:
          """Perform incremental sync since timestamp"""
      
      async def sync_dry_run(self) -> SyncPreview:
          """Preview sync changes without applying"""
  ```

- [ ] **Sync State Management**
  - Track last sync timestamps per platform
  - Maintain sync state database/file
  - Handle interrupted sync recovery
  - Store sync metadata and checksums

- [ ] **Change Detection**
  - Identify new, modified, and deleted bookmarks
  - Compare timestamps and content hashes
  - Detect structural changes (collections, tags)
  - Track user modifications vs system changes

### 2. Intelligent Merge System
**Purpose**: Combine bookmark data from multiple sources intelligently

#### Implementation Tasks
- [ ] **Merge Strategy Engine** (`sync/merge_strategies.py`)
  ```python
  class MergeStrategy:
      def merge_bookmarks(self, source: Bookmark, target: Bookmark) -> Bookmark:
          """Merge two bookmark objects"""
      
      def merge_collections(self, source_collections: List[Collection], 
                          target_collections: List[Collection]) -> List[Collection]:
          """Merge collection structures"""
      
      def merge_tags(self, source_tags: List[str], target_tags: List[str]) -> List[str]:
          """Intelligently merge tag lists"""
  ```

- [ ] **Merge Strategies**
  - **Smart Merge**: Combine best data from both sources
  - **Source Priority**: Prefer source data over target
  - **Target Priority**: Prefer target data over source
  - **User Choice**: Interactive selection for conflicts
  - **Timestamp-Based**: Use most recently modified data

- [ ] **Data Combination Rules**
  - **Titles**: Use longest/most descriptive title
  - **Descriptions**: Combine or choose best description
  - **Tags**: Union of all tags with deduplication
  - **Collections**: Merge hierarchies intelligently
  - **Metadata**: Preserve all available metadata

### 3. Conflict Resolution System
**Purpose**: Handle conflicts when same bookmark exists in multiple sources

#### Implementation Tasks
- [ ] **Conflict Detection** (`sync/conflict_detector.py`)
  ```python
  class ConflictDetector:
      def detect_conflicts(self, source_data: List[Bookmark], 
                          target_data: List[Bookmark]) -> List[Conflict]:
          """Identify all conflicts between datasets"""
      
      def classify_conflict(self, conflict: Conflict) -> ConflictType:
          """Classify conflict type and severity"""
      
      def suggest_resolution(self, conflict: Conflict) -> Resolution:
          """Suggest automatic resolution strategy"""
  ```

- [ ] **Conflict Types**
  - **URL Conflicts**: Same URL, different metadata
  - **Title Conflicts**: Different titles for same URL
  - **Tag Conflicts**: Different tag sets
  - **Collection Conflicts**: Different folder assignments
  - **Deletion Conflicts**: Deleted in one source, modified in another

- [ ] **Resolution Strategies**
  - **Automatic Resolution**: For low-risk conflicts
  - **Interactive Resolution**: User chooses resolution
  - **Rule-Based Resolution**: Apply predefined rules
  - **AI-Assisted Resolution**: Use AI to suggest best option

### 4. Enhanced CLI Commands

#### Sync Commands
```bash
# Full bidirectional sync
linkwarden-enhancer sync \
    --source linkwarden --source-url $LINKWARDEN_URL --source-key $LINKWARDEN_API_KEY \
    --target raindrop --target-token $RAINDROP_TOKEN \
    --strategy smart-merge \
    --conflict-resolution interactive

# Incremental sync (only changes since last sync)
linkwarden-enhancer sync \
    --source linkwarden --source-url $LINKWARDEN_URL --source-key $LINKWARDEN_API_KEY \
    --target pocket --target-consumer-key $POCKET_KEY --target-access-token $POCKET_TOKEN \
    --incremental \
    --since "2024-01-01"

# Dry run sync (preview changes)
linkwarden-enhancer sync \
    --source linkwarden --source-url $LINKWARDEN_URL --source-key $LINKWARDEN_API_KEY \
    --target pinboard --target-token $PINBOARD_TOKEN \
    --dry-run \
    --generate-report
```

#### Merge Commands
```bash
# Merge multiple bookmark files
linkwarden-enhancer merge \
    --input bookmarks1.json bookmarks2.json bookmarks3.json \
    --output merged_bookmarks.json \
    --strategy smart-merge \
    --remove-duplicates

# Merge with enhanced data
linkwarden-enhancer merge \
    --input original_bookmarks.json enhanced_bookmarks.json \
    --output final_bookmarks.json \
    --strategy prefer-enhanced \
    --preserve-user-modifications
```

## Sync Architecture

### Data Flow
```
Source Platform → Change Detection → Conflict Detection → Resolution → Target Platform
       ↓               ↓                    ↓               ↓              ↓
   API Client      Diff Engine       Conflict Engine   Merge Engine   API Client
```

### Sync State Management
```python
class SyncState:
    def __init__(self, source: str, target: str):
        self.source_platform = source
        self.target_platform = target
        self.last_sync_timestamp = None
        self.sync_checksum = None
        self.pending_conflicts = []
        self.sync_statistics = SyncStats()
    
    def save_state(self) -> None:
        """Persist sync state to disk"""
    
    def load_state(self) -> None:
        """Load sync state from disk"""
    
    def reset_state(self) -> None:
        """Reset sync state for fresh sync"""
```

### Change Detection Algorithm
```python
class ChangeDetector:
    def detect_changes(self, source_data: List[Bookmark], 
                      target_data: List[Bookmark], 
                      last_sync: datetime) -> ChangeSet:
        """Detect all changes since last sync"""
        
        changes = ChangeSet()
        
        # Create lookup maps
        source_map = {b.url: b for b in source_data}
        target_map = {b.url: b for b in target_data}
        
        # Detect additions
        for url, bookmark in source_map.items():
            if url not in target_map:
                changes.additions.append(bookmark)
        
        # Detect deletions
        for url, bookmark in target_map.items():
            if url not in source_map:
                changes.deletions.append(bookmark)
        
        # Detect modifications
        for url in source_map.keys() & target_map.keys():
            source_bookmark = source_map[url]
            target_bookmark = target_map[url]
            
            if self.has_changed(source_bookmark, target_bookmark, last_sync):
                changes.modifications.append((source_bookmark, target_bookmark))
        
        return changes
```

## Conflict Resolution

### Interactive Conflict Resolution
```python
class InteractiveResolver:
    def resolve_conflict(self, conflict: Conflict) -> Resolution:
        """Present conflict to user for resolution"""
        
        print(f"\nConflict detected for: {conflict.url}")
        print(f"Source: {conflict.source_data}")
        print(f"Target: {conflict.target_data}")
        
        options = [
            "1. Use source data",
            "2. Use target data", 
            "3. Merge both",
            "4. Skip this bookmark",
            "5. Edit manually"
        ]
        
        choice = self.get_user_choice(options)
        return self.apply_choice(choice, conflict)
```

### Automatic Conflict Resolution
```python
class AutomaticResolver:
    def __init__(self, rules: List[ResolutionRule]):
        self.rules = rules
    
    def resolve_conflict(self, conflict: Conflict) -> Optional[Resolution]:
        """Attempt automatic resolution using rules"""
        
        for rule in self.rules:
            if rule.applies_to(conflict):
                resolution = rule.resolve(conflict)
                if resolution.confidence > 0.8:  # High confidence threshold
                    return resolution
        
        return None  # Escalate to interactive resolution
```

## Data Integrity and Safety

### Backup Before Sync
```python
class SyncSafetyManager:
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
    
    async def safe_sync(self, sync_operation: Callable) -> SyncResult:
        """Perform sync with safety guarantees"""
        
        # Create backup before sync
        backup_id = await self.backup_manager.create_backup(
            description="Pre-sync backup"
        )
        
        try:
            result = await sync_operation()
            
            # Validate sync result
            if not self.validate_sync_result(result):
                await self.backup_manager.restore_backup(backup_id)
                raise SyncValidationError("Sync validation failed, restored backup")
            
            return result
            
        except Exception as e:
            # Restore backup on failure
            await self.backup_manager.restore_backup(backup_id)
            raise SyncError(f"Sync failed, backup restored: {e}")
```

### Sync Validation
```python
class SyncValidator:
    def validate_sync_result(self, result: SyncResult) -> bool:
        """Validate sync completed successfully"""
        
        # Check for data loss
        if result.items_deleted > result.max_allowed_deletions:
            return False
        
        # Check for corruption
        if not self.validate_data_integrity(result.final_data):
            return False
        
        # Check sync statistics
        if result.error_rate > 0.05:  # Max 5% error rate
            return False
        
        return True
```

## Performance Optimization

### Incremental Sync
```python
class IncrementalSyncer:
    def __init__(self, sync_state: SyncState):
        self.sync_state = sync_state
    
    async def sync_incremental(self) -> SyncResult:
        """Sync only changes since last sync"""
        
        last_sync = self.sync_state.last_sync_timestamp
        
        # Get changes from source since last sync
        source_changes = await self.source.get_changes_since(last_sync)
        
        # Get changes from target since last sync  
        target_changes = await self.target.get_changes_since(last_sync)
        
        # Merge and apply changes
        return await self.apply_incremental_changes(source_changes, target_changes)
```

### Batch Processing
```python
class BatchSyncer:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
    
    async def sync_in_batches(self, changes: List[Change]) -> SyncResult:
        """Process sync changes in batches"""
        
        results = []
        
        for i in range(0, len(changes), self.batch_size):
            batch = changes[i:i + self.batch_size]
            
            try:
                batch_result = await self.process_batch(batch)
                results.append(batch_result)
                
                # Rate limiting between batches
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Batch {i//self.batch_size + 1} failed: {e}")
                # Continue with next batch
        
        return self.combine_results(results)
```

## Environment Configuration

### Sync Settings
```bash
# Sync Behavior
DEFAULT_SYNC_STRATEGY=smart-merge
ENABLE_INCREMENTAL_SYNC=true
SYNC_BATCH_SIZE=100
MAX_SYNC_CONFLICTS=50

# Conflict Resolution
DEFAULT_CONFLICT_RESOLUTION=interactive
AUTO_RESOLVE_LOW_RISK_CONFLICTS=true
CONFLICT_CONFIDENCE_THRESHOLD=0.8

# Safety Settings
BACKUP_BEFORE_SYNC=true
MAX_DELETION_PERCENTAGE=10.0
VALIDATE_SYNC_RESULTS=true
ENABLE_SYNC_ROLLBACK=true

# Performance
SYNC_TIMEOUT_MINUTES=30
ENABLE_PARALLEL_PROCESSING=true
MAX_CONCURRENT_REQUESTS=5
```

## Monitoring and Reporting

### Sync Metrics
```python
class SyncMetrics:
    def __init__(self):
        self.items_synced = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        self.sync_duration = 0
        self.error_count = 0
        self.data_transferred = 0
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive sync report"""
        return {
            "summary": {
                "items_synced": self.items_synced,
                "conflicts_detected": self.conflicts_detected,
                "success_rate": (self.items_synced - self.error_count) / self.items_synced,
                "duration": self.sync_duration
            },
            "details": {
                "conflicts_resolved": self.conflicts_resolved,
                "errors": self.error_count,
                "data_transferred_mb": self.data_transferred / 1024 / 1024
            }
        }
```

### Sync History
```python
class SyncHistory:
    def record_sync(self, sync_result: SyncResult) -> None:
        """Record sync operation in history"""
        
    def get_sync_history(self, limit: int = 10) -> List[SyncRecord]:
        """Get recent sync history"""
        
    def analyze_sync_patterns(self) -> SyncAnalysis:
        """Analyze sync patterns and performance"""
```

## Testing Strategy

### Unit Tests
```python
class TestSyncEngine:
    def test_change_detection(self):
        # Test change detection accuracy
        
    def test_conflict_resolution(self):
        # Test various conflict scenarios
        
    def test_merge_strategies(self):
        # Test different merge approaches
```

### Integration Tests
```python
class TestFullSync:
    def test_bidirectional_sync(self):
        # Test complete sync cycle
        
    def test_sync_with_failures(self):
        # Test error recovery and rollback
        
    def test_large_dataset_sync(self):
        # Test performance with large datasets
```

## Success Criteria
1. **Sync Reliability**: 99%+ successful sync operations
2. **Conflict Resolution**: 95%+ of conflicts resolved automatically
3. **Data Integrity**: Zero data loss during sync operations
4. **Performance**: Sync 10,000 bookmarks in <10 minutes
5. **User Experience**: Clear progress indication and conflict resolution
6. **Error Recovery**: Automatic rollback on sync failures
7. **Incremental Efficiency**: 10x faster incremental vs full sync

## Future Enhancements
- Real-time sync with webhooks
- Multi-way sync (3+ platforms simultaneously)
- Advanced conflict resolution using ML
- Sync scheduling and automation
- Collaborative sync for team bookmarks