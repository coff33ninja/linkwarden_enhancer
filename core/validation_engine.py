"""Validation Engine - Comprehensive data validation and schema checking"""

import json
import jsonschema
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict, Counter

from utils.logging_utils import get_logger
from data_models import ValidationResult

logger = get_logger(__name__)


class ValidationEngine:
    """Comprehensive validation engine for Linkwarden data"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize validation engine with configuration"""
        self.config = config
        self.validation_config = config.get('validation', {})
        
        # Load JSON schemas
        self.schemas = self._load_schemas()
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'schema_errors': 0,
            'consistency_errors': 0,
            'integrity_errors': 0
        }
        
        logger.info("Validation engine initialized")
    
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load JSON schemas for Linkwarden data structures"""
        
        schemas = {
            'bookmark': {
                "type": "object",
                "required": ["id", "name", "url"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string", "minLength": 1},
                    "url": {"type": "string", "format": "uri"},
                    "description": {"type": ["string", "null"]},
                    "collection": {
                        "type": ["object", "null"],
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"}
                        }
                    },
                    "tags": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "name"],
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string", "minLength": 1}
                            }
                        }
                    },
                    "content": {
                        "type": ["object", "null"],
                        "properties": {
                            "text_content": {"type": ["string", "null"]},
                            "html_content": {"type": ["string", "null"]},
                            "title": {"type": ["string", "null"]},
                            "description": {"type": ["string", "null"]}
                        }
                    },
                    "created_at": {"type": "string"},
                    "updated_at": {"type": "string"}
                }
            },
            
            'collection': {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": ["string", "null"]},
                    "parent_id": {"type": ["integer", "null"]},
                    "color": {"type": ["string", "null"]},
                    "is_public": {"type": "boolean"},
                    "created_at": {"type": "string"},
                    "updated_at": {"type": "string"}
                }
            },
            
            'tag': {
                "type": "object",
                "required": ["id", "name"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string", "minLength": 1},
                    "created_at": {"type": "string"},
                    "updated_at": {"type": "string"}
                }
            },
            
            'linkwarden_backup': {
                "type": "object",
                "required": ["bookmarks", "collections", "tags"],
                "properties": {
                    "bookmarks": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/bookmark"}
                    },
                    "collections": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/collection"}
                    },
                    "tags": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/tag"}
                    },
                    "version": {"type": "string"},
                    "exported_at": {"type": "string"}
                },
                "definitions": {
                    "bookmark": schemas['bookmark'] if 'bookmark' in locals() else {},
                    "collection": schemas['collection'] if 'collection' in locals() else {},
                    "tag": schemas['tag'] if 'tag' in locals() else {}
                }
            }
        }
        
        # Fix circular reference by updating definitions
        schemas['linkwarden_backup']['definitions'] = {
            'bookmark': schemas['bookmark'],
            'collection': schemas['collection'],
            'tag': schemas['tag']
        }
        
        return schemas
    
    def validate_json_schema(self, data: Dict[str, Any], schema_name: str) -> ValidationResult:
        """Validate data against JSON schema"""
        
        try:
            self.validation_stats['total_validations'] += 1
            
            if schema_name not in self.schemas:
                error_msg = f"Unknown schema: {schema_name}"
                logger.error(error_msg)
                return ValidationResult(
                    valid=False,
                    errors=[error_msg],
                    warnings=[],
                    total_bookmarks=0,
                    total_collections=0,
                    total_tags=0
                )
            
            schema = self.schemas[schema_name]
            
            # Validate against schema
            try:
                jsonschema.validate(data, schema)
                logger.debug(f"Schema validation passed for {schema_name}")
                
                # Count items if it's a full backup
                counts = self._count_items(data, schema_name)
                
                return ValidationResult(
                    valid=True,
                    errors=[],
                    warnings=[],
                    total_bookmarks=counts['bookmarks'],
                    total_collections=counts['collections'],
                    total_tags=counts['tags']
                )
                
            except jsonschema.ValidationError as e:
                self.validation_stats['schema_errors'] += 1
                error_msg = f"Schema validation failed: {e.message} at path {'.'.join(str(p) for p in e.absolute_path)}"
                logger.error(error_msg)
                
                return ValidationResult(
                    valid=False,
                    errors=[error_msg],
                    warnings=[],
                    total_bookmarks=0,
                    total_collections=0,
                    total_tags=0
                )
                
        except Exception as e:
            error_msg = f"Validation engine error: {e}"
            logger.error(error_msg)
            return ValidationResult(
                valid=False,
                errors=[error_msg],
                warnings=[],
                total_bookmarks=0,
                total_collections=0,
                total_tags=0
            )
    
    def validate_data_consistency(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate data consistency and relationships"""
        
        try:
            self.validation_stats['total_validations'] += 1
            errors = []
            warnings = []
            
            bookmarks = data.get('bookmarks', [])
            collections = data.get('collections', [])
            tags = data.get('tags', [])
            
            # Create lookup dictionaries
            collection_ids = {col['id'] for col in collections}
            tag_ids = {tag['id'] for tag in tags}
            bookmark_ids = {bm['id'] for bm in bookmarks}
            
            # Validate bookmark-collection relationships
            for bookmark in bookmarks:
                bookmark_id = bookmark.get('id')
                collection = bookmark.get('collection')
                
                if collection and collection.get('id') not in collection_ids:
                    errors.append(f"Bookmark {bookmark_id} references non-existent collection {collection.get('id')}")
                
                # Validate bookmark-tag relationships
                bookmark_tags = bookmark.get('tags', [])
                for tag in bookmark_tags:
                    tag_id = tag.get('id')
                    if tag_id not in tag_ids:
                        errors.append(f"Bookmark {bookmark_id} references non-existent tag {tag_id}")
            
            # Validate collection hierarchies
            for collection in collections:
                parent_id = collection.get('parent_id')
                if parent_id and parent_id not in collection_ids:
                    errors.append(f"Collection {collection['id']} references non-existent parent {parent_id}")
                
                # Check for circular references
                if self._has_circular_reference(collection, collections):
                    errors.append(f"Collection {collection['id']} has circular parent reference")
            
            # Check for duplicate IDs
            duplicate_bookmarks = self._find_duplicates([bm['id'] for bm in bookmarks])
            duplicate_collections = self._find_duplicates([col['id'] for col in collections])
            duplicate_tags = self._find_duplicates([tag['id'] for tag in tags])
            
            if duplicate_bookmarks:
                errors.append(f"Duplicate bookmark IDs found: {duplicate_bookmarks}")
            if duplicate_collections:
                errors.append(f"Duplicate collection IDs found: {duplicate_collections}")
            if duplicate_tags:
                errors.append(f"Duplicate tag IDs found: {duplicate_tags}")
            
            # Check for orphaned items
            orphaned_collections = self._find_orphaned_collections(collections)
            if orphaned_collections:
                warnings.append(f"Found {len(orphaned_collections)} orphaned collections")
            
            # Update statistics
            if errors:
                self.validation_stats['consistency_errors'] += len(errors)
            
            logger.info(f"Consistency validation completed: {len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                total_bookmarks=len(bookmarks),
                total_collections=len(collections),
                total_tags=len(tags)
            )
            
        except Exception as e:
            error_msg = f"Consistency validation error: {e}"
            logger.error(error_msg)
            return ValidationResult(
                valid=False,
                errors=[error_msg],
                warnings=[],
                total_bookmarks=0,
                total_collections=0,
                total_tags=0
            )
    
    def validate_field_requirements(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate required fields and data quality"""
        
        try:
            errors = []
            warnings = []
            
            bookmarks = data.get('bookmarks', [])
            collections = data.get('collections', [])
            tags = data.get('tags', [])
            
            # Validate bookmark fields
            for bookmark in bookmarks:
                bookmark_id = bookmark.get('id')
                
                # Required fields
                if not bookmark.get('name', '').strip():
                    errors.append(f"Bookmark {bookmark_id} has empty name")
                
                if not bookmark.get('url', '').strip():
                    errors.append(f"Bookmark {bookmark_id} has empty URL")
                
                # URL format validation
                url = bookmark.get('url', '')
                if url and not self._is_valid_url(url):
                    warnings.append(f"Bookmark {bookmark_id} has potentially invalid URL: {url}")
                
                # Check for extremely long names/descriptions
                name = bookmark.get('name', '')
                if len(name) > 500:
                    warnings.append(f"Bookmark {bookmark_id} has very long name ({len(name)} chars)")
                
                description = bookmark.get('description', '') or ''
                if len(description) > 5000:
                    warnings.append(f"Bookmark {bookmark_id} has very long description ({len(description)} chars)")
            
            # Validate collection fields
            for collection in collections:
                collection_id = collection.get('id')
                
                if not collection.get('name', '').strip():
                    errors.append(f"Collection {collection_id} has empty name")
                
                # Check for very long collection names
                name = collection.get('name', '')
                if len(name) > 200:
                    warnings.append(f"Collection {collection_id} has very long name ({len(name)} chars)")
            
            # Validate tag fields
            for tag in tags:
                tag_id = tag.get('id')
                
                if not tag.get('name', '').strip():
                    errors.append(f"Tag {tag_id} has empty name")
                
                # Check for very long tag names
                name = tag.get('name', '')
                if len(name) > 100:
                    warnings.append(f"Tag {tag_id} has very long name ({len(name)} chars)")
            
            logger.info(f"Field validation completed: {len(errors)} errors, {len(warnings)} warnings")
            
            return ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                total_bookmarks=len(bookmarks),
                total_collections=len(collections),
                total_tags=len(tags)
            )
            
        except Exception as e:
            error_msg = f"Field validation error: {e}"
            logger.error(error_msg)
            return ValidationResult(
                valid=False,
                errors=[error_msg],
                warnings=[],
                total_bookmarks=0,
                total_collections=0,
                total_tags=0
            )
    
    def create_data_inventory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive data inventory"""
        
        try:
            bookmarks = data.get('bookmarks', [])
            collections = data.get('collections', [])
            tags = data.get('tags', [])
            
            # Basic counts
            inventory = {
                'total_bookmarks': len(bookmarks),
                'total_collections': len(collections),
                'total_tags': len(tags),
                'timestamp': self._get_current_timestamp()
            }
            
            # Bookmark analysis
            bookmark_analysis = {
                'with_descriptions': sum(1 for bm in bookmarks if bm.get('description')),
                'with_content': sum(1 for bm in bookmarks if bm.get('content')),
                'with_collections': sum(1 for bm in bookmarks if bm.get('collection')),
                'total_tag_associations': sum(len(bm.get('tags', [])) for bm in bookmarks)
            }
            
            # Collection analysis
            collection_analysis = {
                'root_collections': sum(1 for col in collections if not col.get('parent_id')),
                'nested_collections': sum(1 for col in collections if col.get('parent_id')),
                'max_nesting_depth': self._calculate_max_nesting_depth(collections),
                'public_collections': sum(1 for col in collections if col.get('is_public', False))
            }
            
            # Tag analysis
            tag_usage = Counter()
            for bookmark in bookmarks:
                for tag in bookmark.get('tags', []):
                    tag_usage[tag.get('name', '')] += 1
            
            tag_analysis = {
                'most_used_tags': dict(tag_usage.most_common(10)),
                'unused_tags': len([tag for tag in tags if tag.get('name') not in tag_usage]),
                'average_tags_per_bookmark': len(bookmarks) and sum(len(bm.get('tags', [])) for bm in bookmarks) / len(bookmarks) or 0
            }
            
            # URL domain analysis
            domain_counts = Counter()
            for bookmark in bookmarks:
                url = bookmark.get('url', '')
                if url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        domain_counts[domain] += 1
                    except:
                        pass
            
            url_analysis = {
                'unique_domains': len(domain_counts),
                'top_domains': dict(domain_counts.most_common(10)),
                'single_bookmark_domains': sum(1 for count in domain_counts.values() if count == 1)
            }
            
            inventory.update({
                'bookmark_analysis': bookmark_analysis,
                'collection_analysis': collection_analysis,
                'tag_analysis': tag_analysis,
                'url_analysis': url_analysis
            })
            
            logger.info(f"Data inventory created: {inventory['total_bookmarks']} bookmarks, {inventory['total_collections']} collections, {inventory['total_tags']} tags")
            
            return inventory
            
        except Exception as e:
            logger.error(f"Failed to create data inventory: {e}")
            return {'error': str(e)}
    
    def _count_items(self, data: Dict[str, Any], schema_name: str) -> Dict[str, int]:
        """Count items in data based on schema type"""
        
        counts = {'bookmarks': 0, 'collections': 0, 'tags': 0}
        
        if schema_name == 'linkwarden_backup':
            counts['bookmarks'] = len(data.get('bookmarks', []))
            counts['collections'] = len(data.get('collections', []))
            counts['tags'] = len(data.get('tags', []))
        elif schema_name == 'bookmark':
            counts['bookmarks'] = 1
        elif schema_name == 'collection':
            counts['collections'] = 1
        elif schema_name == 'tag':
            counts['tags'] = 1
        
        return counts
    
    def _has_circular_reference(self, collection: Dict[str, Any], all_collections: List[Dict[str, Any]]) -> bool:
        """Check if collection has circular parent reference"""
        
        visited = set()
        current_id = collection['id']
        
        # Create parent lookup
        parent_lookup = {col['id']: col.get('parent_id') for col in all_collections}
        
        while current_id:
            if current_id in visited:
                return True
            
            visited.add(current_id)
            current_id = parent_lookup.get(current_id)
        
        return False
    
    def _find_duplicates(self, items: List[Any]) -> List[Any]:
        """Find duplicate items in list"""
        
        counts = Counter(items)
        return [item for item, count in counts.items() if count > 1]
    
    def _find_orphaned_collections(self, collections: List[Dict[str, Any]]) -> List[int]:
        """Find collections that reference non-existent parents"""
        
        collection_ids = {col['id'] for col in collections}
        orphaned = []
        
        for collection in collections:
            parent_id = collection.get('parent_id')
            if parent_id and parent_id not in collection_ids:
                orphaned.append(collection['id'])
        
        return orphaned
    
    def _is_valid_url(self, url: str) -> bool:
        """Basic URL validation"""
        
        try:
            from urllib.parse import urlparse
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _calculate_max_nesting_depth(self, collections: List[Dict[str, Any]]) -> int:
        """Calculate maximum nesting depth of collections"""
        
        # Create parent-child mapping
        children = defaultdict(list)
        for col in collections:
            parent_id = col.get('parent_id')
            if parent_id:
                children[parent_id].append(col['id'])
        
        def get_depth(collection_id: int) -> int:
            if collection_id not in children:
                return 1
            return 1 + max(get_depth(child_id) for child_id in children[collection_id])
        
        # Find root collections and calculate max depth
        root_collections = [col['id'] for col in collections if not col.get('parent_id')]
        
        if not root_collections:
            return 0
        
        return max(get_depth(root_id) for root_id in root_collections)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation engine statistics"""
        
        return {
            'validation_stats': self.validation_stats.copy(),
            'schemas_loaded': list(self.schemas.keys()),
            'config': self.validation_config
        }