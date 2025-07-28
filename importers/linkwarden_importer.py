"""Linkwarden backup JSON importer"""

from typing import List, Dict, Any
from importers.base_importer import BaseImporter
from data_models import ImportResult
from utils.json_handler import JsonHandler
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class LinkwardenImporter(BaseImporter):
    """Import bookmarks from Linkwarden backup JSON"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Linkwarden importer"""
        super().__init__(config)
        self.json_handler = JsonHandler()
        
        logger.info("Linkwarden importer initialized")
    
    def import_data(self, backup_file_path: str) -> ImportResult:
        """Import bookmarks from Linkwarden backup JSON"""
        
        try:
            logger.info(f"Loading Linkwarden backup from: {backup_file_path}")
            
            # Load JSON data
            data = self.json_handler.load_json(backup_file_path)
            
            # Validate structure
            if not self._validate_linkwarden_structure(data):
                return ImportResult(
                    bookmarks=[],
                    total_imported=0,
                    import_source="linkwarden_backup",
                    errors=self.errors,
                    warnings=self.warnings
                )
            
            # Extract bookmarks
            bookmarks = []
            collections = data.get('collections', [])
            
            for collection in collections:
                collection_bookmarks = self._extract_collection_bookmarks(collection)
                bookmarks.extend(collection_bookmarks)
            
            logger.info(f"Successfully imported {len(bookmarks)} bookmarks from {len(collections)} collections")
            
            return ImportResult(
                bookmarks=bookmarks,
                total_imported=len(bookmarks),
                collections_found=len(collections),
                import_source="linkwarden_backup",
                errors=self.errors,
                warnings=self.warnings
            )
            
        except Exception as e:
            self.add_error(f"Failed to import Linkwarden backup: {e}")
            return ImportResult(
                bookmarks=[],
                total_imported=0,
                import_source="linkwarden_backup",
                errors=self.errors,
                warnings=self.warnings
            )
    
    def _validate_linkwarden_structure(self, data: Dict[str, Any]) -> bool:
        """Validate Linkwarden backup structure"""
        
        required_fields = ['collections']
        
        for field in required_fields:
            if field not in data:
                self.add_error(f"Missing required field: {field}")
                return False
        
        collections = data.get('collections', [])
        if not isinstance(collections, list):
            self.add_error("Collections field must be a list")
            return False
        
        logger.info(f"Linkwarden structure validation passed. Found {len(collections)} collections")
        return True
    
    def _extract_collection_bookmarks(self, collection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract bookmarks from a collection"""
        
        bookmarks = []
        collection_name = collection.get('name', 'Unknown Collection')
        collection_id = collection.get('id')
        links = collection.get('links', [])
        
        logger.debug(f"Processing collection '{collection_name}' with {len(links)} links")
        
        for link in links:
            try:
                bookmark = self._convert_linkwarden_bookmark(link, collection)
                bookmarks.append(bookmark)
            except Exception as e:
                self.add_warning(f"Failed to process link {link.get('id', 'unknown')} in collection '{collection_name}': {e}")
                continue
        
        return bookmarks
    
    def _convert_linkwarden_bookmark(self, link: Dict[str, Any], collection: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Linkwarden link to standard bookmark format"""
        
        # Extract basic information
        bookmark = {
            'id': link.get('id'),
            'name': link.get('name', ''),
            'url': link.get('url', ''),
            'description': link.get('description', ''),
            'type': link.get('type', 'url'),
            'created_at': link.get('createdAt', ''),
            'updated_at': link.get('updatedAt', ''),
            'collection_id': link.get('collectionId'),
            'collection_name': collection.get('name', ''),
            'created_by_id': link.get('createdById'),
        }
        
        # Extract tags
        tags = []
        for tag in link.get('tags', []):
            tag_info = {
                'id': tag.get('id'),
                'name': tag.get('name', ''),
                'owner_id': tag.get('ownerId'),
                'ai_tag': tag.get('aiTag', False),
                'created_at': tag.get('createdAt', ''),
                'updated_at': tag.get('updatedAt', '')
            }
            tags.append(tag_info)
        
        bookmark['tags'] = tags
        
        # Extract content and archives
        bookmark['content'] = {
            'text_content': link.get('textContent'),
            'preview': link.get('preview'),
            'image': link.get('image'),
            'pdf': link.get('pdf'),
            'readable': link.get('readable'),
            'monolith': link.get('monolith')
        }
        
        # Extract metadata
        bookmark['metadata'] = {
            'icon': link.get('icon'),
            'icon_weight': link.get('iconWeight'),
            'color': link.get('color'),
            'ai_tagged': link.get('aiTagged', False),
            'index_version': link.get('indexVersion'),
            'last_preserved': link.get('lastPreserved'),
            'import_date': link.get('importDate')
        }
        
        # Extract collection information
        bookmark['collection_info'] = {
            'id': collection.get('id'),
            'name': collection.get('name', ''),
            'description': collection.get('description', ''),
            'icon': collection.get('icon'),
            'color': collection.get('color'),
            'parent_id': collection.get('parentId'),
            'is_public': collection.get('isPublic', False),
            'owner_id': collection.get('ownerId'),
            'created_by_id': collection.get('createdById')
        }
        
        return bookmark
    
    def get_import_stats(self) -> Dict[str, Any]:
        """Get detailed import statistics"""
        base_stats = super().get_import_stats()
        
        # Add Linkwarden-specific stats
        base_stats.update({
            'importer_type': 'linkwarden_backup',
            'supports_collections': True,
            'supports_tags': True,
            'supports_content': True,
            'supports_metadata': True
        })
        
        return base_stats
    
    def preview_import(self, backup_file_path: str, max_items: int = 10) -> Dict[str, Any]:
        """Preview what would be imported without actually importing"""
        
        try:
            logger.info(f"Previewing Linkwarden backup: {backup_file_path}")
            
            # Load JSON data
            data = self.json_handler.load_json(backup_file_path)
            
            # Get basic stats
            stats = self.json_handler.get_json_stats(data)
            
            # Get sample bookmarks
            sample_bookmarks = []
            collections = data.get('collections', [])
            
            count = 0
            for collection in collections:
                if count >= max_items:
                    break
                
                for link in collection.get('links', []):
                    if count >= max_items:
                        break
                    
                    sample_bookmark = {
                        'name': link.get('name', ''),
                        'url': link.get('url', ''),
                        'collection': collection.get('name', ''),
                        'tags': [tag.get('name', '') for tag in link.get('tags', [])],
                        'created_at': link.get('createdAt', '')
                    }
                    sample_bookmarks.append(sample_bookmark)
                    count += 1
            
            preview = {
                'total_collections': stats.get('collections', 0),
                'total_bookmarks': stats.get('bookmarks', 0),
                'total_tags': stats.get('tags', 0),
                'sample_bookmarks': sample_bookmarks,
                'file_size_mb': round(len(str(data)) / (1024 * 1024), 2)
            }
            
            logger.info(f"Preview complete: {preview['total_bookmarks']} bookmarks in {preview['total_collections']} collections")
            
            return preview
            
        except Exception as e:
            logger.error(f"Failed to preview Linkwarden backup: {e}")
            return {
                'error': str(e),
                'total_collections': 0,
                'total_bookmarks': 0,
                'total_tags': 0,
                'sample_bookmarks': []
            }