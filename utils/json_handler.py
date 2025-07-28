"""JSON handling utilities"""

import json
from pathlib import Path
from typing import Dict, Any, List

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class JsonHandler:
    """Handle JSON file operations with proper encoding and error handling"""
    
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load JSON file with proper encoding handling"""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    data = json.load(f)
                logger.info(f"Successfully loaded JSON from {file_path} with {encoding} encoding")
                return data
            except (UnicodeDecodeError, UnicodeError):
                continue
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error with {encoding} encoding: {e}")
                continue
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error loading {file_path}: {e}")
                continue
        
        # If all encodings fail, try with error handling
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            logger.warning(f"Loaded JSON from {file_path} with character replacement")
            return data
        except Exception as e:
            raise Exception(f"Could not load JSON file with any encoding: {e}")
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str, indent: int = 2) -> None:
        """Save JSON file with proper encoding"""
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            
            logger.info(f"Successfully saved JSON to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            raise
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that JSON has required structure"""
        try:
            for key in required_keys:
                if key not in data:
                    logger.error(f"Missing required key: {key}")
                    return False
            
            logger.info("JSON structure validation passed")
            return True
            
        except Exception as e:
            logger.error(f"JSON structure validation failed: {e}")
            return False
    
    @staticmethod
    def get_json_stats(data: Dict[str, Any]) -> Dict[str, int]:
        """Get statistics about JSON data"""
        stats = {
            'total_keys': len(data.keys()),
            'collections': 0,
            'bookmarks': 0,
            'tags': 0
        }
        
        try:
            collections = data.get('collections', [])
            stats['collections'] = len(collections)
            
            total_bookmarks = 0
            total_tags = 0
            
            for collection in collections:
                links = collection.get('links', [])
                total_bookmarks += len(links)
                
                for link in links:
                    tags = link.get('tags', [])
                    total_tags += len(tags)
            
            stats['bookmarks'] = total_bookmarks
            stats['tags'] = total_tags
            
            logger.info(f"JSON stats: {stats}")
            
        except Exception as e:
            logger.error(f"Failed to calculate JSON stats: {e}")
        
        return stats