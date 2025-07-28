#!/usr/bin/env python3
"""Test script to preview Linkwarden import"""

import sys
from pathlib import Path

from importers.linkwarden_importer import LinkwardenImporter
from config.settings import load_config

def main():
    # Load default config
    config = load_config()
    
    # Initialize importer
    importer = LinkwardenImporter(config)
    
    # Preview the backup
    preview = importer.preview_import('backup.json', max_items=5)
    
    print("📋 Linkwarden Backup Preview:")
    print(f"  📁 Collections: {preview.get('total_collections', 0)}")
    print(f"  🔖 Bookmarks: {preview.get('total_bookmarks', 0)}")
    print(f"  🏷️  Tags: {preview.get('total_tags', 0)}")
    print(f"  📊 File Size: {preview.get('file_size_mb', 0)} MB")
    
    print("\n📝 Sample Bookmarks:")
    for i, bookmark in enumerate(preview.get('sample_bookmarks', []), 1):
        print(f"  {i}. {bookmark['name']}")
        print(f"     URL: {bookmark['url']}")
        print(f"     Collection: {bookmark['collection']}")
        print(f"     Tags: {', '.join(bookmark['tags'])}")
        print()

if __name__ == '__main__':
    main()