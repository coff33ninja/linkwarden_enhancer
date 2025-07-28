#!/usr/bin/env python3
"""
Test script for GitHub caching functionality
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from linkwarden_enhancer.importers.github_importer import GitHubImporter
from linkwarden_enhancer.config.settings import load_config

def test_github_cache():
    """Test GitHub caching functionality"""
    
    print("🧪 Testing GitHub Cache Functionality")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Ensure cache is enabled
    config['github']['cache']['enabled'] = True
    config['github']['cache']['ttl_hours'] = 1  # Short TTL for testing
    
    try:
        # Initialize importer
        importer = GitHubImporter(config)
        
        # Show cache info before
        print("\n📊 Cache Info (Before):")
        cache_info = importer.get_cache_info()
        print(f"  Cache enabled: {cache_info['enabled']}")
        print(f"  Cache TTL: {cache_info['ttl_hours']} hours")
        print(f"  Cache directory: {cache_info['cache_dir']}")
        
        for name, info in cache_info['files'].items():
            if info['exists']:
                print(f"  {name.title()} cache: EXISTS (valid: {info.get('valid', 'N/A')})")
            else:
                print(f"  {name.title()} cache: NOT FOUND")
        
        # Test 1: First import (should hit API)
        print("\n🔄 Test 1: First import (should fetch from API)")
        start_time = time.time()
        
        result = importer.import_data(
            import_starred=True,
            import_owned=False,
            max_repos=5  # Limit for testing
        )
        
        first_import_time = time.time() - start_time
        print(f"  ✅ Imported {result.total_imported} repositories in {first_import_time:.2f}s")
        
        # Test 2: Second import (should use cache)
        print("\n🔄 Test 2: Second import (should use cache)")
        start_time = time.time()
        
        result2 = importer.import_data(
            import_starred=True,
            import_owned=False,
            max_repos=5
        )
        
        second_import_time = time.time() - start_time
        print(f"  ✅ Imported {result2.total_imported} repositories in {second_import_time:.2f}s")
        
        # Compare times
        if second_import_time < first_import_time * 0.5:  # Should be much faster
            print(f"  🎉 Cache working! Second import was {first_import_time/second_import_time:.1f}x faster")
        else:
            print(f"  ⚠️  Cache may not be working as expected")
        
        # Test 3: Force refresh
        print("\n🔄 Test 3: Force refresh (should ignore cache)")
        start_time = time.time()
        
        result3 = importer.import_data(
            import_starred=True,
            import_owned=False,
            max_repos=5,
            force_refresh=True
        )
        
        third_import_time = time.time() - start_time
        print(f"  ✅ Imported {result3.total_imported} repositories in {third_import_time:.2f}s")
        
        # Show cache info after
        print("\n📊 Cache Info (After):")
        cache_info_after = importer.get_cache_info()
        
        for name, info in cache_info_after['files'].items():
            if info['exists']:
                print(f"  {name.title()} cache: {info['size_bytes']} bytes, modified: {info['modified']}")
            else:
                print(f"  {name.title()} cache: NOT FOUND")
        
        # Test cache clearing
        print("\n🧹 Test 4: Cache clearing")
        importer.clear_cache()
        
        cache_info_cleared = importer.get_cache_info()
        cache_files_exist = any(info['exists'] for info in cache_info_cleared['files'].values())
        
        if not cache_files_exist:
            print("  ✅ Cache cleared successfully")
        else:
            print("  ⚠️  Some cache files still exist")
        
        print("\n🎉 GitHub cache testing completed!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_github_cache()
    sys.exit(0 if success else 1)