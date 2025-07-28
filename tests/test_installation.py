#!/usr/bin/env python3
"""
Simple installation test for Linkwarden Enhancer
"""

import sys
import importlib
from pathlib import Path


def test_python_version():
    """Test Python version compatibility"""
    version = sys.version_info
    assert version.major >= 3 and version.minor >= 8, "Python version must be 3.8+"


def test_package_import():
    """Test if the package can be imported"""
    try:
        import main
        print(f"✅ Main script import: main.py")
        assert True
    except ImportError as e:
        print(f"❌ Main script import failed: {e}")
        assert False


def test_core_modules():
    """Test if core modules can be imported"""
    modules = [
        'config',
        'utils',
        'main',
    ]
    
    failed_modules = []
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ Module: {module_name}")
        except ImportError as e:
            print(f"❌ Module: {module_name} - {e}")
            failed_modules.append(module_name)
    
    assert len(failed_modules) == 0


def test_dependencies():
    """Test if key dependencies are available"""
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('requests', 'Requests'),
        ('bs4', 'BeautifulSoup4'),
        ('sklearn', 'scikit-learn'),
        ('nltk', 'NLTK'),
        ('click', 'Click'),
    ]
    
    failed_deps = []
    
    for module_name, display_name in dependencies:
        try:
            importlib.import_module(module_name)
            print(f"✅ Dependency: {display_name}")
        except ImportError:
            print(f"❌ Dependency: {display_name}")
            failed_deps.append(display_name)
    
    assert len(failed_deps) == 0


def test_optional_dependencies():
    """Test optional dependencies"""
    optional_deps = [
        ('github', 'PyGithub'),
        ('selenium', 'Selenium'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('ollama', 'Ollama'),
    ]
    
    for module_name, display_name in optional_deps:
        try:
            importlib.import_module(module_name)
            print(f"✅ Optional: {display_name}")
        except ImportError:
            print(f"⚠️  Optional: {display_name} (not installed)")


def test_directories():
    """Test if required directories exist"""
    directories = ['data', 'backups', 'cache', 'models', 'logs']
    
    for directory in directories:
        path = Path(directory)
        if path.exists():
            print(f"✅ Directory: {directory}")
        else:
            print(f"⚠️  Directory: {directory} (will be created automatically)")


def test_configuration():
    """Test configuration loading"""
    try:
        from config.settings import load_config
        config = load_config()
        print("✅ Configuration: Loaded successfully")
        assert True
    except Exception as e:
        print(f"❌ Configuration: Failed to load - {e}")
        assert False


def main():
    """Run all tests"""
    print("🧪 Testing Linkwarden Enhancer Installation")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Import", test_package_import),
        ("Core Modules", test_core_modules),
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
    ]
    
    failed_tests = []
    
    for test_name, test_function in tests:
        print(f"\n📋 Testing: {test_name}")
        if not test_function():
            failed_tests.append(test_name)
    
    print(f"\n📋 Testing: Optional Dependencies")
    test_optional_dependencies()
    
    print(f"\n📋 Testing: Directories")
    test_directories()
    
    print("\n" + "=" * 50)
    
    if failed_tests:
        print(f"❌ Installation test failed: {', '.join(failed_tests)}")
        print("\nPlease run setup_dev.py to fix installation issues.")
        sys.exit(1)
    else:
        print("✅ Installation test passed!")
        print("\n🎯 Ready to use Linkwarden Enhancer!")
        print("   • Run: linkwarden-enhancer --help")
        print("   • Test with your data: linkwarden-enhancer --input backup.json --dry-run")


if __name__ == "__main__":
    main()