"""
Linkwarden Enhancer

An intelligent, AI-powered bookmark management system that transforms your 
Linkwarden bookmarks into a smart, continuously learning organization tool.
"""

__version__ = "0.1.0"
__author__ = "DJ"
__email__ = "your.email@example.com"

# Import available modules
from .core.safety_manager import SafetyManager
from .importers.github_importer import GitHubImporter
from .importers.linkwarden_importer import LinkwardenImporter
from .importers.universal_importer import UniversalImporter
from .intelligence.dictionary_manager import SmartDictionaryManager

# Import CLI components
from .cli.main_cli import MainCLI

__all__ = [
    "SafetyManager",
    "GitHubImporter",
    "LinkwardenImporter",
    "UniversalImporter",
    "SmartDictionaryManager",
    "MainCLI",
]