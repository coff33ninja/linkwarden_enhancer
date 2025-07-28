"""Version information utilities"""

import sys
from pathlib import Path

from .. import __version__, __author__


def get_version_info() -> str:
    """Get comprehensive version information"""
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    version_info = f"""
Linkwarden Enhancer v{__version__}
Author: {__author__}

Python: {python_version}
Platform: {sys.platform}

Features:
  ✓ AI-powered bookmark analysis
  ✓ GitHub integration (stars & repositories)
  ✓ Smart dictionaries with continuous learning
  ✓ Multi-source import (Linkwarden, GitHub, browsers)
  ✓ Comprehensive safety checks and backups
  ✓ Ollama integration for local LLM
  ✓ Web scraping with multiple engines
  ✓ Semantic similarity and clustering

For help: linkwarden-enhancer --help
Documentation: https://github.com/yourusername/linkwarden-enhancer
"""
    
    return version_info.strip()


def get_short_version() -> str:
    """Get short version string"""
    return f"Linkwarden Enhancer v{__version__}"