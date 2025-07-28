#!/usr/bin/env python3
"""
Linkwarden Enhancer CLI - Standalone entry point

This script provides a convenient way to run the Linkwarden Enhancer CLI
without needing to install the package.

Usage:
    python cli.py [command] [options]
    
Examples:
    python cli.py process input.json output.json --interactive
    python cli.py import --github --github-token TOKEN --github-username USER -o output.json
    python cli.py menu
    python cli.py stats --all
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path so we can import linkwarden_enhancer
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from linkwarden_enhancer.cli.main_cli import main
    
    if __name__ == '__main__':
        sys.exit(main())
        
except ImportError as e:
    print(f"❌ Failed to import Linkwarden Enhancer: {e}")
    print("\nMake sure you have installed the required dependencies:")
    print("  pip install -r requirements.txt")
    print("\nOr install the package in development mode:")
    print("  pip install -e .")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)