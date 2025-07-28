"""Setup script for Linkwarden Enhancer - Standalone Package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="linkwarden-enhancer",
    version="1.0.0",
    author="DJ",
    description="AI-powered bookmark management system with enterprise-grade safety features - Standalone Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coff33ninja/linkwarden-enhancer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP :: Browsers",
        "Topic :: Office/Business :: Groupware",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Archiving :: Backup",
        "Topic :: Utilities",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "bandit>=1.7.0",
            "safety>=1.10.0",
        ],
        "gui": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "python-multipart>=0.0.5",
            "websockets>=10.0",
        ],
        "all": [
            "torch>=1.9.0",
            "transformers>=4.12.0",
            "chromadb>=0.3.0",
        ],
    },
    include_package_data=True,
    package_data={
        "linkwarden_enhancer": [
            "data/*.json",
            "examples/*.json",
            "gui/static/**/*",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/coff33ninja/linkwarden-enhancer/issues",
        "Source": "https://github.com/coff33ninja/linkwarden-enhancer",
        "Documentation": "https://github.com/coff33ninja/linkwarden-enhancer/tree/main/docs",
        "Changelog": "https://github.com/coff33ninja/linkwarden-enhancer/blob/main/CHANGELOG.md",
    },
    keywords=[
        "bookmarks",
        "linkwarden",
        "ai",
        "machine-learning",
        "bookmark-manager",
        "web-scraping",
        "github-integration",
        "content-analysis",
        "duplicate-detection",
        "backup-system",
        "cli",
        "automation",
        "standalone",
    ],
)