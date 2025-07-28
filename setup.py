from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="linkwarden-enhancer",
    version="0.1.0",
    author="DJ",
    author_email="coff33ninja@example.com",
    description="An intelligent, AI-powered bookmark management system for Linkwarden",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/coff33ninja/linkwarden_enhancer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet :: WWW/HTTP :: Browsers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
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
        ],
        "advanced": [
            "torch>=1.9.0",
            "transformers>=4.12.0",
            "chromadb>=0.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "linkwarden-enhancer=linkwarden_enhancer.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "linkwarden_enhancer": [
            "config/*.py",
            "data/*.json",
            "models/*.pkl",
        ],
    },
)