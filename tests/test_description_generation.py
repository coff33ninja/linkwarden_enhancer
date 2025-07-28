#!/usr/bin/env python3
"""Test script for description generation engine"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhancement.description_generator import DescriptionGenerator
from enhancement.meta_description_extractor import MetaDescriptionExtractor
from enhancement.ai_summarizer import AISummarizer
from enhancement.content_extractor import ContentExtractor


def test_description_generation():
    """Test the description generation system"""

    # Test configuration
    config = {
        "description": {
            "min_length": 50,
            "max_length": 200,
            "preserve_user_descriptions": True,
            "enable_meta_extraction": True,
            "enable_ai_summarization": False,  # Disable AI for testing
            "enable_content_extraction": True,
        },
        "enhancement": {
            "min_description_length": 20,
            "max_description_length": 300,
            "min_snippet_length": 50,
            "max_snippet_length": 200,
        },
        "ai": {
            "ollama": {
                "host": "localhost:11434",
                "model": "llama2",
                "auto_start": False,
                "auto_pull": False,
            }
        },
    }

    print("Testing Description Generation Engine")
    print("=" * 50)

    # Initialize the description generator
    try:
        generator = DescriptionGenerator(config)
        print("✓ Description generator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize description generator: {e}")
        return False

    # Test data
    test_cases = [
        {
            "title": "Python Web Scraping Tutorial",
            "content": "Learn how to scrape websites using Python with BeautifulSoup and requests. This comprehensive guide covers HTML parsing, handling forms, and dealing with JavaScript-rendered content.",
            "url": "https://example.com/python-scraping-tutorial",
            "existing_description": "My personal notes on web scraping",
            "html_content": """
            <html>
                <head>
                    <meta name="description" content="Complete guide to web scraping with Python, BeautifulSoup, and requests library">
                    <meta property="og:description" content="Learn web scraping techniques with practical Python examples">
                </head>
                <body>
                    <main>
                        <p>Web scraping is the process of extracting data from websites programmatically. This tutorial will teach you how to use Python libraries like BeautifulSoup and requests to scrape web content effectively.</p>
                    </main>
                </body>
            </html>
            """,
        },
        {
            "title": "GitHub Repository: awesome-python",
            "content": "A curated list of awesome Python frameworks, libraries, software and resources.",
            "url": "https://github.com/vinta/awesome-python",
            "existing_description": "",
            "html_content": """
            <html>
                <head>
                    <meta property="og:description" content="A curated list of awesome Python frameworks, libraries, software and resources">
                </head>
                <body>
                    <article>
                        <p>Awesome Python is a curated list of awesome Python frameworks, libraries, software and resources. It contains links to libraries for web development, data science, machine learning, and more.</p>
                    </article>
                </body>
            </html>
            """,
        },
    ]

    # Test each case
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['title']}")
        print("-" * 40)

        try:
            result = generator.generate_description(
                title=test_case["title"],
                content=test_case["content"],
                url=test_case["url"],
                existing_description=test_case["existing_description"],
                html_content=test_case["html_content"],
            )

            print(f"✓ Generated description: {result.final_description}")
            print(f"  Source: {result.source_used.value}")
            print(f"  Quality: {result.quality_score:.2f}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Length: {len(result.final_description)} chars")
            print(f"  Processing time: {result.processing_time:.3f}s")
            print(f"  Candidates evaluated: {len(result.candidates_evaluated)}")
            print(f"  Preservation applied: {result.preservation_applied}")
            print(f"  Merging applied: {result.merging_applied}")
            print(f"  Validation passed: {result.validation_passed}")

            if result.validation_issues:
                print(f"  Issues: {', '.join(result.validation_issues)}")

        except Exception as e:
            print(f"✗ Test case {i} failed: {e}")
            import traceback

            traceback.print_exc()

    # Test individual components
    print(f"\nTesting Individual Components")
    print("-" * 40)

    # Test meta description extractor
    try:
        meta_extractor = MetaDescriptionExtractor(config)
        meta_result = meta_extractor.extract_from_html(test_cases[0]["html_content"])
        print(
            f"✓ Meta extractor: {meta_result.description[:50]}..."
            if meta_result.description
            else "✓ Meta extractor: No description found"
        )
    except Exception as e:
        print(f"✗ Meta extractor failed: {e}")

    # Test content extractor
    try:
        content_extractor = ContentExtractor(config)
        snippet = content_extractor.extract_content_snippet(
            test_cases[0]["html_content"], test_cases[0]["url"], test_cases[0]["title"]
        )
        print(
            f"✓ Content extractor: {snippet.text[:50]}..."
            if snippet.text
            else "✓ Content extractor: No snippet found"
        )
    except Exception as e:
        print(f"✗ Content extractor failed: {e}")

    # Get generator stats
    try:
        stats = generator.get_generator_stats()
        print(f"\n✓ Generator stats retrieved: {len(stats)} configuration items")
    except Exception as e:
        print(f"✗ Failed to get generator stats: {e}")

    print(f"\nDescription Generation Engine Test Complete!")
    return True


if __name__ == "__main__":
    test_description_generation()
