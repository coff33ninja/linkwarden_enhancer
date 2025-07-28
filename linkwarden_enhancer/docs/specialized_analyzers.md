# Specialized Content Analyzers

The specialized analyzers provide domain-specific analysis for different types of bookmark content, enabling more accurate categorization and tagging based on the specific characteristics of gaming, development, research, and other specialized content.

## Overview

The specialized analysis system consists of three main analyzers:

1. **GamingAnalyzer** - Analyzes gaming-related content
2. **DevelopmentAnalyzer** - Analyzes development and self-hosting content  
3. **ResearchAnalyzer** - Analyzes research, educational, and hobby content

## Usage

### Basic Usage

```python
from linkwarden_enhancer.ai.specialized_analyzers import SpecializedAnalysisEngine

engine = SpecializedAnalysisEngine()

# Analyze a single piece of content
result = engine.get_best_analysis(url, title, content)
if result:
    print(f"Domain: {result.domain}")
    print(f"Content Type: {result.content_type}")
    print(f"Tags: {result.specialized_tags}")
    print(f"Confidence: {result.confidence_score}")
```

### Advanced Usage

```python
# Get all matching analysis results
all_results = engine.analyze_content(url, title, content)

# Get combined tags from all analyzers
all_tags = engine.get_all_specialized_tags(url, title, content)

# Get combined metadata
metadata = engine.get_combined_metadata(url, title, content)
```

## Gaming Analyzer

Specializes in analyzing gaming-related content including:

### Supported Content Types
- **Genshin Impact**: Specialized detection for characters, builds, tools
- **Gaming Platforms**: Steam, Epic Games, GOG, console platforms
- **Gaming Communities**: Reddit gaming subreddits, Discord servers
- **Game Development**: Unity, Unreal Engine, indie development tools

### Example Tags Generated
- `Genshin Impact`, `Character: Ganyu`, `Steam`, `PC Gaming`
- `Game Development`, `Engine: Unity`, `Gaming Community`

### Metadata Extracted
- Game-specific information (characters, platforms)
- Platform details (Steam app IDs, repository info)
- Community type and location

## Development Analyzer

Specializes in development and self-hosting content:

### Supported Content Types
- **GitHub Repositories**: Language detection, repository categorization
- **Programming Languages**: Python, JavaScript, TypeScript, etc.
- **Cloud Platforms**: AWS, Azure, GCP services
- **Self-Hosting**: Docker, Kubernetes, homelab tools
- **Documentation**: API docs, tutorials, guides

### Example Tags Generated
- `GitHub`, `Language: Python`, `AWS`, `Self-Hosting`
- `Documentation`, `Tool: Docker`, `Cloud Platform`

### Metadata Extracted
- Repository information (owner, name, type)
- Programming languages and frameworks
- Cloud provider and services
- Self-hosting categories

## Research Analyzer

Specializes in research, educational, and hobby content:

### Supported Content Types
- **Academic Papers**: ArXiv, PubMed, IEEE publications
- **Educational Content**: Coursera, edX, Khan Academy
- **News Articles**: Tech news, general news sources
- **Wikipedia**: Reference materials
- **Hobby Interests**: Cooking, fitness, photography, etc.

### Example Tags Generated
- `ArXiv`, `Research Paper`, `Wikipedia`, `Coursera`
- `Interest: Cooking`, `Field: Computer Science`, `News`

### Metadata Extracted
- Publication information (type, year, venue)
- Research fields and academic disciplines
- Educational platform and content type
- Hobby and interest categories

## Integration with Main System

The specialized analyzers are designed to integrate with the main AI analysis pipeline:

```python
# In your main analysis workflow
from linkwarden_enhancer.ai import SpecializedAnalysisEngine

def analyze_bookmark(bookmark):
    engine = SpecializedAnalysisEngine()
    
    # Get specialized analysis
    specialized_result = engine.get_best_analysis(
        bookmark['url'], 
        bookmark['title'], 
        bookmark.get('content', '')
    )
    
    if specialized_result:
        # Add specialized tags to bookmark
        bookmark['specialized_tags'] = specialized_result.specialized_tags
        bookmark['specialized_metadata'] = specialized_result.metadata
        bookmark['content_domain'] = specialized_result.domain
        
        # Use suggestions for organization
        bookmark['organization_suggestions'] = specialized_result.suggestions
    
    return bookmark
```

## Configuration

The analyzers use predefined patterns and keywords that can be extended:

- Domain patterns for different platforms
- Keyword lists for content detection
- Category mappings for specialized content

## Performance Considerations

- Analyzers run in parallel and only process content they can handle
- Confidence scores help select the best analysis result
- Caching can be implemented for repeated analysis of similar content

## Testing

Comprehensive tests are available in `linkwarden_enhancer/tests/test_specialized_analyzers.py` covering:

- Individual analyzer functionality
- Combined analysis scenarios
- Edge cases and error handling
- Performance with various content types