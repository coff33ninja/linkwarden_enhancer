# Contributing to Linkwarden Enhancer

Thank you for your interest in contributing to Linkwarden Enhancer! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- Git
- GitHub account
- Basic understanding of Python, AI/ML concepts, and bookmark management

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/coff33ninja/linkwarden-enhancer.git
   cd linkwarden-enhancer
   ```

3. **Set up development environment**:
   ```bash
   python dev_setup.py
   ```
   Or manually:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Write descriptive docstrings for all classes and functions
- Use meaningful variable and function names
- Keep functions focused and single-purpose

### Code Quality Tools

We use several tools to maintain code quality:

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .

# Run tests
pytest tests/
```

### Project Structure

```
linkwarden_enhancer/
├── core/              # Safety, validation, backup systems
├── ai/                # Machine learning and content analysis
├── intelligence/      # Adaptive learning and smart dictionaries
├── enhancement/       # Web scraping and metadata enhancement
├── importers/         # Multi-source data import systems
├── reporting/         # Analytics and report generation
├── cli/               # Command-line interface
├── gui/               # Web GUI interface (new)
├── utils/             # Shared utilities
├── config/            # Configuration management
├── tests/             # Test suites
├── docs/              # Documentation
└── examples/          # Usage examples
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=linkwarden_enhancer

# Run specific test file
pytest tests/test_safety_manager.py

# Run tests with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use fixtures for common test data

Example test structure:
```python
def test_bookmark_enhancement_with_valid_data():
    """Test that bookmark enhancement works with valid input data."""
    # Arrange
    input_data = create_test_bookmark_data()
    
    # Act
    result = enhance_bookmarks(input_data)
    
    # Assert
    assert result.success is True
    assert len(result.enhanced_bookmarks) > 0
```

## 📝 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def process_bookmarks(bookmarks: List[Dict], options: ProcessingOptions) -> ProcessingResult:
    """Process bookmarks with AI enhancement and safety checks.
    
    Args:
        bookmarks: List of bookmark dictionaries to process
        options: Processing configuration options
        
    Returns:
        ProcessingResult containing enhanced bookmarks and statistics
        
    Raises:
        ValidationError: If bookmark data is invalid
        ProcessingError: If processing fails
    """
```

### Documentation Updates

- Update relevant documentation when adding new features
- Include code examples in docstrings
- Update the README.md if adding major features
- Add entries to CHANGELOG.md for significant changes

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected behavior** vs actual behavior
4. **Environment information** (OS, Python version, etc.)
5. **Error messages** and stack traces
6. **Sample data** (if applicable, anonymized)

Use the bug report template:

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Run command '...'
2. With input file '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g. 3.9.7]
- Linkwarden Enhancer Version: [e.g. 1.0.0]

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

For feature requests, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** and problem you're trying to solve
3. **Propose a solution** if you have ideas
4. **Consider the scope** - is this a core feature or plugin?

## 🔧 Types of Contributions

### Core System Improvements
- Safety system enhancements
- Performance optimizations
- Error handling improvements
- Configuration management

### AI/ML Features
- New content analysis algorithms
- Improved similarity detection
- Enhanced learning capabilities
- Domain-specific analyzers

### Import/Export Features
- New platform integrations
- Format support improvements
- Data migration tools
- Sync capabilities

### User Interface
- CLI improvements
- Web GUI enhancements
- Interactive features
- Accessibility improvements

### Documentation
- Code documentation
- User guides
- API documentation
- Examples and tutorials

## 📦 Pull Request Process

1. **Create a feature branch** from `main`
2. **Make your changes** following the guidelines above
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run the test suite** and ensure all tests pass
6. **Create a pull request** with:
   - Clear title and description
   - Reference to related issues
   - Screenshots (if UI changes)
   - Testing instructions

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## 🏷️ Commit Message Guidelines

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

Examples:
```
feat(ai): add sentiment analysis for bookmark content
fix(import): handle malformed GitHub API responses
docs(cli): update command examples in README
```

## 🌟 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- GitHub contributors page

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: Request reviews on pull requests

## 📄 License

By contributing to Linkwarden Enhancer, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Linkwarden Enhancer! 🚀