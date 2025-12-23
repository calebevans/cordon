# Contributing to Cordon

Thank you for your interest in contributing to Cordon! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you are expected to uphold a welcoming and inclusive environment. Please be respectful of differing viewpoints and experiences.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cordon.git
   cd cordon
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/calebevans/cordon.git
   ```

## Development Setup

See the [README](README.md#from-source) for basic installation. For development:

```bash
# Install with development dependencies
uv pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

## Running Tests

We use pytest for testing with coverage reporting.

```bash
# Run all tests with coverage
pytest

# Run tests without coverage
pytest --no-cov

# Run a specific test file
pytest tests/test_analyzer.py

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_basic"
```

### Test Requirements

- Tests should pass on Python 3.10, 3.11, 3.12, and 3.13
- New features should include corresponding tests
- Aim for meaningful test coverage, not just line coverage

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Automatic Checks

Pre-commit hooks will automatically check your code before each commit:

```bash
# Run all pre-commit checks manually
pre-commit run --all-files
```

### Style Guidelines

- Follow PEP 8 conventions
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Write docstrings for public functions and classes

## Submitting Changes

### Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits

3. **Ensure all checks pass**:
   ```bash
   pre-commit run --all-files
   pytest
   ```

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** against `main`

### Pull Request Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Explain what changes you made and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update README or docs if needed
- **One feature per PR**: Keep pull requests focused

### Commit Messages

Use clear, descriptive commit messages:

```
Add support for JSON output format

- Implement JSONFormatter class
- Add --format flag to CLI
- Update documentation with examples
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Python version**: `python --version`
- **OS and version**: e.g., Ubuntu 22.04, macOS 14.0
- **Cordon version**: `cordon --version` or `pip show cordon`
- **GPU info** (if applicable): GPU model, CUDA version
- **Steps to reproduce**: Minimal example to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Log output**: Any error messages or stack traces

### Feature Requests

When requesting features:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: How you envision the feature working
- **Alternatives**: Other solutions you've considered

## Development Tips

### Testing with Different Backends

```bash
# Test with sentence-transformers (default)
cordon examples/apache_sample.log

# Test with llama.cpp
cordon --backend llama-cpp examples/apache_sample.log

# Test with remote API (requires API key)
OPENAI_API_KEY=... cordon --backend remote --model-name openai/text-embedding-3-small examples/apache_sample.log
```

### Debugging

```bash
# Enable verbose output
cordon --detailed examples/apache_sample.log

# Use smaller test files during development
head -100 large.log > test_small.log
```

## Questions?

If you have questions that aren't covered here, feel free to:

- Open a [Discussion](https://github.com/calebevans/cordon/discussions/categories/q-a) on GitHub
- Ask in an existing issue if it's related

Thank you for contributing to Cordon!
