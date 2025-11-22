.PHONY: help install install-dev install-faiss install-faiss-gpu test test-verbose coverage lint format typecheck pre-commit clean clean-all

# default target - show help
help:
	@echo "Cordon Development Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install         Install package in development mode"
	@echo "  make install-dev     Install with development dependencies"
	@echo "  make install-faiss   Install with FAISS CPU support (better performance)"
	@echo "  make install-faiss-gpu Install with FAISS GPU support (CUDA required)"
	@echo ""
	@echo "  make test            Run all tests"
	@echo "  make test-verbose    Run tests with verbose output"
	@echo "  make coverage        Run tests with coverage report"
	@echo ""
	@echo "  make lint            Run linter (ruff) on source code"
	@echo "  make format          Auto-format code with ruff"
	@echo "  make typecheck       Run type checker (mypy)"
	@echo "  make check           Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "  make pre-commit      Install pre-commit hooks"
	@echo "  make pre-commit-run  Run pre-commit on all files"
	@echo ""
	@echo "  make clean           Remove Python cache and build artifacts"
	@echo "  make clean-all       Deep clean (cache, build, venv, coverage)"

# installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-faiss:
	pip install -e ".[faiss-cpu]"

install-faiss-gpu:
	pip install -e ".[faiss-gpu]"

# testing targets
test:
	OMP_NUM_THREADS=1 pytest tests/ --no-cov

test-verbose:
	OMP_NUM_THREADS=1 pytest tests/ -v --no-cov

coverage:
	OMP_NUM_THREADS=1 pytest tests/ --cov=cordon --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

# code quality targets
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/

check: lint typecheck test
	@echo ""
	@echo "✓ All checks passed!"

# pre-commit targets
pre-commit:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# cleaning and learning targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "✓ Cleaned Python cache and build artifacts"

clean-all: clean
	rm -rf .venv/
	@echo "✓ Deep clean complete (removed virtual environment)"

