# Makefile for schema mapper project

.PHONY: install test lint clean docs

# Python settings
PYTHON := python3
VENV := venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
FLAKE8 := $(VENV)/bin/flake8
BLACK := $(VENV)/bin/black
MYPY := $(VENV)/bin/mypy

# Project settings
PROJECT := schema_mapper
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Virtual environment
$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Installation
install: $(VENV)/bin/activate

# Testing
test: install
	$(PYTEST) $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

# Linting
lint: install
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(BLACK) $(SRC_DIR) $(TEST_DIR) --check
	$(MYPY) $(SRC_DIR)

# Formatting
format: install
	$(BLACK) $(SRC_DIR) $(TEST_DIR)

# Documentation
docs: install
	mkdocs build

# Clean up
clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	rm -rf .mypy_cache
	rm -rf site

# Run development server
run:
	$(PYTHON) -m uvicorn src.app.main:app --reload

# Run profiling demo
profile:
	$(PYTHON) scripts/demo_profiling.py

# Run embedding demo
embed:
	$(PYTHON) scripts/demo_embedding_mapping.py

# Create database
db:
	$(PYTHON) src/app/database/mock_collections.py

# Run all checks
check: lint test

# Default target
all: install lint test docs 