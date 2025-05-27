# Makefile for Wikipedia RAG system

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies using Poetry"
	@echo "  make clean     - Remove the virtual environment"
	@echo "  make format    - Format code using isort and black"
	@echo "  make lint      - Run all linters (isort, flake8, mypy, pylint)"

# Install dependencies
# Checks if pyproject.toml or poetry.lock is newer than a potential marker file (not strictly necessary for poetry but good practice)
.PHONY: install
install: pyproject.toml poetry.lock
	@echo "Installing dependencies using Poetry..."
	poetry install

# Clean the virtual environment
.PHONY: clean
clean:
	@echo "Removing virtual environment managed by Poetry..."
	poetry env remove python || true # Use '|| true' to avoid error if env doesn't exist
	@echo "Cleaning complete."

.PHONY: format
format:
	@echo "Running formatters..."
	poetry run isort --profile black src
	poetry run black src

# Lint code (check only)
.PHONY: isort_check
isort_check:
	@echo "Checking imports with isort..."
	poetry run isort --check src

.PHONY: black_check
black_check:
	@echo "Checking formatting with black..."
	poetry run black --check --diff src

.PHONY: flake
flake:
	@echo "Running flake8..."
	poetry run flake8 src

.PHONY: mypy
mypy:
	@echo "Running mypy..."
	poetry run mypy src

.PHONY: pylint
pylint:
	@echo "Running pylint..."
	poetry run pylint src

.PHONY: lint
lint: isort_check black_check flake mypy pylint

# Declare files as prerequisites to potentially trigger install if they change
pyproject.toml:
poetry.lock: