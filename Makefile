# ==============================================================================
# Makefile for Project Automation
#
# Provides a unified interface for common development tasks, such as running
# the application, formatting code, and running tests.
#
# Inspired by the self-documenting Makefile pattern.
# See: https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# ==============================================================================

# Default target when 'make' is run without arguments
.DEFAULT_GOAL := help

# Specify the Streamlit app file name
STREAMLIT_APP_FILE := src/main.py

# ==============================================================================
# HELP
# ==============================================================================

.PHONY: help
help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

.PHONY: setup
setup: ## Project initial setup: install dependencies
	@echo "🐍 Installing python dependencies with Poetry..."
	@poetry install --no-root
	@echo "✅ Dependencies installed."


# ==============================================================================
# APPLICATION
# ==============================================================================

.PHONY: run
run: ## Launch the Streamlit application
	@echo "🚀 Starting Streamlit app..."
	@poetry run streamlit run $(STREAMLIT_APP_FILE)

# ==============================================================================
# CODE QUALITY
# ==============================================================================

.PHONY: format
format: ## Automatically format code using Black and Ruff
	@echo "🎨 Formatting code with black and ruff..."
	@poetry run black .
	@poetry run ruff . --fix

.PHONY: lint
lint: ## Perform static code analysis (check) using Black and Ruff
	@echo "🔬 Linting code with black and ruff..."
	@poetry run black --check .
	@poetry run ruff check .

# ==============================================================================
# TESTING
# ==============================================================================

.PHONY: test
test: ## Run the full test suite
	@echo "Running build tests..."
	@poetry run pytest tests/test_build.py -s
