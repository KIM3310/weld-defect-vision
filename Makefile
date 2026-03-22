.PHONY: install lint format typecheck test test-cov quality-check verify verify-strict clean run-api run-dashboard help

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF   := $(PYTHON) -m ruff
MYPY   := $(PYTHON) -m mypy

help: ## Show this help message
	@echo "weld-defect-vision - Welding Defect Detection System"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff linter
	$(RUFF) check app/ dashboard/ tests/
	@echo "Lint passed."

format: ## Auto-format code with ruff
	$(RUFF) format app/ dashboard/ tests/
	$(RUFF) check --fix app/ dashboard/ tests/

typecheck: ## Run mypy type checker
	$(MYPY) app/ --ignore-missing-imports
	@echo "Type check passed."

test: ## Run test suite
	$(PYTEST) tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -v --cov=app --cov-report=term-missing --cov-report=html

quality-check: lint typecheck test ## Run all quality gates (lint + typecheck + test)
	@echo ""
	@echo "All quality gates passed."

verify: lint test ## Run the default verification gate

verify-strict: quality-check ## Run the strict verification gate (includes mypy)

clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete  2>/dev/null || true
	@echo "Cleaned."

run-api: ## Start FastAPI inference server (dev mode)
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard: ## Start Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

generate-samples: ## Generate synthetic sample images for testing
	$(PYTHON) -c "from tests.conftest import generate_all_samples; generate_all_samples()"
