.SHELLFLAGS := -eu -c
PYTHON_MIN_VERSION := 3.11
PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_STAMP := $(VENV)/.installed-dev
PYTHON_CANDIDATES = $(VENV_PYTHON) python3.13 python3.12 python3.11 $(PYTHON)
BOOTSTRAP_PYTHON ?= $(shell for py in $(PYTHON_CANDIDATES); do \
	if command -v $$py >/dev/null 2>&1 && $$py -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1; then \
		command -v $$py; \
		break; \
	fi; \
done)

.PHONY: check-python install lint format-check typecheck test repository-verify verify deploy-cloudflare-pages

check-python:
	@if [ -z "$(BOOTSTRAP_PYTHON)" ]; then \
		echo "Python $(PYTHON_MIN_VERSION)+ is required." >&2; \
		echo "Install Python $(PYTHON_MIN_VERSION)+ or run: make BOOTSTRAP_PYTHON=/path/to/python$(PYTHON_MIN_VERSION) <target>" >&2; \
		exit 1; \
	fi
	@$(BOOTSTRAP_PYTHON) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' || { \
		echo "BOOTSTRAP_PYTHON=$(BOOTSTRAP_PYTHON) is not Python $(PYTHON_MIN_VERSION)+." >&2; \
		exit 1; \
	}

$(VENV_PYTHON): check-python
	@if [ ! -x "$(VENV_PYTHON)" ] || ! $(VENV_PYTHON) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then \
		rm -rf $(VENV); \
		$(BOOTSTRAP_PYTHON) -m venv $(VENV); \
	fi

$(VENV_STAMP): pyproject.toml $(VENV_PYTHON)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -e ".[dev]"
	touch $(VENV_STAMP)

install: $(VENV_STAMP)

lint: install
	$(VENV_PYTHON) -m ruff check api benchmarks edge integrations serving src tests

format-check: install
	$(VENV_PYTHON) -m black --check api benchmarks src tests

typecheck: install
	$(VENV_PYTHON) -m mypy --ignore-missing-imports src api

test: install
	MPLBACKEND=Agg WELD_DEFECT_CI=1 $(VENV_PYTHON) -m pytest -q

repository-verify: install
	$(VENV_PYTHON) scripts/validate_repository_surface.py
	$(VENV_PYTHON) scripts/validate_architecture_blueprint.py

verify: lint format-check typecheck test repository-verify

deploy-cloudflare-pages:
	npx --yes wrangler@latest pages deploy site --project-name weld-defect-vision
