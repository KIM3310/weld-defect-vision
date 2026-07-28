.SHELLFLAGS := -eu -o pipefail -c
PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_STAMP := $(VENV)/.installed-dev

.PHONY: check-python install lint format-check typecheck test repository-verify verify deploy-cloudflare-pages

check-python:
	@interpreter="$(PYTHON)"; \
	if [ -x "$(VENV_PYTHON)" ] && $(VENV_PYTHON) -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then \
		interpreter="$(VENV_PYTHON)"; \
	fi; \
	"$$interpreter" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'

$(VENV_PYTHON): check-python
	@if [ ! -x "$(VENV_PYTHON)" ] || ! $(VENV_PYTHON) -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >/dev/null 2>&1; then \
		rm -rf $(VENV); \
		$(PYTHON) -m venv $(VENV); \
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
	$(VENV_PYTHON) -m pytest -q

repository-verify: install
	$(VENV_PYTHON) scripts/validate_repository_surface.py
	$(VENV_PYTHON) scripts/validate_architecture_blueprint.py

verify: lint format-check typecheck test repository-verify

deploy-cloudflare-pages:
	npx --yes wrangler@latest pages deploy site --project-name weld-defect-vision
