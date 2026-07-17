# exactIMM — developer task runner
#
# Every target uses the project virtual-env interpreter explicitly
# (.venv/bin/python) so commands never run against a wrong, system-wide
# Python by accident — the classic cause of "No module named 'scipy'" or
# "requires a different Python: 3.11 not in '>=3.14'" failures.
#
# Quick start:
#   make venv        # create .venv with Python 3.14
#   make install     # editable install + dev deps
#   make test        # run the 209-test suite
#
# Run `make help` for the full list.

PYTHON_SYS ?= python3.14
VENV       := .venv
PY         := $(VENV)/bin/python
PIP        := $(PY) -m pip

.DEFAULT_GOAL := help

.PHONY: help venv install install-gui install-paper test cov lint fmt typecheck check clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(PY): ## (internal) create the venv if missing
	$(PYTHON_SYS) -m venv $(VENV)
	$(PIP) install --upgrade pip

venv: $(PY) ## Create the virtual-env with $(PYTHON_SYS)
	@$(PY) --version

install: $(PY) ## Editable install with dev dependencies
	$(PIP) install -e ".[dev]"

install-gui: $(PY) ## Editable install with GUI extras (PyQt6 + matplotlib)
	$(PIP) install -e ".[gui]"

install-paper: $(PY) ## Editable install with paper-reproduction extras
	$(PIP) install -e ".[paper]"

test: ## Run the full pytest suite
	$(PY) -m pytest

cov: ## Run pytest with coverage report
	$(PY) -m pytest --cov=prg --cov-report=term --cov-report=xml --tb=short

lint: ## Lint with ruff (check only)
	$(PY) -m ruff check .

fmt: ## Auto-fix lint issues with ruff
	$(PY) -m ruff check --fix .

typecheck: ## Run mypy (strict on prg/utils/ab_constraint.py)
	$(PY) -m mypy

check: lint typecheck test ## Run lint + typecheck + tests (CI parity)

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
