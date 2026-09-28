.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format typecheck test test-fast coverage check clean

help:  ## Show the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create the uv environment with dev tools and install the git hooks
	$(UV) sync
	$(UV) run pre-commit install

lint:  ## Lint and check formatting with ruff
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Auto-fix lint findings and reformat the code
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:  ## Type-check with mypy in strict mode
	$(UV) run mypy

test:  ## Run the whole test suite
	$(UV) run pytest -q

test-fast:  ## Run only the tests that do not spawn subprocesses
	$(UV) run pytest -q -m "not slow"

coverage:  ## Run the tests with branch coverage
	$(UV) run pytest -q --cov --cov-report=term-missing

check: lint typecheck test  ## Everything CI runs: lint, typecheck, test

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage .coverage.* htmlcov dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
