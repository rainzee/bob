.DEFAULT_GOAL := help

.PHONY: help install check test clean-build build publish build-and-publish

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "%-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Sync Python dependencies
	@echo "==> Syncing Python dependencies with uv"
	uv sync

check: ## Validate the lockfile, lint, check formatting, and type-check src
	@echo "==> Verifying uv.lock matches pyproject.toml"
	uv lock --locked
	@echo "==> Running ruff"
	uv run ruff check .
	@echo "==> Checking formatting"
	uv run ruff format --check .
	@echo "==> Running mypy on src"
	uv run mypy src

test: ## Run pytest with doctests enabled
	@echo "==> Running pytest with doctests"
	uv run python -m pytest --doctest-modules

clean-build: ## Remove local build artifacts
	@echo "==> Removing dist/"
	rm -rf dist

build: clean-build ## Build source and wheel distributions
	@echo "==> Building package distributions"
	uv build

publish: ## Publish the contents of dist/ with uv
	@echo "==> Publishing dist/ with uv"
	uv publish

build-and-publish: build publish ## Build distributions, then publish them
