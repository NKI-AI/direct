.PHONY: clean clean-test clean-pyc clean-cpp clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-cpp clean-ipynb clean-test clean-docs ## remove all build, test, coverage, docs and Python and C++ artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-cpp: ## remove compiled C++ extension artifacts
	find direct -name '*.so' -exec rm -f {} +
	find direct -name '*.pyd' -exec rm -f {} +

clean-ipynb: ## remove ipynb artifacts
	find . -name '.ipynb_checkpoints' -exec rm -rf {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .ruff_cache

clean-docs: ## clean sphinx docs
	rm -rf docs/_build/
	rm -rf docs/_project_figures/
	rm -f docs/direct.rst
	rm -f docs/direct.*.rst

lint: ## check style with ruff (lint + format)
	uv run ruff check direct tests
	uv run ruff format --check direct tests

format: ## auto-format and fix lint issues with ruff
	uv run ruff check --fix direct tests
	uv run ruff format direct tests

test: ## run tests with pytest
	uv run pytest --ignore=projects

coverage: ## check code coverage quickly with the default Python
	uv run coverage run --source direct -m pytest
	uv run coverage report -m
	uv run coverage html
	$(BROWSER) htmlcov/index.html

docs: clean-docs ## generate Sphinx HTML documentation, including API docs
	uv sync --all-groups
	uv run sphinx-apidoc -o docs/ direct --separate --module-first --no-toc
	$(MAKE) -C docs clean
	$(MAKE) -C docs html SPHINXOPTS="-Q -w $(CURDIR)/docs/_build/warnings.log"

viewdocs: docs ## open documentation in browser
	$(BROWSER) docs/_build/html/index.html

uploaddocs: docs ## upload documentation to the docs server
	rsync -avh docs/_build/html/ docs@142.93.235.165:/var/www/html/docs/direct --delete

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	uv run twine upload dist/*

dist: clean ## builds source and wheel package
	uv build
	ls -l dist

install: clean ## install the package and dev dependencies into a uv-managed environment
	uv sync --all-groups
