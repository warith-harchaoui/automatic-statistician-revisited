# Fast local checks, the same ones continuous integration (CI) runs.
#   make check   format, lint, prose and the fast tests
#   make test-all   every test, including the slow searches

.PHONY: check format lint prose test test-all

check: format lint prose test

format:
	ruff format --check src tests scripts benchmark

lint:
	ruff check src tests scripts benchmark

prose:
	python scripts/lint_prose.py

test:
	pytest -m "not slow"

test-all:
	pytest
