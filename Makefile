.PHONY: setup test lint format

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

format:
	python -m ruff format src tests
