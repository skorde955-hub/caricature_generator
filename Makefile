.PHONY: install lint format test run

install:
	poetry install

lint:
	poetry run ruff check src tests

format:
	poetry run black src tests

test:
	poetry run pytest

run:
	poetry run python scripts/run_pipeline.py

