# Load .env if it exists to make variables available for make
ifneq ("$(wildcard .env)","")
    include .env
    export $(shell sed 's/=.*//' .env)
endif

# Default values if not set in .env or environment
APP_HOST ?= 0.0.0.0
APP_PORT ?= 8080

.PHONY: help install lint format test validate clean run

help:
	@echo "Available commands:"
	@echo "  install    : Install dependencies using uv"
	@echo "  lint       : Run linter (ruff)"
	@echo "  format     : Run formatter (ruff)"
	@echo "  test       : Run unit tests (pytest)"
	@echo "  validate   : Run OCR validation on images in tests/samples/"
	@echo "  clean      : Remove temporary files"
	@echo "  run        : Run the FastAPI application"

install:
	uv sync
	$(MAKE) setup-hooks

setup-hooks:
	@echo "Installing git hooks..."
	uv run pre-commit install --hook-type commit-msg
	uv run pre-commit install --hook-type pre-push
	@echo "Hooks installed successfully."

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	PYTHONPATH=. uv run pytest tests/

validate:
	PYTHONPATH=. uv run python tests/test_sample.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf logs
	rm -rf storage

run:
	uv run uvicorn src.main:app --reload --host $(APP_HOST) --port $(APP_PORT)
