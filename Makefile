.PHONY: install test lint type fmt all

install:
	pip install -e ".[dev,sim]"
	pre-commit install

test:
	pytest -v

lint:
	ruff check src tests

type:
	mypy src

fmt:
	ruff format src tests

all: fmt lint type test
