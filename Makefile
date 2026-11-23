.PHONY: install test lint bench demo
install:
	pip install -e ".[dev]"
test:
	python -m pytest
lint:
	ruff check src tests
bench:
	python benchmarks/run_benchmark.py
demo:
	python examples/demo.py
