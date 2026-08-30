.PHONY: ingest build query test clean

ingest:
	uv run python -m nq_research.cli ingest

build: ingest
	uv run python -m nq_research.cli build

query:
	uv run python -m nq_research.cli query-example

test:
	uv run --with pytest python -m pytest tests/ -q

clean:
	rm -rf data/raw/* data/nq_research.duckdb .pytest_cache **/__pycache__