.PHONY: evals test

test:
	uv run pytest tests/unit/ -v

evals:
	uv run pytest tests/evals/ -v -s
