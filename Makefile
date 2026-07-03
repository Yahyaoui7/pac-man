.PHONY: install run debug clean lint lint-strict


run:
	uv run python pac_man.py config.json

install:
	uv sync

debug:
	uv run python -m pdb -m pac_man.py config.json

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

lint:
	uv run flake8 . --exclude=.venv,mazegenerator-2.0.2-py3-none-any
	uv run mypy . \
		--exclude=.venv \
		--exclude=mazegenerator-2.0.2-py3-none-any \
		--warn-return-any \
		--warn-unused-ignores \
		--disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	uv run flake8 . --exclude=.venv,mazegenerator-2.0.2-py3-none-any
	uv run mypy . --strict --exclude=.venv --exclude=mazegenerator-2.0.2-py3-none-any
