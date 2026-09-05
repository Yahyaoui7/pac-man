.PHONY: install run run-ai run-ai-fast run-ghost-ai run-full-ai debug clean lint lint-strict help

help:
	@echo "Neon Pac-Man Arcade & AI Commands:"
	@echo "  make install       - Install dependencies using uv"
	@echo "  make run           - Launch game in Manual mode"
	@echo "  make run-ai        - Launch with AI Pac-Man (Lookahead Search + Neural Net)"
	@echo "  make run-ai-fast   - Launch with AI Pac-Man (Pure 1-step Neural Net)"
	@echo "  make run-ghost-ai  - Launch with AI Ghosts"
	@echo "  make run-full-ai   - Launch with AI Pac-Man vs AI Ghosts"
	@echo "  make clean         - Clean temporary cache files"
	@echo "  make lint          - Run flake8 and mypy checks"

install:
	uv sync

run:
	uv run python pac_man.py config.json

run-ai:
	uv run python pac_man.py config.json --ai-player

run-ai-fast:
	uv run python pac_man.py config.json --ai-player --no-search

run-ghost-ai:
	uv run python pac_man.py config.json --ai-ghosts

run-full-ai:
	uv run python pac_man.py config.json --ai-player --ai-ghosts

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
