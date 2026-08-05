.PHONY: install run debug collect train collect-player train-player clean lint lint-strict


run:
	uv run python pac_man.py config.json

install:
	uv sync

debug:
	uv run python -m pdb -m pac_man.py config.json

collect:
	uv run python -m AI_arena.data_collector.main_loop 

train:
	uv run python -m AI_arena.ghosts.ghost_training --epochs 30 --patience 5

collect-player:
	uv run python -m AI_arena.player.player_collector --samples 10000 --stage 2

train-player:
	uv run python -m AI_arena.player.player_training --epochs 20 --batch-size 64

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
