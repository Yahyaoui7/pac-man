#!/usr/bin/env python3
"""Main entry point for Neon Pac-Man arcade game."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from src.game_loop import GameStarter
from src.logic.parsing import Parser


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    if not Path(config_path).exists():
        print(
            f"Error: Config file '{config_path}' not found.",
            file=sys.stderr,
        )
        return 1

    parser = Parser(config_path)
    config = parser.parser_all()

    os.environ.pop("SDL_VIDEODRIVER", None)

    game = GameStarter(config)
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
