#!/usr/bin/env python3
"""Main entry point for Neon Pac-Man arcade game."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from src.game_loop import GameStarter
from src.logic.parsing import Parser


def main() -> int:
    # If running inside a PyInstaller frozen bundle, set working dir to bundle internal root
    if hasattr(sys, "_MEIPASS"):
        os.chdir(sys._MEIPASS)

    exe_dir = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        config_path = sys.argv[1]
    elif (exe_dir / "config.json").exists():
        config_path = str(exe_dir / "config.json")
    elif Path("config.json").exists():
        config_path = "config.json"
    else:
        print("Error: Config file 'config.json' not found.", file=sys.stderr)
        return 1

    parser = Parser(config_path)
    config = parser.parser_all()

    os.environ.pop("SDL_VIDEODRIVER", None)

    game = GameStarter(config)
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
