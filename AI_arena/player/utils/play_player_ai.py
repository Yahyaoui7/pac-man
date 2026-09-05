"""Visual evaluation launcher for Pac-Man Player AI model in real time."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.game_loop import GameStarter
from src.logic.parsing import Parser

DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "../config.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch Pac-Man game with active AI Player model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Path to level configuration JSON file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .pt checkpoint (defaults to latest trained)",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable lookahead beam search and use pure 1-step neural network",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file {config_path} not found.")
        return 1

    parsed = Parser(str(config_path))
    config = parsed.parser_all()

    import os

    os.environ.pop("SDL_VIDEODRIVER", None)

    game = GameStarter(config)
    game.use_ai_player = True
    game.player_ai_no_search = args.no_search
    game.player_ai_checkpoint = args.checkpoint

    mode_str = "Pure 1-Step Neural Network" if args.no_search else "Lookahead Tactical Search Engine + Neural Network"
    print("============================================================")
    print(f"Launching Visual Game with AI Player Model Active")
    print(f"Mode: {mode_str}")
    print("Controls: Press 'P' or 'A' in game to toggle manual vs AI control")
    print("============================================================")
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
