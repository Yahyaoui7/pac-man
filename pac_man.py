#!/usr/bin/env python3
"""Main entry point for running Neon Pac-Man with AI Player and Ghost models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from src.game_loop import GameStarter
from src.logic.parsing import Parser


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for launching the game."""
    parser = argparse.ArgumentParser(
        description="Neon Pac-Man Arcade Game with Neural Network AI Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.json",
        help="Path to level configuration JSON file",
    )
    parser.add_argument(
        "--ai-player",
        "--ai",
        dest="ai_player",
        action="store_true",
        help="Launch in AI Player mode (toggle in-game with 'P' or 'A')",
    )
    parser.add_argument(
        "--ai-ghosts",
        action="store_true",
        help=(
            "Enable Ghost Neural Network controller "
            "(falls back to scripted if weights not found)"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to custom Player AI model checkpoint (.pt)",
    )
    parser.add_argument(
        "--ghost-checkpoint",
        type=str,
        default=None,
        help="Path to custom Ghost AI model checkpoint (.pt)",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help=(
            "Disable lookahead tactical search for Pac-Man "
            "(uses pure 1-step neural network)"
        ),
    )
    return parser.parse_args()


def print_banner(args: argparse.Namespace) -> None:
    """Print an arcade startup banner with runtime settings."""
    player_mode = (
        "AI Autopilot (Pure 1-Step Neural Net)"
        if args.ai_player and args.no_search
        else (
            "AI Autopilot (Hybrid Lookahead Search + Neural Net)"
            if args.ai_player
            else "Manual Player (Arrow Keys / WASD)"
        )
    )
    ghost_mode = (
        "Neural Network AI"
        if args.ai_ghosts
        else "Smart Scripted AI (BFS Chase + Runaway)"
    )
    ckpt_label = args.checkpoint or "Default (player_rl_best.pt)"

    print("=" * 66)
    print("           🌟 NEON PAC-MAN ARCADE & AI ENGINE 🌟")
    print("=" * 66)
    print(f" Config File    : {args.config}")
    print(f" Pac-Man Mode   : {player_mode}")
    print(f" Ghosts Mode    : {ghost_mode}")
    print(f" Player Weights : {ckpt_label}")
    print("-" * 66)
    print(" Controls:")
    print("   • Move Pac-Man       : Arrow Keys / WASD")
    print("   • Toggle AI / Manual : Press 'P' or 'A'")
    print("   • Pause Game         : Press 'SPACE'")
    print("   • Cheats (Numpad/Key): [I]nvincible, [S]peed, [G]host Freeze")
    print("=" * 66)


def main() -> int:
    args = parse_arguments()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.", file=sys.stderr)
        return 1

    parser = Parser(str(config_path))
    config = parser.parser_all()

    os.environ.pop("SDL_VIDEODRIVER", None)

    print_banner(args)

    game = GameStarter(config)
    game.use_ai_player = args.ai_player
    game.use_ai_ghosts = args.ai_ghosts
    game.player_ai_no_search = args.no_search
    game.player_ai_checkpoint = args.checkpoint
    game.ghost_ai_checkpoint = args.ghost_checkpoint

    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
