# Main entry point for running Pac-Man game with optional AI Player mode.

import sys
from src.game_loop import GameStarter
from src.logic.parsing import Parser


def main() -> int:
    if len(sys.argv) < 2:
        print("Command: python3 pac_man.py config.json [--ai-player]")
        return 1

    config_path = sys.argv[1]
    use_ai_player = "--ai-player" in sys.argv or "--ai" in sys.argv

    parser = Parser(config_path)
    config = parser.parser_all()

    import os
    os.environ.pop("SDL_VIDEODRIVER", None)

    game = GameStarter(config)
    game.use_ai_player = use_ai_player
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
