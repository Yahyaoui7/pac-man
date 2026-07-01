# the main start for parsing + starting the game + saving the updates
# main start
# ├── Parse config
# ├── Validate config
# ├── Load highscores
# └── GameStarter(config, highscores).run()


import sys
from src.logic.parsing import Parser
from src.game_loop import GameStarter


def main() -> int:
    if len(sys.argv) != 2:
        print("Command should be: python3 pac_man.py config.json")
        return 1

    parser = Parser(sys.argv[1])
    config = parser.parser_all()

    game = GameStarter(config)
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
