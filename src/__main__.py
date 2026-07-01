# the main start for parsing + starting the game + saving the updates
# main start
# ├── Parse config
# ├── Validate config
# ├── Load highscores
# └── GameStarter(config, highscores).run()

from .game_loop import GameStarter


def main():
    GameStarter.run()


if __name__ == "__main__":
    main()
