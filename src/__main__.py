# the main start for parsing + starting the game + saving the updates
# main start
# ├── Parse config
# ├── Validate config
# ├── Load highscores
# └── GameStarter(config, highscores).run()

from .game_loop import GameStarter

Config = {
    "lives": 3,
    "points_per_pacgum": 10,
    "points_per_super_pacgum": 50,
    "points_per_ghost": 200,
    "highscore_filename": "highscores.json",
    "levels": [
        {"width": 21, "height": 21, "seed": 42, "level_max_time": 90},
        {"width": 23, "height": 23, "seed": 100, "level_max_time": 90},
        {"width": 25, "height": 25, "seed": 150, "level_max_time": 90},
        {"width": 27, "height": 27, "seed": 200, "level_max_time": 90},
        {"width": 29, "height": 29, "seed": 250, "level_max_time": 90},
        {"width": 31, "height": 31, "seed": 300, "level_max_time": 90},
        {"width": 33, "height": 33, "seed": 350, "level_max_time": 90},
        {"width": 35, "height": 35, "seed": 400, "level_max_time": 90},
        {"width": 37, "height": 37, "seed": 450, "level_max_time": 90},
        {"width": 39, "height": 39, "seed": 500, "level_max_time": 90},
    ],
}


def main():
    game = GameStarter(Config)
    game.run()


if __name__ == "__main__":
    main()
