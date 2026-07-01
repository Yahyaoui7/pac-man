import json
from src.config import GameConfig, LevelConfig

DEFAULT_CONFIG = {
    "lives": 1,
    "points_per_pacgum": 10,
    "points_per_super_pacgum": 50,
    "points_per_ghost": 200,
    "highscore_filename": "highscores.json",
    "levels": [{"width": 21, "height": 21, "seed": 42, "level_max_time": 90}],
}


class Parser:
    def __init__(self, path):
        self.path = path

    def parser_all(self):
        try:
            with open(self.path) as file:
                data_lines = file.readlines()

                valid_lines = self.remove_ignore_lines(data_lines)
                convert = "".join(valid_lines)
                data = json.loads(convert)
                return self.convert_to_config(data)
        except json.JSONDecodeError as error:
            print(
                f"Config error: invalid JSON at line {error.lineno}. Using default config."
            )
            return self.convert_to_config(DEFAULT_CONFIG)

    def remove_ignore_lines(self, lines):
        valid_lines = []
        for line in lines:
            if line.strip().startswith("#"):
                continue
            valid_lines.append(line)
        return valid_lines



    def check_keys(self, data)




    def convert_to_config(self, data: dict) -> GameConfig:
        levels = []

        for level in data.get("levels", []):
            levels.append(
                LevelConfig(
                    width=level.get("width", 21),
                    height=level.get("height", 21),
                    seed=level.get("seed", 42),
                    level_max_time=level.get("level_max_time", 90),
                )
            )

        return GameConfig(
            lives=data.get("lives", 3),
            points_per_pacgum=data.get("points_per_pacgum", 10),
            points_per_super_pacgum=data.get("points_per_super_pacgum", 50),
            points_per_ghost=data.get("points_per_ghost", 200),
            highscore_filename=data.get(
                "highscore_filename", "highscores.json"
            ),
            levels=levels,
        )
