"""Parser for the Pac-Man configuration file."""

import json
from typing import Any

from src.logic.config import GameConfig, LevelConfig

DEFAULT_CONFIG: dict[str, Any] = {
    "lives": 3,
    "points_per_pacgum": 10,
    "points_per_super_pacgum": 50,
    "points_per_ghost": 200,
    "highscore_filename": "highscores.json",
    "levels": [{"width": 21, "height": 21, "seed": 42, "level_max_time": 90}],
}


class Parser:
    """Read, clean, validate, and convert the config file."""

    def __init__(self, path: str) -> None:
        """Store the path of the config file."""
        self.path = path

    def parser_all(self) -> GameConfig:
        """Parse the full config file and return a GameConfig object."""
        try:
            with open(self.path, "r", encoding="utf-8") as file:
                data_lines = file.readlines()

            valid_lines = self.remove_ignore_lines(data_lines)
            content = "".join(valid_lines)
            data = json.loads(content)

            if not isinstance(data, dict):
                print(
                    "Config error: config must be a JSON object. Using default config."
                )
                return self.convert_to_config(DEFAULT_CONFIG)

            return self.convert_to_config(data)

        except FileNotFoundError:
            print("Config error: file not found. Using default config.")
            return self.convert_to_config(DEFAULT_CONFIG)

        except json.JSONDecodeError as error:
            print(
                f"Config error: invalid JSON at line {error.lineno}. "
                "Using default config."
            )
            return self.convert_to_config(DEFAULT_CONFIG)

        except OSError:
            print(
                "Config error: cannot read config file. Using default config."
            )
            return self.convert_to_config(DEFAULT_CONFIG)

    def remove_ignore_lines(self, lines: list[str]) -> list[str]:
        """Remove lines that start with #."""
        valid_lines = []

        for line in lines:
            if line.strip().startswith("#"):
                continue
            valid_lines.append(line)

        return valid_lines

    def convert_to_config(self, data: dict[str, Any]) -> GameConfig:
        """Convert dictionary data to a GameConfig object."""
        raw_levels = data.get("levels", DEFAULT_CONFIG["levels"])

        if not isinstance(raw_levels, list):
            print("Config error: levels must be a list. Using default levels.")
            raw_levels = DEFAULT_CONFIG["levels"]

        levels = []

        for raw_level in raw_levels:

            if not isinstance(raw_level, dict):
                print(
                    "Config error: one level is invalid. Using default level."
                )
                raw_level = DEFAULT_CONFIG["levels"][0]

            levels.append(
                LevelConfig(
                    width=self.get_positive_int(raw_level, "width", 21),
                    height=self.get_positive_int(raw_level, "height", 21),
                    seed=self.get_int(raw_level, "seed", 42),
                    level_max_time=self.get_positive_int(
                        raw_level,
                        "level_max_time",
                        90,
                    ),
                )
            )

        if not levels:
            print("Config error: no levels found. Using default level.")
            levels = [
                LevelConfig(width=21, height=21, seed=42, level_max_time=90)
            ]

        return GameConfig(
            lives=self.get_positive_int(data, "lives", 3),
            points_per_pacgum=self.get_positive_int(
                data,
                "points_per_pacgum",
                10,
            ),
            points_per_super_pacgum=self.get_positive_int(
                data,
                "points_per_super_pacgum",
                50,
            ),
            points_per_ghost=self.get_positive_int(
                data,
                "points_per_ghost",
                200,
            ),
            highscore_filename=self.get_string(
                data,
                "highscore_filename",
                "highscores.json",
            ),
            levels=levels,
        )

    def get_int(self, data: dict[str, Any], key: str, default: int) -> int:
        """Return an integer value or a default value."""
        value = data.get(key, default)

        if isinstance(value, int):
            return value

        print(f"Config error: {key} must be an integer. Using {default}.")
        return default

    def get_positive_int(
        self, data: dict[str, Any], key: str, default: int
    ) -> int:
        """Return a positive integer value or a default value."""
        value = data.get(key, default)

        if isinstance(value, int) and value > 0:
            return value

        print(
            f"Config error: {key} must be a positive integer. Using {default}."
        )
        return default

    def get_string(self, data: dict[str, Any], key: str, default: str) -> str:
        """Return a string value or a default value."""
        value = data.get(key, default)

        if isinstance(value, str) and value.strip():
            return value

        print(f"Config error: {key} must be a string. Using {default}.")
        return default
