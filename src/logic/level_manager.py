import random
from typing import Optional
from mazegenerator import MazeGenerator  # type: ignore
from src.logic.config import GameConfig, LevelConfig


class LevelManager:

    def __init__(self, conf: GameConfig) -> None:
        self.config: GameConfig = conf
        self.current_level_index: int = 0
        self.current_maze: Optional[MazeGenerator] = None
        self.remaining_time: float = 0.0

    def get_current_level_config(self) -> LevelConfig:
        if 0 <= self.current_level_index < len(self.config.levels):
            return self.config.levels[self.current_level_index]
        return self.config.levels[0]

    @staticmethod
    def build_maze(width: int, height: int, seed: int) -> MazeGenerator:
        maze = MazeGenerator(
            size=(width, height),
            perfect=False,
            entry_cell=(0, 0),
            exit_cell=(-1, -1),
            seed=seed,
        )
        return maze

    # TODO: get the old size since this one is just for this trainig stage
    @staticmethod
    def clamp_dimensions(width: int, height: int) -> tuple[int, int]:
        """Force any configured size into the range the game can render."""
        return max(5, min(10, width)), max(5, min(10, height))

    def load_level(self, level_index: int) -> bool:
        if level_index >= len(self.config.levels):
            return False
        self.current_level_index = level_index
        level_conf = self.get_current_level_config()
        try:

            width, height = self.clamp_dimensions(level_conf.width, level_conf.height)

            self.current_maze = self.build_maze(
                width,
                height,
                random.randint(0, 44444),
            )

            # self.current_maze = self.build_maze(
            #     width,
            #     height,
            #     level_conf.seed,
            # )

            self.remaining_time = float(level_conf.level_max_time)
            return True

        except Exception as e:
            print(
                "faile to load the maze, ",
                "falling into the default maze\nError: ",
                e,
            )
            level_conf = self.config.levels[0]
            width, height = self.clamp_dimensions(level_conf.width, level_conf.height)
            self.current_maze = self.build_maze(
                width,
                height,
                level_conf.seed,
            )
            self.remaining_time = float(level_conf.level_max_time)
            return True

    def update_time(self, dt: float) -> None:
        self.remaining_time = max(0.0, self.remaining_time - dt)

    def is_time_out(self) -> bool:
        return self.remaining_time <= 0.0
