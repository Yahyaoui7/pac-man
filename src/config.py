"""Classes used to store game configuration."""

from dataclasses import dataclass, field


@dataclass
class LevelConfig:
    """Store one level configuration."""

    width: int = 21
    height: int = 21
    seed: int = 42
    level_max_time: int = 90


@dataclass
class GameConfig:
    """Store full game configuration."""

    lives: int = 3
    points_per_pacgum: int = 10
    points_per_super_pacgum: int = 50
    points_per_ghost: int = 200
    highscore_filename: str = "highscores.json"
    levels: list[LevelConfig] = field(default_factory=list)
