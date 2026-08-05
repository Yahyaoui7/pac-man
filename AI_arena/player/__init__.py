"""Pac-Man player supervised imitation-learning package."""

from AI_arena.player.player_controller import CNNPlayerController
from AI_arena.player.player_env import PacmanPlayerEnv

__all__ = [
    "PacmanPlayerEnv",
    "CNNPlayerController",
]
