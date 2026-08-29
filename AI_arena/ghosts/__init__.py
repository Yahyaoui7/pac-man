"""Ghost AI training and inference package."""

from AI_arena.ghosts.ghost_controller import CNNGhostController
from AI_arena.ghosts.ghost_env import PacmanGhostEnv

__all__ = [
    "PacmanGhostEnv",
    "CNNGhostController",
]
