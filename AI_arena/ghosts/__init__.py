"""Ghost AI training and inference package."""

from AI_arena.ghosts.ghost_controller import CNNGhostController
from AI_arena.ghosts.ghost_env import PacmanGhostEnv
from AI_arena.ghosts.ghost_training import train as train_ghost_cnn

__all__ = [
    "PacmanGhostEnv",
    "CNNGhostController",
    "train_ghost_cnn",
]
