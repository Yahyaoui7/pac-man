"""Headless Pac-Man environment template for ghost reinforcement learning."""

from __future__ import annotations

import random
from typing import Any

import torch

from AI_arena.data.constants import ACTION_COUNT, GHOST_COUNT
from src.logic.movement import MovementSystem

GHOST_NAMES = ["Blinky", "Pinky", "Inky", "Clyde"]
MIN_MAZE_WIDTH = 10
MAX_MAZE_WIDTH = 25
MIN_MAZE_HEIGHT = 10
MAX_MAZE_HEIGHT = 50

NORTH = 1 << 0
EAST = 1 << 1
SOUTH = 1 << 2
WEST = 1 << 3


class PacmanGhostEnv:
    """Headless environment in which one policy controls four ghosts."""

    def __init__(
        self,
        seed: int | None = None,
        maze_width: int = 20,
        maze_height: int = 25,
        max_steps: int = 2000,
    ) -> None:
        self.max_steps = max_steps
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.step_count = 0
        self.seed = seed
        self.rng = random.Random(seed)

        self.maze: list[list[int]] | None = None
        self.movement: MovementSystem | None = None

    def reset(self) -> dict[str, Any]:
        """Reset episode state and return initial observation."""
        self.step_count = 0
        return {}

    def step(
        self,
        actions: list[int] | torch.Tensor,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Apply actions for ghosts and advance simulation."""
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return {}, 0.0, done, {"step": self.step_count}
