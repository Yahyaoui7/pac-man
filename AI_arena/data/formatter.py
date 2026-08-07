"""Unified observation formatter for both Pac-Man and Ghost models."""

from __future__ import annotations

from typing import Any

import torch

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    CNN_HEIGHT,
    CNN_WIDTH,
    EXTRA_FEATURE_COUNT,
    GHOST_COUNT,
)
from src.logic.config import EAST, NORTH, SOUTH, WEST

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class ObservationFormatter:
    """Centralized observation builder creating identical tensors for Pac-Man and Ghost models."""

    @staticmethod
    def format_observation(
        maze: list[list[int]],
        pellets: list[list[int]],
        player_pos: tuple[int, int],
        player_direction: str,
        ghost_states: list[dict[str, Any]],
        movement: Any,
        device: torch.device | str = "cpu",
        last_action: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct unified tensors efficiently."""
        device = torch.device(device)
        height = len(maze)
        width = len(maze[0]) if height else 0
        max_dim = max(width, height, 1)

        grid = torch.zeros(
            (1, CNN_CHANNEL_COUNT, CNN_HEIGHT, CNN_WIDTH),
            dtype=torch.float32,
            device=device,
        )

        # Convert maze to tensor for fast wall extraction
        maze_tensor = torch.tensor(maze, dtype=torch.int32, device=device)
        grid[0, 0, :height, :width] = (maze_tensor & NORTH).bool().float()
        grid[0, 1, :height, :width] = (maze_tensor & SOUTH).bool().float()
        grid[0, 2, :height, :width] = (maze_tensor & WEST).bool().float()
        grid[0, 3, :height, :width] = (maze_tensor & EAST).bool().float()
        grid[0, 11, :height, :width] = (maze_tensor != 15).float()

        # Pellets
        pellets_tensor = torch.tensor(pellets, dtype=torch.int32, device=device)
        grid[0, 4, :height, :width] = (pellets_tensor == 1).float()
        grid[0, 5, :height, :width] = (pellets_tensor == 2).float()

        px, py = player_pos
        py = max(0, min(CNN_HEIGHT - 1, py))
        px = max(0, min(CNN_WIDTH - 1, px))
        grid[0, 6, py, px] = 1.0

        for idx in range(min(GHOST_COUNT, len(ghost_states))):
            gst = ghost_states[idx]
            gx, gy = gst["grid_x"], gst["grid_y"]
            gy = max(0, min(CNN_HEIGHT - 1, gy))
            gx = max(0, min(CNN_WIDTH - 1, gx))
            grid[0, 7 + idx, gy, gx] = 1.0

        # Build Extra Features (44 floats)
        player_dir_vec = [float(player_direction == d) for d in DIRECTIONS]
        last_action_vec = [
            float(last_action == a_idx) if last_action is not None else 0.0
            for a_idx in range(ACTION_COUNT)
        ]
        player_powered = float(any(gst.get("is_edible", False) for gst in ghost_states))
        ghost_edible_flags = [
            float(gst.get("is_edible", False)) for gst in ghost_states
        ]

        features = [
            *player_dir_vec,
            *last_action_vec,
            player_powered,
            *ghost_edible_flags,
            float(width) / 50.0,
            float(height) / 25.0,
            float(width * height - 1) / 1000.0,
        ]

        bfs_dist = (
            movement.bfs_distances((py, px))
            if movement is not None
            else [0] * (width * height)
        )

        for gst in ghost_states:
            gx, gy = gst["grid_x"], gst["grid_y"]
            cell_idx = gy * width + gx
            dist = bfs_dist[cell_idx] if (0 <= cell_idx < len(bfs_dist)) else -1
            features.extend(
                [
                    (px - gx) / max_dim,
                    (py - gy) / max_dim,
                    (dist + 1) / max_dim,
                ]
            )

        for gst in ghost_states:
            g_dir = gst.get("direction", "NONE")
            features.extend([float(g_dir == d) for d in DIRECTIONS])

        extra_features = torch.tensor(
            [features],
            dtype=torch.float32,
            device=device,
        )

        # Valid Player Actions (1, 4)
        valid_player_actions = torch.zeros(
            (1, ACTION_COUNT),
            dtype=torch.bool,
            device=device,
        )
        if movement is not None:
            for a_idx, d in enumerate(DIRECTIONS):
                valid_player_actions[0, a_idx] = movement.can_move(py, px, d)

        # Valid Ghost Actions (4, 4)
        valid_ghost_actions = torch.zeros(
            (GHOST_COUNT, ACTION_COUNT),
            dtype=torch.bool,
            device=device,
        )
        if movement is not None:
            for idx in range(min(GHOST_COUNT, len(ghost_states))):
                gst = ghost_states[idx]
                gx, gy = gst["grid_x"], gst["grid_y"]
                for a_idx, d in enumerate(DIRECTIONS):
                    valid_ghost_actions[idx, a_idx] = movement.can_move(gy, gx, d)

        return grid, extra_features, valid_player_actions, valid_ghost_actions
