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

VISIT_COUNT_NORMALIZE = 10.0  # ← visits >= 10 saturate at 1.0


class ObservationFormatter:
    """Centralized observation builder creating identical tensors for Pac-Man and Ghost models."""

    @staticmethod
    def _paint_entity_patch(
        channel: torch.Tensor, cy: int, cx: int, height: int, width: int
    ) -> None:
        """Paint a 3×3 graduated heat map patch (center=1.0, orthogonal=0.5, diagonal=0.25).

        Natural border clipping is preserved; out-of-bounds cells remain implicit 0s.
        """
        pattern = (
            (0, 0, 1.00),
            (-1, 0, 0.50),
            (1, 0, 0.50),
            (0, -1, 0.50),
            (0, 1, 0.50),
            (-1, -1, 0.25),
            (-1, 1, 0.25),
            (1, -1, 0.25),
            (1, 1, 0.25),
        )
        for dy, dx, val in pattern:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < height and 0 <= nx < width:
                channel[ny, nx] = torch.maximum(
                    channel[ny, nx],
                    torch.tensor(val, device=channel.device, dtype=channel.dtype),
                )

    @staticmethod
    def _paint_signed_ghost_patch(
        channel: torch.Tensor,
        cy: int,
        cx: int,
        height: int,
        width: int,
        is_edible: bool,
    ) -> None:
        """Paint signed 3x3 ghost patch: Positive for non-edible (danger), Negative for edible."""
        sign = -1.0 if is_edible else 1.0
        pattern = (
            (0, 0, 1.00),
            (-1, 0, 0.50),
            (1, 0, 0.50),
            (0, -1, 0.50),
            (0, 1, 0.50),
            (-1, -1, 0.25),
            (-1, 1, 0.25),
            (1, -1, 0.25),
            (1, 1, 0.25),
        )
        for dy, dx, val in pattern:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < height and 0 <= nx < width:
                target = sign * val
                curr = channel[ny, nx].item()
                if is_edible:
                    channel[ny, nx] = min(curr, target)
                else:
                    channel[ny, nx] = max(curr, target)

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
        visit_counts: (
            list[list[int]] | None
        ) = None,
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

        maze_tensor = torch.tensor(maze, dtype=torch.int32, device=device)
        pellets_tensor = torch.tensor(pellets, dtype=torch.int32, device=device)

        # Channel 0: Raw Maze Bitmask Topology (maze[y][x] / 15.0)
        grid[0, 0, :height, :width] = maze_tensor.float() / 15.0
        # Channel 1: Normal pellets
        grid[0, 1, :height, :width] = (pellets_tensor == 1).float()

        # Channel 2: Power pellets
        grid[0, 2, :height, :width] = (pellets_tensor == 2).float()

        # Channel 3: Player position (3×3 positive heat map patch)
        px, py = player_pos
        py = max(0, min(CNN_HEIGHT - 1, py))
        px = max(0, min(CNN_WIDTH - 1, px))
        ObservationFormatter._paint_entity_patch(grid[0, 3], py, px, height, width)

        # Channel 4: Signed Ghost positions (Positive = dangerous, Negative = edible)
        for idx in range(min(GHOST_COUNT, len(ghost_states))):
            gst = ghost_states[idx]
            if gst.get("in_prison", False):
                continue
            gx, gy = gst["grid_x"], gst["grid_y"]
            gy = max(0, min(CNN_HEIGHT - 1, gy))
            gx = max(0, min(CNN_WIDTH - 1, gx))
            is_edible = gst.get("is_edible", False)
            ObservationFormatter._paint_signed_ghost_patch(
                grid[0, 4], gy, gx, height, width, is_edible=is_edible
            )

        # Channel 5: Walkable Path & Active Map Mask (1.0 = Walkable cell inside active maze, 0.0 = Wall or Padded region)
        grid[0, 5, :height, :width] = (maze_tensor != 15).float()

        # Build Extra Features
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

        # Power-pellet distance
        power_pellet_positions = [
            (gy, gx)
            for gy in range(height)
            for gx in range(width)
            if pellets[gy][gx] == 2
        ]
        if power_pellet_positions:
            nearest_pp_dist = min(
                bfs_dist[gy * width + gx] for gy, gx in power_pellet_positions
            )
        else:
            nearest_pp_dist = -1
        features.append((nearest_pp_dist + 1) / max_dim)

        extra_features = torch.tensor(
            [features],
            dtype=torch.float32,
            device=device,
        )

        valid_player_actions = torch.zeros(
            (1, ACTION_COUNT),
            dtype=torch.bool,
            device=device,
        )
        if movement is not None:
            for a_idx, d in enumerate(DIRECTIONS):
                valid_player_actions[0, a_idx] = movement.can_move(py, px, d)

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
