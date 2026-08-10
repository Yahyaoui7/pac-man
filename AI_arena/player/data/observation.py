"""Pac-Man-specific observation features for supervised imitation."""

from __future__ import annotations

from typing import Any

import torch

from AI_arena.data.formatter import DIRECTIONS, ObservationFormatter

PLAYER_EXTRA_FEATURE_COUNT = 45
POWER_TIMER_MAX = 30.0


def format_player_observation(
    *,
    maze: list[list[int]],
    pellets: list[list[int]],
    player: Any,
    ghosts: list[Any],
    movement: Any,
    initial_pellet_count: int | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return grid [1,12,50,25], player features [1,45], and mask [1,4]."""
    ghost_states = [
        {
            "grid_x": ghost.grid_x,
            "grid_y": ghost.grid_y,
            "is_edible": ghost.is_edible,
            "direction": ghost.direction,
        }
        for ghost in ghosts
    ]
    grid, _, valid_actions, _ = ObservationFormatter.format_observation(
        maze=maze,
        pellets=pellets,
        player_pos=(player.grid_x, player.grid_y),
        player_direction=player.direction,
        ghost_states=ghost_states,
        movement=movement,
        device=device,
    )

    height = len(maze)
    width = len(maze[0]) if height else 0
    max_dim = max(width, height, 1)
    px, py = player.grid_x, player.grid_y

    player_direction = [float(player.direction == d) for d in DIRECTIONS]
    ghost_directions = [
        float(ghost.direction == direction)
        for ghost in ghosts
        for direction in DIRECTIONS
    ]
    edible = [float(ghost.is_edible) for ghost in ghosts]
    timers = [
        max(
            0.0,
            min(
                1.0,
                float(
                    getattr(ghost, "frightened_timer", 0.0)
                    or (player.powered_timer if ghost.is_edible else 0.0)
                )
                / POWER_TIMER_MAX,
            ),
        )
        for ghost in ghosts
    ]
    action_features = valid_actions[0].float().tolist()
    normal_remaining = sum(cell == 1 for row in pellets for cell in row)
    power_remaining = sum(cell == 2 for row in pellets for cell in row)
    walkable_count = sum(cell != 15 for row in maze for cell in row)
    denominator = max(initial_pellet_count or walkable_count, 1)
    remaining = [normal_remaining / denominator, power_remaining / denominator]
    power_timer = [max(0.0, min(1.0, float(player.powered_timer) / POWER_TIMER_MAX))]

    # ── NEW: BFS-based spatial features (10 features) ──
    bfs_dist = (
        movement.bfs_distances((py, px))
        if movement is not None
        else [0] * (width * height)
    )

    # Distance to each ghost (4)
    ghost_distances = []
    for ghost in ghosts:
        gx, gy = ghost.grid_x, ghost.grid_y
        cell_idx = gy * width + gx
        dist = bfs_dist[cell_idx] if (0 <= cell_idx < len(bfs_dist)) else -1
        ghost_distances.append((dist + 1) / max_dim)

    # Nearest power pellet distance (1)
    power_pellet_positions = [
        (gy, gx) for gy in range(height) for gx in range(width) if pellets[gy][gx] == 2
    ]
    if power_pellet_positions:
        nearest_pp_dist = min(
            bfs_dist[gy * width + gx] for gy, gx in power_pellet_positions
        )
    else:
        nearest_pp_dist = -1
    nearest_pp_dist_norm = (nearest_pp_dist + 1) / max_dim

    # Nearest normal pellet distance (1)
    normal_pellet_positions = [
        (gy, gx) for gy in range(height) for gx in range(width) if pellets[gy][gx] == 1
    ]
    if normal_pellet_positions:
        nearest_np_dist = min(
            bfs_dist[gy * width + gx] for gy, gx in normal_pellet_positions
        )
    else:
        nearest_np_dist = -1
    nearest_np_dist_norm = (nearest_np_dist + 1) / max_dim

    # Maze size context (3)
    maze_size = [
        float(width) / 50.0,
        float(height) / 25.0,
        float(width * height - 1) / 1000.0,
    ]

    # Player powered boolean (1) — distinct from continuous power_timer
    player_powered_flag = [1.0 if player.powered_timer > 0 else 0.0]

    features = [
        *player_direction,  # 4
        *ghost_directions,  # 16
        *edible,  # 4
        *timers,  # 4
        *action_features,  # 4
        *remaining,  # 2
        *power_timer,  # 1
        *ghost_distances,  # 4  ← NEW
        nearest_pp_dist_norm,  # 1  ← NEW
        nearest_np_dist_norm,  # 1  ← NEW
        *maze_size,  # 3  ← NEW
        *player_powered_flag,  # 1  ← NEW
    ]
    if len(features) != PLAYER_EXTRA_FEATURE_COUNT:
        raise ValueError(
            f"Expected {PLAYER_EXTRA_FEATURE_COUNT} player features, "
            f"got {len(features)}"
        )
    return (
        grid,
        torch.tensor([features], dtype=torch.float32, device=device),
        valid_actions,
    )
