"""Pac-Man-specific observation features for supervised imitation."""

from __future__ import annotations

from typing import Any

import torch

from AI_arena.data.formatter import DIRECTIONS, ObservationFormatter

PLAYER_EXTRA_FEATURE_COUNT = 61
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
    visit_counts: list[list[int]] | None = None,  # ← CHANGED
    prev_nearest_pellet_dist: float = -1.0,
    prev_nearest_ghost_dist: float = -1.0,
    prev_nearest_pp_dist: float = -1.0,
    steps_since_pellet: int = 0,
    last_positions: list[tuple[int, int]] = [],
    just_died: float = 0.0,
    same_action_count: int = 0,
    region_completion_frac: float = 0.0,
    region_is_dirty: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return grid [1,7,50,25], player features [1,61], and mask [1,4]."""
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
        visit_counts=visit_counts,  # ← CHANGED
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

    normal_remaining = 0
    power_remaining = 0
    nearest_np_dist = -1
    nearest_pp_dist = -1

    # BFS-based spatial features
    bfs_dist = (
        movement.bfs_distances((py, px))
        if movement is not None
        else [0] * (width * height)
    )

    ghost_distances_raw = []
    ghost_distances = []
    for ghost in ghosts:
        gx, gy = ghost.grid_x, ghost.grid_y
        cell_idx = gy * width + gx
        dist = bfs_dist[cell_idx] if (0 <= cell_idx < len(bfs_dist)) else -1
        ghost_distances_raw.append(dist)
        ghost_distances.append((dist + 1) / max_dim)

    # Single pass over pellets to count and find min distances
    for gy in range(height):
        p_row = pellets[gy]
        for gx in range(width):
            val = p_row[gx]
            if val == 1:
                normal_remaining += 1
                d = bfs_dist[gy * width + gx]
                if d >= 0 and (nearest_np_dist == -1 or d < nearest_np_dist):
                    nearest_np_dist = d
            elif val == 2:
                power_remaining += 1
                d = bfs_dist[gy * width + gx]
                if d >= 0 and (nearest_pp_dist == -1 or d < nearest_pp_dist):
                    nearest_pp_dist = d

    walkable_count = sum(cell != 15 for row in maze for cell in row)
    denominator = max(initial_pellet_count or walkable_count, 1)
    remaining = [normal_remaining / denominator, power_remaining / denominator]
    power_timer = [max(0.0, min(1.0, float(player.powered_timer) / POWER_TIMER_MAX))]

    nearest_pp_dist_norm = (nearest_pp_dist + 1) / max_dim
    nearest_np_dist_norm = (nearest_np_dist + 1) / max_dim

    # Local adjacent pellet features (4)
    local_pellet = [0.0, 0.0, 0.0, 0.0]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dy, dx) in enumerate(dirs):
        ny, nx = py + dy, px + dx
        if 0 <= ny < height and 0 <= nx < width and pellets[ny][nx] in (1, 2):
            local_pellet[i] = 1.0

    # Delta / trend features (3)
    delta_pellet = 0.0
    if prev_nearest_pellet_dist >= 0 and nearest_np_dist >= 0:
        delta_pellet = (prev_nearest_pellet_dist - nearest_np_dist) / max_dim

    delta_ghost = 0.0
    min_ghost_dist = min((d for d in ghost_distances_raw if d >= 0), default=-1)
    if prev_nearest_ghost_dist >= 0 and min_ghost_dist >= 0:
        delta_ghost = (prev_nearest_ghost_dist - min_ghost_dist) / max_dim

    delta_pp = 0.0
    if prev_nearest_pp_dist >= 0 and nearest_pp_dist >= 0:
        delta_pp = (prev_nearest_pp_dist - nearest_pp_dist) / max_dim

    steps_since_pellet_norm = min(steps_since_pellet, 100) / 100.0

    last_offset_y = 0.0
    last_offset_x = 0.0
    second_last_offset_y = 0.0
    second_last_offset_x = 0.0
    if last_positions:
        last_offset_y = (py - last_positions[-1][0]) / max_dim
        last_offset_x = (px - last_positions[-1][1]) / max_dim
    if len(last_positions) > 1:
        second_last_offset_y = (py - last_positions[-2][0]) / max_dim
        second_last_offset_x = (px - last_positions[-2][1]) / max_dim

    just_died_flag = [just_died]
    same_action_norm = min(same_action_count, 20) / 20.0

    maze_size = [
        float(width) / 50.0,
        float(height) / 25.0,
        float(width * height - 1) / 1000.0,
    ]

    player_powered_flag = [1.0 if player.powered_timer > 0 else 0.0]

    features = [
        *player_direction,
        *ghost_directions,
        *edible,
        *timers,
        *action_features,
        *remaining,
        *power_timer,
        *ghost_distances,
        nearest_pp_dist_norm,
        nearest_np_dist_norm,
        *maze_size,
        *player_powered_flag,
        *local_pellet,
        delta_pellet,
        delta_ghost,
        delta_pp,
        steps_since_pellet_norm,
        last_offset_y,
        last_offset_x,
        second_last_offset_y,
        second_last_offset_x,
        region_completion_frac,
        region_is_dirty,
        *just_died_flag,
        same_action_norm,
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
