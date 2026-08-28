"""Pac-Man-specific observation features for supervised imitation."""

from __future__ import annotations

from typing import Any

import torch

from AI_arena.data.formatter import DIRECTIONS, ObservationFormatter

PLAYER_EXTRA_FEATURE_COUNT = 50
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
    visit_counts: list[list[int]] | None = None,
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
    """Return grid [1, 5, 25, 50], player features [1, 50], and mask [1, 4]."""
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
        visit_counts=visit_counts,
    )

    height = len(maze)
    width = len(maze[0]) if height else 0
    norm_denom = float(width + height) if (width + height) > 0 else 1.0
    px, py = player.grid_x, player.grid_y

    # 1. Player Direction (4: UP, DOWN, LEFT, RIGHT)
    player_direction = [float(player.direction == d) for d in DIRECTIONS]

    # 2. Ghost Edible Flags (4: -1.0 = dangerous non-edible, +1.0 = edible)
    edible = [1.0 if getattr(ghost, "is_edible", False) else -1.0 for ghost in ghosts]

    # 3. Ghost Timers (4: normalized [0, 1] if edible, -1.0 if not edible)
    timers = []
    for ghost in ghosts:
        is_edible = getattr(ghost, "is_edible", False)
        if is_edible:
            t = float(
                getattr(ghost, "frightened_timer", 0.0)
                or (player.powered_timer if is_edible else 0.0)
            )
            timers.append(max(0.0, min(1.0, t / POWER_TIMER_MAX)))
        else:
            timers.append(-1.0)

    # 4. Valid Actions Mask (4: UP, DOWN, LEFT, RIGHT)
    action_features = valid_actions[0].float().tolist()

    # 5. Pellets Remaining Fractions (2: normal, power)
    normal_remaining = 0
    power_remaining = 0
    nearest_np_dist = -1
    nearest_pp_dist = -1

    # BFS spatial distances
    bfs_dist = (
        movement.bfs_distances((py, px))
        if movement is not None
        else [0] * (width * height)
    )

    # 6. Ghost BFS Distances (4: normalized by width+height, capped at 1.0, or -1.0 if unreachable)
    ghost_distances_raw = []
    ghost_distances = []
    for ghost in ghosts:
        gx, gy = ghost.grid_x, ghost.grid_y
        cell_idx = gy * width + gx
        dist = bfs_dist[cell_idx] if (0 <= cell_idx < len(bfs_dist)) else -1
        ghost_distances_raw.append(dist)
        if dist >= 0:
            ghost_distances.append(min(1.0, (dist + 1) / norm_denom))
        else:
            ghost_distances.append(-1.0)

    # Pellets single pass
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

    # 7. Nearest Pellets Distances (2: power pellet, normal pellet)
    nearest_pp_dist_norm = (
        min(1.0, (nearest_pp_dist + 1) / norm_denom) if nearest_pp_dist >= 0 else -1.0
    )
    nearest_np_dist_norm = (
        min(1.0, (nearest_np_dist + 1) / norm_denom) if nearest_np_dist >= 0 else -1.0
    )

    # 8. Local Adjacent Pellets (4: UP, DOWN, LEFT, RIGHT)
    local_pellet = [0.0, 0.0, 0.0, 0.0]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i, (dy, dx) in enumerate(dirs):
        ny, nx = py + dy, px + dx
        if 0 <= ny < height and 0 <= nx < width and pellets[ny][nx] in (1, 2):
            local_pellet[i] = 1.0

    # 9. Local Adjacent Danger (4: UP, DOWN, LEFT, RIGHT)
    local_danger = [0.0, 0.0, 0.0, 0.0]
    if player.powered_timer <= 0:
        active_ghost_cells = [
            (g.grid_y, g.grid_x)
            for g in ghosts
            if not getattr(g, "in_prison", False) and not getattr(g, "is_edible", False)
        ]
        for i, (dy, dx) in enumerate(dirs):
            ny, nx = py + dy, px + dx
            if (ny, nx) in active_ghost_cells:
                local_danger[i] = 1.0
            elif movement is not None and movement.can_move(py, px, DIRECTIONS[i]):
                for gy, gx in active_ghost_cells:
                    if abs(gy - ny) + abs(gx - nx) <= 1:
                        local_danger[i] = 1.0
                        break

    # 10. Distance Deltas (3: pellet, ghost, pp)
    delta_pellet = 0.0
    if prev_nearest_pellet_dist >= 0 and nearest_np_dist >= 0:
        delta_pellet = (prev_nearest_pellet_dist - nearest_np_dist) / norm_denom

    delta_ghost = 0.0
    min_ghost_dist = min((d for d in ghost_distances_raw if d >= 0), default=-1)
    if prev_nearest_ghost_dist >= 0 and min_ghost_dist >= 0:
        delta_ghost = (prev_nearest_ghost_dist - min_ghost_dist) / norm_denom

    delta_pp = 0.0
    if prev_nearest_pp_dist >= 0 and nearest_pp_dist >= 0:
        delta_pp = (prev_nearest_pp_dist - nearest_pp_dist) / norm_denom

    # 11. Step & Action Memory (2)
    steps_since_pellet_norm = min(steps_since_pellet, 100) / 100.0
    same_action_norm = min(same_action_count, 20) / 20.0

    # 12. NEW: Ghost Relative Coordinates (8: 4 ghosts x 2 axes [dx, dy] in [-1.0, 1.0])
    ghost_rel_coords = []
    for g in ghosts:
        if not getattr(g, "in_prison", False):
            dx_norm = (g.grid_x - px) / float(max(width, 1))
            dy_norm = (g.grid_y - py) / float(max(height, 1))
            ghost_rel_coords.extend([max(-1.0, min(1.0, dx_norm)), max(-1.0, min(1.0, dy_norm))])
        else:
            ghost_rel_coords.extend([-1.0, -1.0])

    # 13. NEW: Surrounded Danger Counters (2: within Manhattan dist 3 and 5)
    active_dangerous_ghosts = [
        g for g in ghosts
        if not getattr(g, "in_prison", False) and not getattr(g, "is_edible", False)
    ]
    ghost_count_d3 = sum(
        1 for g in active_dangerous_ghosts
        if (abs(g.grid_x - px) + abs(g.grid_y - py)) <= 3
    )
    ghost_count_d5 = sum(
        1 for g in active_dangerous_ghosts
        if (abs(g.grid_x - px) + abs(g.grid_y - py)) <= 5
    )
    surrounded_d3_norm = ghost_count_d3 / 4.0
    surrounded_d5_norm = ghost_count_d5 / 4.0

    # 14. NEW: Topology Flags (2: dead-end flag, junction flag)
    num_valid_actions = sum(1 for v in action_features if v > 0.5)
    dead_end_flag = 1.0 if num_valid_actions == 1 else 0.0
    junction_flag = 1.0 if num_valid_actions >= 3 else 0.0

    features = [
        *player_direction,         # 4
        *edible,                   # 4
        *timers,                   # 4
        *action_features,          # 4
        *remaining,                # 2
        *power_timer,              # 1
        *ghost_distances,          # 4
        nearest_pp_dist_norm,      # 1
        nearest_np_dist_norm,      # 1
        *local_pellet,             # 4
        *local_danger,             # 4
        delta_pellet,              # 1
        delta_ghost,               # 1
        delta_pp,                  # 1
        steps_since_pellet_norm,   # 1
        same_action_norm,          # 1
        *ghost_rel_coords,         # 8
        surrounded_d3_norm,        # 1
        surrounded_d5_norm,        # 1
        dead_end_flag,             # 1
        junction_flag,             # 1
    ]

    if len(features) != PLAYER_EXTRA_FEATURE_COUNT:
        raise ValueError(
            f"Expected {PLAYER_EXTRA_FEATURE_COUNT} player features, got {len(features)}"
        )
    return (
        grid,
        torch.tensor([features], dtype=torch.float32, device=device),
        valid_actions,
    )
