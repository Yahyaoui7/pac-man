"""Pac-Man-specific observation features for supervised imitation."""

from __future__ import annotations

from typing import Any

import torch

from AI_arena.data.formatter import DIRECTIONS, ObservationFormatter

PLAYER_EXTRA_FEATURE_COUNT = 35
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
    """Return grid [1,12,50,25], player features [1,35], and mask [1,4]."""
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

    player_direction = [float(player.direction == d) for d in DIRECTIONS]
    ghost_directions = [
        float(ghost.direction == direction)
        for ghost in ghosts
        for direction in DIRECTIONS
    ]
    edible = [float(ghost.is_edible) for ghost in ghosts]
    # The game currently owns the shared power timer on Player. Ghost timers
    # are retained when available so this remains compatible with later rules.
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
    power_timer = [
        max(0.0, min(1.0, float(player.powered_timer) / POWER_TIMER_MAX))
    ]

    features = [
        *player_direction,
        *ghost_directions,
        *edible,
        *timers,
        *action_features,
        *remaining,
        *power_timer,
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
