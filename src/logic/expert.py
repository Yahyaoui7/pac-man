
"""Risk-aware search teacher that produces Pac-Man imitation labels."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

from AI_arena.data.formatter import DIRECTIONS


DELTAS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

ACTION_INDEX = {
    direction: index
    for index, direction in enumerate(DIRECTIONS)
}


@dataclass(frozen=True)
class ExpertDecision:
    action: int
    scores: tuple[float, float, float, float]


class PacmanExpert:
    """Fast short-horizon teacher for Pac-Man imitation learning.

    The expert:
      - rewards pellets
      - avoids dangerous ghosts
      - avoids dead ends
      - slightly prefers continuing in the same direction
      - searches a few steps into the future
    """

    def __init__(
        self,
        horizon: int = 4,
        safety_margin: int = 3,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")

        self.horizon = horizon
        self.safety_margin = safety_margin

        self.distance_cache: dict[
            tuple[int, int],
            list[int],
        ] = {}

    def choose_action(self, env: Any) -> ExpertDecision:
        """Choose the best Pac-Man action for the current state."""

        if (
            env.movement is None
            or env.player is None
            or env.pellets is None
            or env.maze is None
        ):
            raise RuntimeError(
                "Environment must be reset before expert use"
            )

        movement = env.movement
        player = env.player

        start = (player.grid_y, player.grid_x)

        # Find directions Pac-Man can actually move.
        legal = [
            direction
            for direction in DIRECTIONS
            if movement.can_move(*start, direction)
        ]

        if not legal:
            raise RuntimeError("Pac-Man has no legal action")

        # ---------------------------------------------------------
        # Dangerous ghost BFS maps
        # ---------------------------------------------------------

        dangerous_maps: list[list[int]] = []

        for ghost in env.ghosts:
            if ghost.is_edible:
                continue

            distances = movement.bfs_distances(
                (ghost.grid_y, ghost.grid_x)
            )
            dangerous_maps.append(distances)

        # ---------------------------------------------------------
        # Pellet positions
        # ---------------------------------------------------------

        width = len(env.maze[0])

        pellet_cells = {
            (y, x): value
            for y, row in enumerate(env.pellets)
            for x, value in enumerate(row)
            if value in (1, 2)
        }

        current_direction = player.direction

        # Use the cache initialized in __init__
        def distances_from(
            cell: tuple[int, int],
        ) -> list[int]:
            """Return BFS distances from a cell, using a cache."""

            if cell not in self.distance_cache:
                self.distance_cache[cell] = movement.bfs_distances(cell)

            return self.distance_cache[cell]

        def nearest_ghost_distance(
            cell: tuple[int, int],
        ) -> int:
            """Return the distance to the nearest dangerous ghost."""

            if not dangerous_maps:
                return 10**6

            index = cell[0] * width + cell[1]

            distances = [
                ghost_map[index]
                for ghost_map in dangerous_maps.values()
            ]

            reachable = [
                value
                for value in distances
                if value >= 0
            ]

            return min(reachable, default=10**6)

        def search(
            cell: tuple[int, int],
            depth: int,
            eaten: frozenset[tuple[int, int]],
            previous_direction: str,
        ) -> float:
            """Search future moves and return the best future score."""

            # -----------------------------------------------------
            # Search finished
            # -----------------------------------------------------

            if depth >= self.horizon:
                remaining_pellets = [
                    pellet
                    for pellet in pellet_cells
                    if pellet not in eaten
                ]

                if not remaining_pellets:
                    return 100.0

                distances = distances_from(cell)

                nearest_pellet = min(
                    (
                        distances[y * width + x]
                        for y, x in remaining_pellets
                        if distances[y * width + x] >= 0
                    ),
                    default=50,
                )

                return -0.6 * nearest_pellet

            best_score = -inf

            # Find possible moves from this cell.
            moves = [
                direction
                for direction in DIRECTIONS
                if movement.can_move(*cell, direction)
            ]

            for direction in moves:
                dy, dx = DELTAS[direction]

                next_cell = (
                    cell[0] + dy,
                    cell[1] + dx,
                )

                arrival = depth + 1

                # -------------------------------------------------
                # Ghost safety
                # -------------------------------------------------

                danger_distance = nearest_ghost_distance(
                    next_cell
                )

                if danger_distance <= (
                    arrival + self.safety_margin
                ):
                    value = (
                        -100000.0
                        - (self.horizon - depth) * 100.0
                    )
                else:
                    # -------------------------------------------------
                    # Basic position value
                    # -------------------------------------------------

                    exits = sum(
                        movement.can_move(*next_cell, d)
                        for d in DIRECTIONS
                    )

                    value = 0.3 * exits

                    # Prefer being farther from dangerous ghosts.
                    if danger_distance < 10**6:
                        safe_distance = min(
                            float(danger_distance - arrival),
                            6.0,
                        )
                        value += safe_distance * 1.5

                    # -------------------------------------------------
                    # Pellet reward
                    # -------------------------------------------------

                    new_eaten = eaten

                    pellet = pellet_cells.get(
                        next_cell,
                        0,
                    )

                    if pellet and next_cell not in eaten:
                        new_eaten = eaten | {next_cell}

                        if pellet == 1:
                            value += 18.0
                        else:
                            value += 35.0

                    # -------------------------------------------------
                    # Dead-end penalty
                    # -------------------------------------------------

                    if exits <= 1 and dangerous_maps:
                        value -= 25.0

                    # -------------------------------------------------
                    # Continue current direction
                    # -------------------------------------------------

                    if direction == previous_direction:
                        value += 0.2

                    # -------------------------------------------------
                    # Continue searching
                    # -------------------------------------------------

                    value += search(
                        next_cell,
                        depth + 1,
                        new_eaten,
                        direction,
                    )

                best_score = max(
                    best_score,
                    value,
                )

            return best_score

        # ---------------------------------------------------------
        # Score the first action
        # ---------------------------------------------------------

        scores = [-inf] * len(DIRECTIONS)

        for direction in legal:
            dy, dx = DELTAS[direction]

            next_cell = (
                start[0] + dy,
                start[1] + dx,
            )

            eaten = (
                frozenset({next_cell})
                if next_cell in pellet_cells
                else frozenset()
            )

            # Small bonus for continuing in the same direction.
            immediate = (
                10.0
                if direction == current_direction
                else 0.0
            )

            # Immediate pellet reward.
            pellet = pellet_cells.get(
                next_cell,
                0,
            )

            if pellet == 1:
                immediate += 18.0
            elif pellet == 2:
                immediate += 35.0

            scores[ACTION_INDEX[direction]] = (
                immediate
                + search(
                    next_cell,
                    1,
                    eaten,
                    direction,
                )
            )

        # ---------------------------------------------------------
        # Choose the highest-scoring action
        # ---------------------------------------------------------

        action = max(
            range(len(scores)),
            key=scores.__getitem__,
        )

        score_tuple = (
            scores[0],
            scores[1],
            scores[2],
            scores[3],
        )

        return ExpertDecision(
            action=action,
            scores=score_tuple,
        )

