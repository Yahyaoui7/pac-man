"""Risk-aware search teacher that produces Pac-Man imitation labels."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

from AI_arena.data.formatter import DIRECTIONS

DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ACTION_INDEX = {direction: index for index, direction in enumerate(DIRECTIONS)}


@dataclass(frozen=True)
class ExpertDecision:
    action: int
    scores: tuple[float, float, float, float]


class PacmanExpert:
    """Short-horizon search using BFS ghost-arrival maps and food rewards."""

    def __init__(self, horizon: int = 7, safety_margin: int = 0) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.safety_margin = safety_margin

    def choose_action(self, env: Any) -> ExpertDecision:
        if (
            env.movement is None
            or env.player is None
            or env.pellets is None
            or env.maze is None
        ):
            raise RuntimeError("Environment must be reset before expert use")
        movement = env.movement
        player = env.player
        start = (player.grid_y, player.grid_x)
        legal = [d for d in DIRECTIONS if movement.can_move(*start, d)]
        if not legal:
            raise RuntimeError("Pac-Man has no legal action")

        dangerous_maps: list[list[int]] = []
        edible_maps: list[list[int]] = []
        for ghost in env.ghosts:
            distances = movement.bfs_distances((ghost.grid_y, ghost.grid_x))
            target_maps = edible_maps if ghost.is_edible else dangerous_maps
            target_maps.append(distances)

        width = len(env.maze[0])
        pellet_cells = {
            (y, x): value
            for y, row in enumerate(env.pellets)
            for x, value in enumerate(row)
            if value in (1, 2)
        }
        current_direction = player.direction
        distance_cache: dict[tuple[int, int], list[int]] = {}

        def distances_from(cell: tuple[int, int]) -> list[int]:
            if cell not in distance_cache:
                distance_cache[cell] = movement.bfs_distances(cell)
            return distance_cache[cell]

        def distance(maps: list[list[int]], cell: tuple[int, int]) -> int:
            values = [m[cell[0] * width + cell[1]] for m in maps]
            reachable = [value for value in values if value >= 0]
            return min(reachable, default=10**6)

        def search(
            cell: tuple[int, int],
            depth: int,
            eaten: frozenset[tuple[int, int]],
            previous: str,
        ) -> float:
            if depth >= self.horizon:
                remaining = [p for p in pellet_cells if p not in eaten]
                if not remaining:
                    return 100.0
                distances = distances_from(cell)
                nearest = min(
                    (
                        distances[y * width + x]
                        for y, x in remaining
                        if distances[y * width + x] >= 0
                    ),
                    default=50,
                )
                return -0.6 * nearest

            best = -inf
            moves = [d for d in DIRECTIONS if movement.can_move(*cell, d)]
            for direction in moves:
                dy, dx = DELTAS[direction]
                nxt = (cell[0] + dy, cell[1] + dx)
                arrival = depth + 1
                danger_distance = distance(dangerous_maps, nxt)
                if danger_distance <= arrival + self.safety_margin:
                    value = -100000.0 - (self.horizon - depth) * 100.0
                else:
                    exits = sum(movement.can_move(*nxt, d) for d in DIRECTIONS)
                    value = 0.3 * exits
                    if danger_distance < 10**6:
                        safe_distance = min(
                            float(danger_distance - arrival), 6.0
                        )
                        value += safe_distance * 1.5
                    new_eaten = eaten
                    pellet = pellet_cells.get(nxt, 0)
                    if pellet and nxt not in eaten:
                        new_eaten = eaten | {nxt}
                        value += 18.0 if pellet == 1 else (
                            35.0 if dangerous_maps else 12.0
                        )
                    edible_distance = distance(edible_maps, nxt)
                    timer_cells = (
                        float(getattr(player, "powered_timer", 0.0)) / 5.0
                    )
                    if edible_distance <= max(0.0, timer_cells - arrival):
                        value += 7.0 / (edible_distance + 1.0)
                    if exits <= 1 and dangerous_maps:
                        value -= 25.0
                    if direction == previous:
                        value += 0.2
                    value += search(nxt, depth + 1, new_eaten, direction)
                best = max(best, value)
            return best

        scores = [-inf] * len(DIRECTIONS)
        for direction in legal:
            dy, dx = DELTAS[direction]
            nxt = (start[0] + dy, start[1] + dx)
            eaten = frozenset({nxt}) if nxt in pellet_cells else frozenset()
            immediate = 0.2 if direction == current_direction else 0.0
            pellet = pellet_cells.get(nxt, 0)
            if pellet == 1:
                immediate += 18.0
            elif pellet == 2:
                immediate += 35.0 if dangerous_maps else 12.0
            scores[ACTION_INDEX[direction]] = immediate + search(
                nxt, 1, eaten, direction
            )

        # max() is deterministic: ties follow UP, DOWN, LEFT, RIGHT.
        action = max(range(len(scores)), key=scores.__getitem__)
        score_tuple = (scores[0], scores[1], scores[2], scores[3])
        return ExpertDecision(action=action, scores=score_tuple)
