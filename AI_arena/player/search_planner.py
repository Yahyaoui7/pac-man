"""Chess-like Lookahead Search Planner for Pac-Man.

Forward-simulates future candidate move sequences up to `horizon` steps ahead,
evaluating ghost pursuit trajectories, pellet yields, dead-end traps, and
endgame clusters.

Provides:
- `get_best_action()`: Optimal discrete move (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).
- `get_action_distribution()`: Softmax distribution over actions for AlphaZero distillation.
- `get_action_scores()`: Raw score for each legal action.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn.functional as F


class PacmanLookaheadSearch:
    """High-performance lookahead search engine for Pac-Man navigation & survival."""

    DIRECTIONS = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    REVERSE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(
        self,
        env: Any = None,
        maze: list[list[int]] | None = None,
        movement: Any = None,
        horizon: int = 12,
        beam_width: int = 30,
        ghost_speed_ratio: float = 0.70,
    ) -> None:
        self.env = env
        self.movement: Any = (
            movement
            if movement is not None
            else (getattr(env, "movement", None) if env else None)
        )
        self.maze: list[list[int]] = (
            maze
            if maze is not None
            else (getattr(env, "maze", []) if env else [])
        )
        self.h: int = len(self.maze)
        self.w: int = len(self.maze[0]) if self.h > 0 else 0
        self.horizon = horizon
        self.beam_width = beam_width
        self.ghost_speed_ratio = (
            getattr(env, "ghost_speed_ratio", ghost_speed_ratio)
            if env is not None
            else ghost_speed_ratio
        )
        self._cell_to_trap: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
        self._precompute_traps()

    def _precompute_traps(self) -> None:
        if self.env is not None and hasattr(self.env, "_cell_to_trap") and self.env._cell_to_trap:
            self._cell_to_trap = self.env._cell_to_trap
            return
        if not self.maze or not self.movement:
            return
        h, w = self.h, self.w
        dead_ends = [
            (y, x)
            for y in range(h)
            for x in range(w)
            if len(self.movement.get_neighbors(y, x)) == 1
        ]
        cell_to_trap: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
        for de in dead_ends:
            curr = de
            visited = [curr]
            prev = None
            while True:
                nbrs = [n for n in self.movement.get_neighbors(curr[0], curr[1]) if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                deg = len(self.movement.get_neighbors(nxt[0], nxt[1]))
                visited.append(nxt)
                if deg >= 3:
                    junction = nxt
                    for d_idx, c in enumerate(reversed(visited[:-1])):
                        cell_to_trap[c] = (junction, d_idx + 1)
                    break
                prev = curr
                curr = nxt
        self._cell_to_trap = cell_to_trap

    def _compute_pellet_distance_grid(self, pellets: list[list[int]]) -> list[list[int]]:
        from collections import deque
        from src.logic.config import EAST, NORTH, SOUTH, WEST

        h, w = self.h, self.w
        dist = [[-1] * w for _ in range(h)]
        if not self.maze or h == 0 or w == 0:
            return dist
        q: deque[tuple[int, int]] = deque()
        for y in range(h):
            for x in range(w):
                if pellets[y][x] in (1, 2):
                    dist[y][x] = 0
                    q.append((y, x))

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        wall_bits = [NORTH, SOUTH, WEST, EAST]

        while q:
            y, x = q.popleft()
            d = dist[y][x]
            cell = self.maze[y][x]
            for i, (dy, dx) in enumerate(directions):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not (cell & wall_bits[i]):
                    if dist[ny][nx] == -1:
                        dist[ny][nx] = d + 1
                        q.append((ny, nx))
        return dist

    def get_action_scores(
        self,
        player: Any = None,
        ghosts: list[Any] | None = None,
        pellets: list[list[int]] | None = None,
    ) -> dict[int, float]:
        """Evaluate all candidate initial actions using beam lookahead search.

        Returns a dict mapping action index to its evaluated path score.
        """
        curr_player = player if player is not None else (self.env.player if self.env else None)
        curr_ghosts = ghosts if ghosts is not None else (self.env.ghosts if self.env else [])
        curr_pellets = pellets if pellets is not None else (self.env.pellets if self.env else None)

        if curr_player is None or self.movement is None or curr_pellets is None:
            return {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        py, px = curr_player.grid_y, curr_player.grid_x
        powered_timer = float(getattr(curr_player, "powered_timer", 0.0))

        # Extract current active ghost states
        active_ghosts: list[dict[str, Any]] = []
        for g in curr_ghosts:
            if not getattr(g, "in_prison", False):
                active_ghosts.append(
                    {
                        "pos": (g.grid_y, g.grid_x),
                        "edible": bool(getattr(g, "is_edible", False)),
                        "accum": float(getattr(g, "_tick_accumulator", 0.0)),
                        "speed": float(self.ghost_speed_ratio),
                    }
                )

        # Legal actions at current cell
        legal_actions = [
            a
            for a, (dy, dx) in self.ACTION_DELTAS.items()
            if self.movement.can_move(py, px, self.DIRECTIONS[a])
        ]
        if not legal_actions:
            return {0: -1e4, 1: -1e4, 2: -1e4, 3: -1e4}

        if len(legal_actions) == 1:
            scores = {a: -1e4 for a in range(4)}
            scores[legal_actions[0]] = 100.0
            return scores

        # Endgame check: count remaining pellets
        if self.env is not None and hasattr(self.env, "remaining_pellets"):
            rem_pellets = self.env.remaining_pellets
        else:
            rem_pellets = sum(1 for row in curr_pellets for cell in row if cell in (1, 2))
        is_endgame = rem_pellets <= 15

        # Initialize beam with all legal first moves
        beam: list[tuple[float, list[int], tuple[int, int], dict[tuple[int, int], float], list[dict[str, Any]], float]] = []
        action_leaf_scores: dict[int, list[float]] = {a: [] for a in legal_actions}

        for a in legal_actions:
            dy, dx = self.ACTION_DELTAS[a]
            ny, nx = py + dy, px + dx

            sim_ghosts, collided, eaten_g = self._simulate_ghosts(
                active_ghosts, (ny, nx), powered_timer > 0
            )
            if collided:
                action_leaf_scores[a].append(-1000.0)
                continue

            pellet_val = 0.0
            is_super = False
            if curr_pellets[ny][nx] == 1:
                pellet_val = 1.0
            elif curr_pellets[ny][nx] == 2:
                pellet_val = 15.0
                is_super = True

            new_pwr = 45.0 if is_super else max(0.0, powered_timer - 0.8)
            init_score = pellet_val + eaten_g * 100.0
            beam.append(
                (
                    init_score,
                    [a],
                    (ny, nx),
                    {(ny, nx): pellet_val},
                    sim_ghosts,
                    new_pwr,
                )
            )

        if not beam:
            # All immediate actions collide: pick action maximizing ghost distance
            flee_a = self._flee_action(py, px, active_ghosts, legal_actions)
            scores = {a: -1000.0 for a in range(4)}
            scores[flee_a] = 0.0
            return scores

        # Expand search tree up to horizon
        for depth in range(1, self.horizon):
            next_beam = []
            for score, path, pos, visited_p, g_states, pwr in beam:
                last_a = path[-1]
                nbr_actions = [
                    a
                    for a, (dy, dx) in self.ACTION_DELTAS.items()
                    if self.movement.can_move(pos[0], pos[1], self.DIRECTIONS[a])
                ]

                # Anti-oscillation: prune immediate 180-degree reversal unless trapped or threatened
                if (
                    len(nbr_actions) > 1
                    and self.REVERSE_ACTION[last_a] in nbr_actions
                    and pwr <= 0
                ):
                    min_g_d = min(
                        (
                            abs(g["pos"][0] - pos[0]) + abs(g["pos"][1] - pos[1])
                            for g in g_states
                            if not g["edible"] and g["pos"][0] >= 0
                        ),
                        default=999,
                    )
                    if min_g_d > 2:
                        nbr_actions = [
                            a for a in nbr_actions if a != self.REVERSE_ACTION[last_a]
                        ]

                for a in nbr_actions:
                    dy, dx = self.ACTION_DELTAS[a]
                    ny, nx = pos[0] + dy, pos[1] + dx

                    sim_g, collided, eaten_g = self._simulate_ghosts(
                        g_states, (ny, nx), pwr > 0
                    )
                    if collided:
                        continue

                    pellet_val = 0.0
                    is_super = False
                    if (ny, nx) not in visited_p:
                        if curr_pellets[ny][nx] == 1:
                            pellet_val = 1.0
                        elif curr_pellets[ny][nx] == 2:
                            pellet_val = 15.0
                            is_super = True

                    new_pwr = 45.0 if is_super else max(0.0, pwr - 0.8)
                    new_visited = dict(visited_p)
                    new_visited[(ny, nx)] = pellet_val

                    # Safety distance to nearest non-edible ghost
                    min_g_d = min(
                        (
                            abs(g["pos"][0] - ny) + abs(g["pos"][1] - nx)
                            for g in sim_g
                            if not g["edible"] and g["pos"][0] >= 0
                        ),
                        default=999,
                    )

                    # Dead-end trap penalty
                    dead_end_penalty = 0.0
                    if pwr <= 0 and (ny, nx) in self._cell_to_trap:
                        junc, dist_to_j = self._cell_to_trap[(ny, nx)]
                        if min_g_d <= dist_to_j * 1.5 + 2:
                            dead_end_penalty = -100.0

                    safety_bonus = min(min_g_d, 6) * 0.3 + dead_end_penalty
                    step_score = score + pellet_val + eaten_g * 100.0 + safety_bonus

                    next_beam.append(
                        (
                            step_score,
                            path + [a],
                            (ny, nx),
                            new_visited,
                            sim_g,
                            new_pwr,
                        )
                    )

            if not next_beam:
                break

            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[: self.beam_width]

        # Evaluate leaf states
        if self.env is not None and hasattr(self.env, "_pellet_dist_grid") and self.env._pellet_dist_grid:
            pellet_dist_grid = self.env._pellet_dist_grid
        else:
            pellet_dist_grid = self._compute_pellet_distance_grid(curr_pellets)

        for score, path, final_pos, visited_p, g_states, pwr in beam:
            first_action = path[0]
            final_eval = score

            # Pellet distance potential
            if pellet_dist_grid is not None:
                p_dist = pellet_dist_grid[final_pos[0]][final_pos[1]]
                if p_dist >= 0:
                    weight = 0.25 if is_endgame else 0.1
                    final_eval += (50.0 - p_dist) * weight

            # Exit mobility bonus
            exits = sum(
                1
                for a, (dy, dx) in self.ACTION_DELTAS.items()
                if self.movement.can_move(
                    final_pos[0], final_pos[1], self.DIRECTIONS[a]
                )
            )
            final_eval += exits * 0.5

            action_leaf_scores[first_action].append(final_eval)

        # Aggregate scores per candidate initial action
        final_scores = {a: -1e4 for a in range(4)}
        for a in legal_actions:
            scores_a = action_leaf_scores[a]
            if scores_a:
                # Use 90th percentile / max score for that initial action branch
                final_scores[a] = float(np.max(scores_a))

        return final_scores

    def get_best_action(
        self,
        player: Any = None,
        ghosts: list[Any] | None = None,
        pellets: list[list[int]] | None = None,
    ) -> int:
        """Return the single best discrete action chosen by lookahead search."""
        scores = self.get_action_scores(player=player, ghosts=ghosts, pellets=pellets)
        return int(max(scores, key=lambda a: scores[a]))

    def get_action_distribution(
        self,
        player: Any = None,
        ghosts: list[Any] | None = None,
        pellets: list[list[int]] | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return a softmax distribution over actions for policy distillation."""
        scores = self.get_action_scores(player=player, ghosts=ghosts, pellets=pellets)
        raw_vals = [scores.get(a, -1e4) for a in range(4)]
        t_vals = torch.tensor(raw_vals, dtype=torch.float32)
        valid_mask = t_vals > -500.0
        if not valid_mask.any():
            return torch.full((4,), 0.25, dtype=torch.float32)

        # Scale non-masked values by temperature
        t_vals[~valid_mask] = -1e4
        probs = F.softmax(t_vals / max(0.01, temperature), dim=-1)
        return probs

    def _simulate_ghosts(
        self,
        ghosts: list[dict[str, Any]],
        pac_pos: tuple[int, int],
        powered: bool,
    ) -> tuple[list[dict[str, Any]], bool, int]:
        """Simulate ghost positions 1 step ahead given Pac-Man's target cell."""
        new_ghosts: list[dict[str, Any]] = []
        collided = False
        eaten_count = 0
        if self.movement is None:
            return ghosts, False, 0
        w = self.w

        target_dists = self.movement.bfs_distances(pac_pos)

        for g in ghosts:
            gy, gx = g["pos"]
            if gy < 0 or gx < 0 or gy >= self.h or gx >= self.w:
                new_ghosts.append(g)
                continue

            edible = g["edible"] or powered
            accum = g["accum"] + g["speed"]
            new_gy, new_gx = gy, gx

            if accum >= 1.0:
                accum -= 1.0
                nbrs = self.movement.get_neighbors(gy, gx)
                if nbrs:
                    if not edible:
                        # Chase Pac-Man: step to neighbor closest to pac_pos
                        best_nbr = min(nbrs, key=lambda n: target_dists[n[0] * w + n[1]])
                        new_gy, new_gx = best_nbr
                    else:
                        # Flee Pac-Man: step to neighbor farthest from pac_pos
                        best_nbr = max(nbrs, key=lambda n: target_dists[n[0] * w + n[1]])
                        new_gy, new_gx = best_nbr

            # Collision check
            if (new_gy, new_gx) == pac_pos or (gy, gx) == pac_pos:
                if edible:
                    eaten_count += 1
                    new_gy, new_gx = (-1, -1)
                    edible = False
                else:
                    collided = True

            new_ghosts.append(
                {
                    "pos": (new_gy, new_gx),
                    "edible": edible,
                    "accum": accum,
                    "speed": g["speed"],
                }
            )

        return new_ghosts, collided, eaten_count

    def _flee_action(
        self,
        py: int,
        px: int,
        ghosts: list[dict[str, Any]],
        legal_actions: list[int],
    ) -> int:
        """Emergency action: select legal move maximizing immediate distance to non-edible ghosts."""
        best_a = legal_actions[0]
        max_min_d = -1
        for a in legal_actions:
            dy, dx = self.ACTION_DELTAS[a]
            ny, nx = py + dy, px + dx
            min_d = min(
                (
                    abs(g["pos"][0] - ny) + abs(g["pos"][1] - nx)
                    for g in ghosts
                    if not g["edible"] and g["pos"][0] >= 0
                ),
                default=999,
            )
            if min_d > max_min_d:
                max_min_d = min_d
                best_a = a
        return best_a
