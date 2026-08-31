"""BFS-based oracle that produces optimal action labels for all four ghosts.

Two behaviours, mirroring GhostController:
  - Ghost NOT edible (hunting):  move toward the player  → shortest BFS path
  - Ghost IS edible (frightened): move away from player  → direction that
    maximises BFS distance from the player
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Direction order must match ACTION_COUNT index used everywhere in the project
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ACTION_INDEX = {d: i for i, d in enumerate(DIRECTIONS)}


@dataclass(frozen=True)
class GhostExpertDecision:
    """One label per ghost, plus per-ghost scores for diagnostics."""

    labels: tuple[int, ...]          # length == number of ghosts
    scores: tuple[tuple[float, ...], ...]  # shape (n_ghosts, 4)


class GhostExpert:
    """Compute BFS-optimal action labels for every non-imprisoned ghost.

    For a ghost that is NOT edible: choose the first step of the shortest
    BFS path toward the player (chase).

    For a ghost that IS edible: choose the legal move that maximises the
    BFS distance from the player (flee). Ties are broken by DIRECTIONS order.
    """

    # Fallback label when no legal move exists (should never happen on a
    # valid maze, but we must return a valid index).
    _FALLBACK_LABEL = 0

    def choose_actions(self, env: Any) -> GhostExpertDecision:
        """Return optimal labels for all ghosts given the current env state."""
        if env.movement is None or env.player is None or env.maze is None:
            raise RuntimeError("Environment must be reset before expert use.")

        movement = env.movement
        width = len(env.maze[0])
        player_cell = (env.player.grid_y, env.player.grid_x)

        # BFS distances from the player — used for both hunting and fleeing
        player_dists = movement.bfs_distances(player_cell)

        labels: list[int] = []
        all_scores: list[tuple[float, ...]] = []

        for ghost in env.ghosts:
            label, scores = self._label_for_ghost(
                ghost, movement, player_cell, player_dists, width
            )
            labels.append(label)
            all_scores.append(scores)

        return GhostExpertDecision(
            labels=tuple(labels),
            scores=tuple(all_scores),
        )

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _label_for_ghost(
        self,
        ghost: Any,
        movement: Any,
        player_cell: tuple[int, int],
        player_dists: list[int],
        width: int,
    ) -> tuple[int, tuple[float, ...]]:
        """Return (label, scores) for a single ghost.

        Scores are the BFS distance to use for that action (higher = better
        for edible, lower = better for hunting). Illegal moves get -inf.
        """
        ghost_cell = (ghost.grid_y, ghost.grid_x)

        # Collect legal moves
        legal = [d for d in DIRECTIONS if movement.can_move(*ghost_cell, d)]

        # No legal moves (shouldn't happen on a normal maze)
        if not legal:
            return self._FALLBACK_LABEL, (-float("inf"),) * 4

        if ghost.in_prison:
            # Prison ghosts are skipped during labelling — label is still
            # recorded so the index aligns, but it will be masked during
            # training (valid_ghost_actions[idx] will be all-False).
            return self._FALLBACK_LABEL, (-float("inf"),) * 4

        if ghost.is_edible:
            return self._flee_label(ghost_cell, legal, player_dists, width)
        else:
            return self._hunt_label(ghost_cell, legal, player_dists, width)

    def _hunt_label(
        self,
        ghost_cell: tuple[int, int],
        legal: list[str],
        player_dists: list[int],
        width: int,
    ) -> tuple[int, tuple[float, ...]]:
        """Choose the move that minimises BFS distance to the player."""
        scores = [-float("inf")] * 4
        for direction in legal:
            dy, dx = DELTAS[direction]
            ny, nx = ghost_cell[0] + dy, ghost_cell[1] + dx
            idx = ny * width + nx
            dist = player_dists[idx] if 0 <= idx < len(player_dists) else -1
            if dist >= 0:
                # Negate so that the best (smallest) distance gives the
                # highest score — consistent with argmax selection below.
                scores[ACTION_INDEX[direction]] = -float(dist)

        best = max(range(4), key=scores.__getitem__)
        return best, tuple(scores)

    def _flee_label(
        self,
        ghost_cell: tuple[int, int],
        legal: list[str],
        player_dists: list[int],
        width: int,
    ) -> tuple[int, tuple[float, ...]]:
        """Choose the move that maximises BFS distance from the player."""
        scores = [-float("inf")] * 4
        for direction in legal:
            dy, dx = DELTAS[direction]
            ny, nx = ghost_cell[0] + dy, ghost_cell[1] + dx
            idx = ny * width + nx
            dist = player_dists[idx] if 0 <= idx < len(player_dists) else -1
            if dist >= 0:
                scores[ACTION_INDEX[direction]] = float(dist)

        best = max(range(4), key=scores.__getitem__)
        return best, tuple(scores)
