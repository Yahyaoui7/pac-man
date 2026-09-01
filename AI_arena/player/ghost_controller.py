"""Ghost AI and movement controller."""

from __future__ import annotations

from AI_arena.player.constants import DIRECTIONS, GHOST_RESPAWN_TICKS


class GhostController:
    """Encapsulates ghost update logic: respawn, frightened, confused, BFS."""

    def __init__(self, movement, rng, ghost_speed_ratio: float = 0.35) -> None:
        self.movement = movement
        self.rng = rng
        self.ghost_speed_ratio = ghost_speed_ratio

    def update(
        self,
        *,
        ghosts: list,
        player,
        stage: int,
        ghost_respawn_ticks: list[int],
        ghost_confusion_prob: float = 0.0,
    ) -> None:
        """Update all ghosts for one physics tick."""
        if stage == 1:
            for ghost in ghosts:
                ghost.in_prison = True
                ghost.is_edible = False
            return

        for idx, ghost in enumerate(ghosts):
            if ghost.in_prison:
                self._handle_respawn(ghost, idx, ghost_respawn_ticks)
            elif ghost.is_edible:
                self._update_frightened(ghost, player)
            else:
                self._update_hunting(ghost, player, ghost_confusion_prob)

    def _handle_respawn(self, ghost, idx: int, ghost_respawn_ticks: list[int]) -> None:
        if ghost_respawn_ticks[idx] > 0:
            ghost_respawn_ticks[idx] -= 1
        else:
            ghost.in_prison = False
            ghost.is_edible = False
            ghost.runaway_target = None

    def _update_frightened(self, ghost, player) -> None:
        frightened_speed = min(0.5, self.ghost_speed_ratio)
        ghost._tick_accumulator += frightened_speed
        if ghost._tick_accumulator >= 1.0:
            ghost._tick_accumulator -= 1.0
            self.movement.update_runaway_ghost(ghost, player)

    def _update_hunting(self, ghost, player, ghost_confusion_prob: float) -> None:
        ghost._tick_accumulator += self.ghost_speed_ratio
        if ghost._tick_accumulator >= 1.0:
            ghost._tick_accumulator -= 1.0

            if ghost_confusion_prob > 0.0 and self.rng.random() < ghost_confusion_prob:
                valid_dirs = [
                    d
                    for d in DIRECTIONS
                    if self.movement.can_move(ghost.grid_y, ghost.grid_x, d)
                ]
                if len(valid_dirs) > 1:
                    current_idx = (
                        DIRECTIONS.index(ghost.direction)
                        if ghost.direction in DIRECTIONS
                        else -1
                    )
                    rev_dir = (
                        DIRECTIONS[self._reverse_action(current_idx)]
                        if current_idx >= 0
                        else None
                    )
                    candidates = [d for d in valid_dirs if d != rev_dir]
                    chosen = self.rng.choice(candidates if candidates else valid_dirs)
                elif valid_dirs:
                    chosen = valid_dirs[0]
                else:
                    chosen = ghost.direction

                ghost.next_direction = chosen
                self.movement.update_entity(ghost)
            else:
                self.movement.update_bfs_ghost(ghost, player)

    @staticmethod
    def _reverse_action(action: int) -> int:
        return {0: 1, 1: 0, 2: 3, 3: 2}[action]
