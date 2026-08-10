"""Factory for creating and respawning Player and Ghost entities."""

from __future__ import annotations

from src.graphics.entitys.ghost import Ghost
from src.graphics.entitys.player import Player
from src.logic.helpers import grid_to_pixel


class EntityFactory:
    """Handles creation and respawning of Player and Ghost entities."""

    @staticmethod
    def create_player(maze: list[list[int]]) -> Player:
        height = len(maze)
        width = len(maze[0])

        center_y = height // 2
        center_x = width // 2
        player = Player(center_y, center_x)

        if not player.is_valid_spawn(center_y, center_x, maze):
            if not player.find_player_spawn(None, maze):
                raise RuntimeError("Could not find valid player spawn.")

        player.powered_mode = None
        player.powered_timer = 0.0
        return player

    @staticmethod
    def create_ghosts(
        maze: list[list[int]],
        specs: list[tuple[str, tuple[int, int, int]]],
    ) -> list[Ghost]:
        height = len(maze)
        width = len(maze[0])
        ghost_cells = [
            (0, 0),
            (0, width - 1),
            (height - 1, 0),
            (height - 1, width - 1),
        ]

        ghosts: list[Ghost] = []
        for (y, x), (name, color) in zip(ghost_cells, specs):
            ghost = Ghost(y, x, color, name)
            ghost.reset()
            ghost._tick_accumulator = 0.0
            ghosts.append(ghost)
        return ghosts

    @staticmethod
    def respawn_player(player: Player, maze: list[list[int]]) -> None:
        """Respawn player at nearest walkable cell from maze center."""
        h, w = len(maze), len(maze[0])
        cy, cx = h // 2, w // 2

        for radius in range(max(w, h)):
            for ry in range(cy - radius, cy + radius + 1):
                for rx in range(cx - radius, cx + radius + 1):
                    if 0 <= ry < h and 0 <= rx < w and maze[ry][rx] != 15:
                        px, py = grid_to_pixel(ry, rx)
                        player.grid_y = ry
                        player.grid_x = rx
                        player.x = float(px)
                        player.y = float(py)
                        player.direction = None
                        player.next_direction = None
                        player.end_powered_mode()
                        return
