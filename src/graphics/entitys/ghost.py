from typing import Optional
import pygame

from src.graphics.entitys.entity_manager import Entity, facing_from_direction

from src.logic.config import NORTH, SOUTH
from src.logic.helpers import grid_to_pixel, pixel_to_screen

from src.graphics.entitys.graphic_lib import (
    SpriteLibrary,
    Facing,
    GhostColor,
)
from src.graphics.entitys.graphic_lib import GhostState as gs

NAME_TO_GHOST_COLOR = {
    "Blinky": GhostColor.RED,
    "Pinky": GhostColor.PINK,
    "Inky": GhostColor.CYAN,
    "Clyde": GhostColor.ORANGE,
}


class Ghost(Entity):
    def __init__(
        self,
        y: int,
        x: int,
        color: tuple[int, int, int] | pygame.Color,
        name: str,
    ) -> None:
        super().__init__(y, x, speed=1.5)

        self.name = name
        self.color = color
        self.ghost_color = NAME_TO_GHOST_COLOR.get(name, GhostColor.RED)

        self.spawn_x = x
        self.spawn_y = y

        self.is_edible = False
        self.is_eaten = False
        self.frightened_timer = 0.0
        self.respawn_timer = 0.0
        self.runaway_target: Optional[tuple[int, int]] = None

        self.going_to_prison = False
        self.in_prison = False
        self.prison_target = None
        self.prison_cells: Optional[list[tuple[int, int]]] = None

        self.sprites = SpriteLibrary.instance()
        self.state = gs.HUNT
        self._last_vertical: Optional[str] = None
        self.animation = self.sprites.new_ghost_animation(
            gs.HUNT, color=self.ghost_color, facing=self.facing
        )

    def _current_state(self) -> gs:
        if self.is_eaten:
            return gs.EATEN
        if self.is_edible:
            return gs.FRIGHTENED
        return gs.HUNT

    def update_animation(self, dt_ms: float) -> None:
        old_facing = self.facing
        self.facing = facing_from_direction(self.direction, self.facing)

        vertical = None
        if self.direction == NORTH:
            vertical = "up"
        elif self.direction == SOUTH:
            vertical = "down"

        new_state = self._current_state()

        changed = (
            new_state != self.state
            or self.facing != old_facing
            or (new_state == gs.EATEN and vertical != self._last_vertical)
        )

        if changed:
            self.state = new_state
            self._last_vertical = vertical
            self.animation = self.sprites.new_ghost_animation(
                new_state,
                color=self.ghost_color,
                facing=self.facing,
                vertical=vertical,
            )

        self.animation.update(dt_ms)

    def draw(self, screen: pygame.Surface) -> None:
        x, y = pixel_to_screen(int(self.x), int(self.y))

        frame = self.animation.current_frame

        if self.state == gs.EATEN:
            rect = frame.get_rect(center=(int(x) - 8, int(y)))
            screen.blit(frame, rect)

            second_rect = frame.get_rect(center=(int(x) + 8, int(y)))
            screen.blit(frame, second_rect)
        else:
            rect = frame.get_rect(center=(int(x), int(y)))
            screen.blit(frame, rect)

    def reset(self) -> None:
        self.grid_y = self.spawn_y
        self.grid_x = self.spawn_x

        self.x, self.y = grid_to_pixel(self.spawn_y, self.spawn_x)
        self.is_edible = False
        self.is_eaten = False
        self.frightened_timer = 0.0
        self.runaway_target = None

        self.facing = Facing.RIGHT
        self.state = gs.HUNT
        self.animation = self.sprites.new_ghost_animation(
            gs.HUNT, color=self.ghost_color, facing=self.facing
        )
