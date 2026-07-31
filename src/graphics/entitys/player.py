from typing import Optional, Any

import pygame

from src.graphics.entitys.entity import Entity
from src.logic.config import NORTH, EAST, SOUTH, WEST
from src.logic.helpers import grid_to_pixel, pixel_to_screen

from src.graphics.entitys.graphic_lib import (
    SpriteLibrary,
    Facing,
    facing_from_direction,
)
from src.graphics.entitys.graphic_lib import PacmanMode as pm

ABILITY_NONE = "none"
ABILITY_PUNCH = "punch"
ABILITY_KICK = "kick"


class Player(Entity):
    def __init__(self, y: int, x: int) -> None:
        super().__init__(y, x, speed=3.5)

        self.score = 0
        self.msg_txt = ""

        self.sprites = SpriteLibrary.instance()
        self.mode = pm.NORMAL
        self.is_punching = False
        self.is_kicking = False
        self.current_ability: str = ABILITY_NONE
        self.animation = self.sprites.new_animation(pm.NORMAL)

        self.powered_mode: Optional[pm] = None
        self.powered_timer: float = 0.0
        self.is_attacking: bool = False

    def grant_ability(self, ability: str) -> None:
        """Grant an ability to the player (punch, kick, or none)."""
        self.current_ability = ability

    def activate_ability(self, ability: str) -> None:
        """Start powered walk mode (called when eating a powered super gum).
        The walk animation loops for the duration of the fright timer.
        Attack animation triggers on ghost collision."""
        if self.mode != pm.NORMAL:
            return

        if ability == ABILITY_PUNCH:
            self.powered_mode = pm.PUNCH
        elif ability == ABILITY_KICK:
            self.powered_mode = pm.KICK
        else:
            return

        self.mode = self.powered_mode
        self.animation = self.sprites.new_walk_animation(self.powered_mode)

    def trigger_attack(self) -> None:
        """Trigger attack animation on ghost collision during powered mode."""
        if self.powered_mode is None or self.is_attacking:
            return
        self.is_attacking = True
        self.animation = self.sprites.new_attack_animation(self.powered_mode)

    def start_powered_mode(self, mode: pm, duration: float) -> None:
        """Enter powered walk mode for the given duration (fright timer)."""
        self.powered_mode = mode
        self.mode = mode
        self.powered_timer = duration
        self.is_attacking = False
        self.animation = self.sprites.new_walk_animation(mode)

    def end_powered_mode(self) -> None:
        """Exit powered mode and revert to normal."""
        self.powered_mode = None
        self.powered_timer = 0.0
        self.is_attacking = False
        self.is_punching = False
        self.is_kicking = False
        self.mode = pm.NORMAL
        self.animation = self.sprites.new_animation(pm.NORMAL)

    def use_ability(self) -> None:
        """Activate the current ability if available
        and not already in a special mode."""
        if self.mode != pm.NORMAL or self.current_ability == ABILITY_NONE:
            return

        if self.current_ability == ABILITY_PUNCH:
            self.mode = pm.PUNCH
            self.is_punching = True
            self.animation = self.sprites.new_animation(pm.PUNCH)
        elif self.current_ability == ABILITY_KICK:
            self.mode = pm.KICK
            self.is_kicking = True
            self.animation = self.sprites.new_animation(pm.KICK)

        self.current_ability = ABILITY_NONE

    def update_animation(self, dt_ms: float) -> None:
        self.facing = facing_from_direction(self.direction, self.facing)

        self.animation.update(dt_ms)

        if self.is_attacking and self.animation.finished:
            self.is_attacking = False
            assert self.powered_mode is not None
            self.animation = self.sprites.new_walk_animation(self.powered_mode)

        if (
            self.mode != pm.NORMAL
            and self.powered_mode is None
            and self.animation.finished
        ):
            self.is_punching = False
            self.is_kicking = False
            self.mode = pm.NORMAL
            self.animation = self.sprites.new_animation(pm.NORMAL)

    def draw(self, screen: pygame.Surface) -> None:
        x, y = pixel_to_screen(int(self.x), int(self.y))

        frame = self.animation.current_frame
        if self.facing == Facing.LEFT:
            frame = pygame.transform.flip(frame, True, False)

        rect = frame.get_rect(center=(int(x), int(y)))
        screen.blit(frame, rect)

    def is_valid_spawn(
        self,
        row: int,
        col: int,
        maze: list[list[int]],
    ) -> bool:
        cell = maze[row][col]
        return cell != (NORTH | EAST | SOUTH | WEST)

    def find_player_spawn(
        self,
        game: Any | None,
        maze: list[list[int]],
    ) -> bool:
        # The graphical game exposes dimensions through curr_level; the
        # headless RL environment only provides the maze itself.
        if game is None:
            height = len(maze)
            width = len(maze[0])
        else:
            height = game.curr_level.height
            width = game.curr_level.width

        middle_row = height // 2
        middle_col = width // 2

        for radius in range(
            max(width, height)
        ):
            for row in range(middle_row - radius, middle_row + radius + 1):
                for col in range(middle_col - radius, middle_col + radius + 1):
                    if (
                        0 <= row < height
                        and 0 <= col < width
                        and self.is_valid_spawn(row, col, maze)
                    ):
                        self.spawn_x = col
                        self.spawn_y = row
                        self.grid_y = row
                        self.grid_x = col
                        self.x, self.y = grid_to_pixel(row, col)
                        return True

        return False

    def reset_location(self) -> None:
        self.x, self.y = grid_to_pixel(self.spawn_y, self.spawn_x)

        self.direction = None
        self.next_direction = None

        self.row_direction = 0
        self.col_direction = 0

        self.facing = Facing.RIGHT
        self.mode = pm.NORMAL
        self.is_punching = False
        self.is_kicking = False
        self.is_attacking = False
        self.current_ability = ABILITY_NONE
        self.powered_mode = None
        self.powered_timer = 0.0
        self.animation = self.sprites.new_animation(pm.NORMAL)
