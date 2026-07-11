import math
import random
from typing import Optional

import pygame

from src.logic.config import CELL_SIZE, GameConfig
from src.logic.config import NORTH, EAST, SOUTH, WEST
from src.logic.helpers import grid_to_pixel, pixel_to_screen, pellet_screen_pos
from src.graphics.ui_helpers import COLOR_PELLET
from src.sounds.soud_manager import SoundManager

from src.graphics.graphic_lib import (
    SpriteLibrary,
    PacmanMode,
    Facing,
    GhostColor,
    GhostState,
)

NAME_TO_GHOST_COLOR = {
    "Blinky": GhostColor.RED,
    "Pinky": GhostColor.PINK,
    "Inky": GhostColor.CYAN,
    "Clyde": GhostColor.ORANGE,
}

# Ability types that can be granted by super gums
ABILITY_NONE = "none"
ABILITY_PUNCH = "punch"
ABILITY_KICK = "kick"


class EntityManager:
    def __init__(self, config: GameConfig, score_management) -> None:
        """Initialize systems and settings from configuration."""
        self.config: GameConfig = config
        self.score_management = score_management
        self.player: Optional[Player] = None
        self.ghosts: list[Ghost] = []
        self.pellets: list[list[int]] = []
        self.total_pellets: int = 0
        self.sound = SoundManager()
        self.super_gum_abilities: dict[tuple[int, int], str] = {}

        # Sprite frames are loaded once here and shared by every entity.
        self.sprites = SpriteLibrary.instance()
        self.sprites.load(CELL_SIZE)
        self.sprites.load_ghosts(CELL_SIZE)

    def _init_pellet_grid(self, maze: list[list[int]]) -> None:
        """Initialize pellet grid, super-gum abilities, and counts from maze."""
        height, width = len(maze), len(maze[0])
        self.pellets = [[0] * width for _ in range(height)]
        self.total_pellets = 0
        self.super_gum_abilities = {}

        corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
        abilities = random.sample([ABILITY_PUNCH, ABILITY_KICK], 2)
        self.super_gum_abilities[corners[0]] = abilities[0]
        self.super_gum_abilities[corners[1]] = abilities[1]

        center = (width // 2, height // 2)
        for y in range(height):
            for x in range(width):
                if (x, y) in corners:
                    self.pellets[y][x] = 2
                    self.total_pellets += 1
                elif (x, y) == center or maze[y][x] == 15:
                    self.pellets[y][x] = 0
                else:
                    self.pellets[y][x] = 1
                    self.total_pellets += 1

    def load_level_entities(self, maze: list[list[int]]) -> None:
        """Setup maze grid, pellets, and spawn entities."""
        self.maze = maze
        self._init_pellet_grid(maze)

        height = len(maze)
        width = len(maze[0])
        center_x = width // 2
        center_y = height // 2

        self.player = Player(center_y, center_x)

        self.ghosts = [
            Ghost(0, 0, (255, 0, 0), "Blinky"),
            Ghost(0, width - 1, (255, 182, 193), "Pinky"),
            Ghost(height - 1, 0, (0, 255, 255), "Inky"),
            Ghost(height - 1, width - 1, (255, 165, 0), "Clyde"),
        ]

    def play_sound(self, name):
        self.sound.play_sound_with_duck(name)

    def update(self, maze: list[list[int]], dt: float) -> None:

        px, py = self.player.grid_x, self.player.grid_y
        pellet = self.pellets[py][px]

        if pellet != 0:
            self.pellets[py][px] = 0
            self.total_pellets -= 1

        if pellet == 1:
            self.play_sound("eat_normal_pellet")
            self.score_management.add_normal_pellet()

        elif pellet == 2:
            self.play_sound("eat_super_pacgum")
            self.score_management.add_super_pacgum()

            fright_duration = 7.0
            for ghost in self.ghosts:
                ghost.is_edible = True
                ghost.frightened_timer = fright_duration

            ability = self.super_gum_abilities.pop((px, py), ABILITY_NONE)
            if ability != ABILITY_NONE:
                self.player.start_powered_mode(
                    PacmanMode.PUNCH if ability == ABILITY_PUNCH else PacmanMode.KICK,
                    fright_duration,
                )

        # dt arrives in seconds from the caller -- Animation.update expects milliseconds.
        dt_ms = dt * 1000.0

        # Tick powered-mode timer (synced with ghost fright)
        if self.player.powered_mode is not None:
            self.player.powered_timer = max(0.0, self.player.powered_timer - dt)
            if self.player.powered_timer == 0.0:
                self.player.end_powered_mode()

        self.player.update_animation(dt_ms)

        for ghost in self.ghosts:
            if ghost.is_edible:
                ghost.frightened_timer = max(0.0, ghost.frightened_timer - dt)
                if ghost.frightened_timer == 0.0:
                    ghost.is_edible = False
            ghost.update_animation(dt_ms)

    def init_pellets(self):
        self._init_pellet_grid(self.maze)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw pellets, player, and Ghosts."""
        height = len(self.pellets)
        width = len(self.pellets[0])

        for y in range(height):
            for x in range(width):
                px, py = pellet_screen_pos(x, y)
                if self.pellets[y][x] == 1:
                    pygame.draw.circle(
                        screen,
                        COLOR_PELLET,
                        (px, py),
                        max(2, CELL_SIZE // 8),
                    )
                elif self.pellets[y][x] == 2:
                    pulse = int(math.sin(pygame.time.get_ticks() * 0.01) * 2)
                    ability = self.super_gum_abilities.get((x, y), ABILITY_NONE)
                    if ability == ABILITY_PUNCH:
                        color = (255, 100, 100)
                        r = max(5, CELL_SIZE // 3) + pulse
                    elif ability == ABILITY_KICK:
                        color = (100, 180, 255)
                        r = max(5, CELL_SIZE // 3) + pulse
                    else:
                        color = COLOR_PELLET
                        r = max(4, CELL_SIZE // 4) + pulse
                    pygame.draw.circle(screen, color, (px, py), r)

        self.player.draw(screen)

        for ghost in self.ghosts:
            ghost.draw(screen)

    def reset_positions(self):
        """Reset Pacman and Ghosts to start positions."""
        self.init_pellets()
        if self.player:
            self.player.reset_location()

        for ghost in self.ghosts:
            ghost.reset()


class Entity:
    def __init__(
        self,
        y: int,
        x: int,
        speed: int,
    ) -> None:
        self.spawn_x = x
        self.spawn_y = y
        self.grid_y = y
        self.grid_x = x

        self.x, self.y = grid_to_pixel(y, x)

        self.speed = speed

        self.direction: Optional[str] = None
        self.next_direction: Optional[str] = None

        self.row_direction = 0
        self.col_direction = 0

        # Horizontal facing, persisted across frames. This is the ONLY
        # axis we ever flip on. NORTH/SOUTH movement deliberately leaves
        # this untouched -- see the note on draw() in Player/Ghost for why.
        self.facing = Facing.RIGHT


def _facing_from_direction(direction, current_facing: Facing) -> Facing:
    """
    Map a movement direction string to a horizontal Facing, WITHOUT ever
    producing a vertical flip.

    "LEFT"  -> face left
    "RIGHT" -> face right
    "UP"/"DOWN"/None -> keep whatever horizontal facing we already had

    This is the fix for "Pac-Man flips upside down when turning
    up/down": the bug happens when code tries to flip/rotate on both
    axes (or rotates 180 degrees) to "point" the sprite up or down. Our
    frames only have one vertical orientation (mouth chomping sideways,
    hat on top), so there is nothing to rotate into for NORTH/SOUTH
    without the art looking broken -- we simply keep the last known
    left/right facing and only ever flip on X.
    """
    if direction == "LEFT":
        return Facing.LEFT
    if direction == "RIGHT":
        return Facing.RIGHT
    return current_facing


class Player(Entity):
    def __init__(self, y: int, x: int) -> None:
        super().__init__(y, x, speed=3)

        self.score = 0
        self.msg_txt = ""
        self.is_invincible = False

        # --- special-move mode flags -----------------------------------
        self.sprites = SpriteLibrary.instance()
        self.mode = PacmanMode.NORMAL
        self.is_punching = False
        self.is_kicking = False
        self.current_ability: str = ABILITY_NONE
        self.animation = self.sprites.new_animation(PacmanMode.NORMAL)

        # Powered-mode state (active while super gum fright lasts)
        self.powered_mode: Optional[PacmanMode] = None
        self.powered_timer: float = 0.0
        self.is_attacking: bool = False

    # ------------------------------------------------------------ modes --
    def grant_ability(self, ability: str) -> None:
        """Grant an ability to the player (punch, kick, or none)."""
        self.current_ability = ability

    def activate_ability(self, ability: str) -> None:
        """Start powered walk mode (called when eating a powered super gum).
        The walk animation loops for the duration of the fright timer.
        Attack animation triggers on ghost collision."""
        if self.mode != PacmanMode.NORMAL:
            return

        if ability == ABILITY_PUNCH:
            self.powered_mode = PacmanMode.PUNCH
        elif ability == ABILITY_KICK:
            self.powered_mode = PacmanMode.KICK
        else:
            return

        self.mode = self.powered_mode
        self.animation = self.sprites.new_walk_animation(self.powered_mode)

    def trigger_attack(self) -> None:
        """Trigger the attack animation on ghost collision during powered mode."""
        if self.powered_mode is None or self.is_attacking:
            return
        self.is_attacking = True
        self.animation = self.sprites.new_attack_animation(self.powered_mode)

    def start_powered_mode(self, mode: PacmanMode, duration: float) -> None:
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
        self.mode = PacmanMode.NORMAL
        self.animation = self.sprites.new_animation(PacmanMode.NORMAL)

    def use_ability(self) -> None:
        """Activate the current ability if available and not already in a special mode."""
        if self.mode != PacmanMode.NORMAL or self.current_ability == ABILITY_NONE:
            return

        if self.current_ability == ABILITY_PUNCH:
            self.mode = PacmanMode.PUNCH
            self.is_punching = True
            self.animation = self.sprites.new_animation(PacmanMode.PUNCH)
        elif self.current_ability == ABILITY_KICK:
            self.mode = PacmanMode.KICK
            self.is_kicking = True
            self.animation = self.sprites.new_animation(PacmanMode.KICK)

        self.current_ability = ABILITY_NONE

    def update_animation(self, dt_ms: float) -> None:
        self.facing = _facing_from_direction(self.direction, self.facing)

        self.animation.update(dt_ms)

        # Powered mode: attack finished -> return to walk loop
        if self.is_attacking and self.animation.finished:
            self.is_attacking = False
            self.animation = self.sprites.new_walk_animation(self.powered_mode)

        # Powered mode expired (handled by EntityManager via powered_timer)
        # Normal punch/kick one-shot finish (legacy fallback)
        if (
            self.mode != PacmanMode.NORMAL
            and self.powered_mode is None
            and self.animation.finished
        ):
            self.is_punching = False
            self.is_kicking = False
            self.mode = PacmanMode.NORMAL
            self.animation = self.sprites.new_animation(PacmanMode.NORMAL)

    # ------------------------------------------------------------- draw --
    def draw(self, screen: pygame.Surface) -> None:
        x, y = pixel_to_screen(self.x, self.y)

        frame = self.animation.current_frame
        if self.facing == Facing.LEFT:
            frame = pygame.transform.flip(frame, True, False)

        rect = frame.get_rect(center=(int(x), int(y)))
        screen.blit(frame, rect)

    def is_valid_spawn(self, row: int, col: int) -> bool:
        cell = self.maze[row][col]
        return cell != (NORTH | EAST | SOUTH | WEST)

    def find_player_spawn(self) -> tuple[int, int]:
        middle_row = self.game.curr_level.height // 2
        middle_col = self.game.curr_level.width // 2

        for radius in range(
            max(self.game.curr_level.width, self.game.curr_level.height)
        ):
            for row in range(middle_row - radius, middle_row + radius + 1):
                for col in range(middle_col - radius, middle_col + radius + 1):
                    if (
                        0 <= row < self.game.curr_level.height
                        and 0 <= col < self.game.curr_level.width
                        and self.is_valid_spawn(row, col)
                    ):
                        return row, col

        return 0, 0

    def reset_location(self):
        self.x, self.y = grid_to_pixel(self.spawn_y, self.spawn_x)

        self.direction = None
        self.next_direction = None

        self.row_direction = 0
        self.col_direction = 0

        self.facing = Facing.RIGHT
        self.mode = PacmanMode.NORMAL
        self.is_punching = False
        self.is_kicking = False
        self.is_attacking = False
        self.current_ability = ABILITY_NONE
        self.powered_mode = None
        self.powered_timer = 0.0
        self.animation = self.sprites.new_animation(PacmanMode.NORMAL)


class Ghost(Entity):
    def __init__(
        self,
        y: int,
        x: int,
        color,
        name: str,
    ) -> None:
        super().__init__(y, x, speed=1)

        self.name = name
        self.color = color
        self.ghost_color = NAME_TO_GHOST_COLOR.get(name, GhostColor.RED)

        self.spawn_x = x
        self.spawn_y = y

        self.is_edible = False
        self.is_eaten = False
        self.frightened_timer = 0.0
        self.runaway_target = None

        self.sprites = SpriteLibrary.instance()
        self.state = GhostState.HUNT
        self._last_vertical = None
        self.animation = self.sprites.new_ghost_animation(
            GhostState.HUNT, color=self.ghost_color, facing=self.facing
        )

    # ------------------------------------------------------------ modes --
    def _current_state(self) -> GhostState:
        if self.is_eaten:
            return GhostState.EATEN
        if self.is_edible:
            return GhostState.FRIGHTENED
        return GhostState.HUNT

    def update_animation(self, dt_ms: float) -> None:
        old_facing = self.facing
        self.facing = _facing_from_direction(self.direction, self.facing)

        vertical = None
        if self.direction == NORTH:
            vertical = "up"
        elif self.direction == SOUTH:
            vertical = "down"

        new_state = self._current_state()

        changed = (
            new_state != self.state
            or self.facing != old_facing
            or (new_state == GhostState.EATEN and vertical != self._last_vertical)
        )

        if changed:
            self.state = new_state
            self._last_vertical = vertical
            self.animation = self.sprites.new_ghost_animation(
                new_state, color=self.ghost_color, facing=self.facing, vertical=vertical
            )

        self.animation.update(dt_ms)

    def draw(self, screen: pygame.Surface) -> None:
        x, y = pixel_to_screen(self.x, self.y)

        frame = self.animation.current_frame
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
        self.state = GhostState.HUNT
        self.animation = self.sprites.new_ghost_animation(
            GhostState.HUNT, color=self.ghost_color, facing=self.facing
        )
