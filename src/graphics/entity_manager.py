import math
from typing import Optional

import pygame

from src.logic.config import CELL_SIZE, PADDING, TOP_BAR_HEIGHT, GameConfig
from src.logic.config import NORTH, EAST, SOUTH, WEST
from src.graphics.graphic_lib import (
    SpriteLibrary,
    PacmanMode,
    Facing,
    GhostColor,
    GhostState,
)

# Blinky/Pinky/Inky/Clyde -> the color keys used in assets/ghost_sprites
NAME_TO_GHOST_COLOR = {
    "Blinky": GhostColor.RED,
    "Pinky": GhostColor.PINK,
    "Inky": GhostColor.CYAN,
    "Clyde": GhostColor.ORANGE,
}


class EntityManager:
    def __init__(self, config: GameConfig) -> None:
        """Initialize systems and settings from configuration."""
        self.config: GameConfig = config
        self.player: Optional[Player] = None
        self.ghosts: list[Ghost] = []
        self.pellets: list[list[int]] = []
        self.total_pellets: int = 0
        self.maze = None

        # Sprite frames are loaded once here and shared by every entity.
        self.sprites = SpriteLibrary.instance()
        self.sprites.load(CELL_SIZE)
        self.sprites.load_ghosts(CELL_SIZE)

    def load_level_entities(self, maze: list[list[int]]) -> None:
        """Setup maze grid, pellets, and spawn entities."""
        self.maze = maze
        self.init_pellets()
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

    def update(self, maze: list[list[int]], dt: float) -> None:

        px, py = self.player.grid_x, self.player.grid_y

        if self.pellets[py][px] == 1:
            self.pellets[py][px] = 0
            self.total_pellets -= 1
        elif self.pellets[py][px] == 2:
            self.pellets[py][px] = 0
            self.total_pellets -= 1

            for ghost in self.ghosts:
                ghost.is_edible = True
                if ghost.is_edible:
                    ghost.frightened_timer = 7.0

        # dt arrives in seconds from the caller (see frightened_timer usage
        # below) -- Animation.update expects milliseconds.
        dt_ms = dt * 1000.0

        self.player.update_animation(dt_ms)

        for ghost in self.ghosts:
            if ghost.is_edible:
                ghost.frightened_timer = max(0.0, ghost.frightened_timer - dt)
                if ghost.frightened_timer == 0.0:
                    ghost.is_edible = False
            ghost.update_animation(dt_ms)

    def init_pellets(self):
        height = len(self.maze)
        width = len(self.maze[0])

        self.pellets = [[0] * width for _ in range(height)]
        self.total_pellets = 0

        corners = {
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
        }

        center = (width // 2, height // 2)

        for y in range(height):
            for x in range(width):

                if (x, y) in corners:
                    self.pellets[y][x] = 2
                    self.total_pellets += 1
                elif (x, y) == center or self.maze[y][x] == 15:
                    self.pellets[y][x] = 0
                else:
                    self.pellets[y][x] = 1
                    self.total_pellets += 1

    def draw(self, screen: pygame.Surface) -> None:
        """Draw pellets, player, and Ghosts."""
        height = len(self.pellets)
        width = len(self.pellets[0])

        for y in range(height):
            for x in range(width):
                px = int(PADDING // 2 + x * CELL_SIZE + CELL_SIZE // 2)
                py = int(PADDING // 2 + y * CELL_SIZE + CELL_SIZE // 2 + 30)
                if self.pellets[y][x] == 1:
                    pygame.draw.circle(
                        screen,
                        (255, 184, 151),
                        (px, py),
                        max(2, CELL_SIZE // 8),
                    )
                elif self.pellets[y][x] == 2:
                    pulse = int(math.sin(pygame.time.get_ticks() * 0.01) * 2)
                    r = max(4, CELL_SIZE // 4) + pulse
                    pygame.draw.circle(screen, (255, 184, 151), (px, py), r)

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

        self.x = x * CELL_SIZE + CELL_SIZE // 2
        self.y = y * CELL_SIZE + CELL_SIZE // 2

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
    Map a movement direction to a horizontal Facing, WITHOUT ever
    producing a vertical flip.

    WEST  -> face left
    EAST  -> face right
    NORTH/SOUTH/None -> keep whatever horizontal facing we already had

    This is the fix for "Pac-Man flips upside down when turning
    up/down": the bug happens when code tries to flip/rotate on both
    axes (or rotates 180 degrees) to "point" the sprite up or down. Our
    frames only have one vertical orientation (mouth chomping sideways,
    hat on top), so there is nothing to rotate into for NORTH/SOUTH
    without the art looking broken -- we simply keep the last known
    left/right facing and only ever flip on X.
    """
    if direction == WEST:
        return Facing.LEFT
    if direction == EAST:
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
        self.animation = self.sprites.new_animation(PacmanMode.NORMAL)

    # ------------------------------------------------------------ modes --
    def activate_punch(self) -> None:
        """Turn on punch mode. No-op if already mid-special so a mashed
        input can't cancel/restart the animation partway through."""
        if self.mode == PacmanMode.NORMAL:
            self.mode = PacmanMode.PUNCH
            self.is_punching = True
            self.animation = self.sprites.new_animation(PacmanMode.PUNCH)

    def activate_kick(self) -> None:
        if self.mode == PacmanMode.NORMAL:
            self.mode = PacmanMode.KICK
            self.is_kicking = True
            self.animation = self.sprites.new_animation(PacmanMode.KICK)

    def update_animation(self, dt_ms: float) -> None:
        self.facing = _facing_from_direction(self.direction, self.facing)

        self.animation.update(dt_ms)
        if self.mode != PacmanMode.NORMAL and self.animation.finished:
            self.mode = PacmanMode.NORMAL
            self.is_punching = False
            self.is_kicking = False
            self.animation = self.sprites.new_animation(PacmanMode.NORMAL)

    # ------------------------------------------------------------- draw --
    def draw(self, screen: pygame.Surface) -> None:
        x = PADDING // 2 + self.x
        y = TOP_BAR_HEIGHT + PADDING // 2 + self.y

        frame = self.animation.current_frame
        # Horizontal-only mirror. flip(frame, True, False) -- the second
        # argument (vertical flip) must always stay False here, or turning
        # to face up/down will flip Pac-Man upside down.
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
        self.x = self.spawn_x * CELL_SIZE + CELL_SIZE // 2
        self.y = self.spawn_y * CELL_SIZE + CELL_SIZE // 2

        self.direction = None
        self.next_direction = None

        self.row_direction = 0
        self.col_direction = 0

        self.facing = Facing.RIGHT
        self.mode = PacmanMode.NORMAL
        self.is_punching = False
        self.is_kicking = False
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

        # eaten-eyes get proper up/down art since it exists; everyone else
        # only ever changes horizontal facing (see _facing_from_direction)
        vertical = None
        if self.direction == NORTH:
            vertical = "up"
        elif self.direction == SOUTH:
            vertical = "down"

        new_state = self._current_state()

        # Rebuild the (cheap, index-reset) Animation only when something
        # that actually changes which frame-set we should show has
        # changed -- state, horizontal facing, or (for EATEN) vertical
        # direction. Rebuilding every tick would reset the frame index
        # each time and the wiggle/blink would never animate.
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

    # ------------------------------------------------------------- draw --
    def draw(self, screen: pygame.Surface) -> None:
        x = PADDING // 2 + self.x
        y = TOP_BAR_HEIGHT + PADDING // 2 + self.y

        frame = self.animation.current_frame
        rect = frame.get_rect(center=(int(x), int(y)))
        screen.blit(frame, rect)

    def reset(self) -> None:
        self.grid_y = self.spawn_y
        self.grid_x = self.spawn_x

        self.x = self.spawn_x * CELL_SIZE + CELL_SIZE // 2
        self.y = self.spawn_y * CELL_SIZE + CELL_SIZE // 2
        self.is_edible = False
        self.is_eaten = False
        self.frightened_timer = 0.0

        self.facing = Facing.RIGHT
        self.state = GhostState.HUNT
        self.animation = self.sprites.new_ghost_animation(
            GhostState.HUNT, color=self.ghost_color, facing=self.facing
        )
