import math
import random
from typing import Optional, Any

import pygame

from src.graphics.entitys.ghost import Ghost
from src.graphics.entitys.player import Player
from src.logic.config import CELL_SIZE, GameConfig

from src.logic.helpers import pellet_screen_pos
from src.graphics.ui_helpers import COLOR_PELLET
from src.sounds.soud_manager import SoundManager

from src.graphics.entitys.graphic_lib import (
    SpriteLibrary,
    GhostColor,
)
from src.graphics.entitys.graphic_lib import PacmanMode as pm

NAME_TO_GHOST_COLOR = {
    "Blinky": GhostColor.RED,
    "Pinky": GhostColor.PINK,
    "Inky": GhostColor.CYAN,
    "Clyde": GhostColor.ORANGE,
}


ABILITY_NONE = "none"
ABILITY_PUNCH = "punch"
ABILITY_KICK = "kick"


class EntityManager:
    def __init__(
        self,
        config: GameConfig,
        score_management: Any,
        game: Any,
    ) -> None:
        """Initialize systems and settings from configuration."""
        self.config: GameConfig = config
        self.score_management = score_management
        self.game = game
        self.player: Optional[Player] = None
        self.ghosts: list[Ghost] = []
        self.pellets: list[list[int]] = []
        self.total_pellets: int = 0
        self.initial_total_pellets: int = 0
        self.spawned_milestones: set[int] = set()
        self.sound = SoundManager()
        self.super_gum_abilities: dict[tuple[int, int], str] = {}

        self.sprites = SpriteLibrary.instance()
        self.sprites.load(CELL_SIZE)
        self.sprites.load_ghosts(CELL_SIZE)

    def _init_pellet_grid(self, maze: list[list[int]]) -> None:
        """Initialize pellet grid,super-gum abilities, and counts from maze."""
        height, width = len(maze), len(maze[0])
        self.pellets = [[0] * width for _ in range(height)]
        self.total_pellets = 0
        self.super_gum_abilities = {}

        corners = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
        ]
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

        self.initial_total_pellets = self.total_pellets
        self.spawned_milestones = set()

    def spawn_special_pellet(self) -> Optional[tuple[int, int]]:
        """Spawn a special gamble pellet at a random accessible floor cell."""
        if not self.maze:
            return None
        height, width = len(self.maze), len(self.maze[0])
        candidates = [
            (y, x)
            for y in range(height)
            for x in range(width)
            if self.maze[y][x] != 15 and self.pellets[y][x] == 0
        ]
        occupied = set()
        if self.player:
            occupied.add((self.player.grid_y, self.player.grid_x))
        for g in self.ghosts:
            occupied.add((g.grid_y, g.grid_x))

        filtered = [c for c in candidates if c not in occupied]
        choices = filtered if filtered else candidates

        if not choices:
            choices = [
                (y, x)
                for y in range(height)
                for x in range(width)
                if self.maze[y][x] != 15
            ]

        if choices:
            sy, sx = random.choice(choices)
            self.pellets[sy][sx] = 3
            return (sy, sx)
        return None

    def load_level_entities(self, maze: list[list[int]]) -> None:
        """Setup maze grid, pellets, and spawn entities."""
        self.maze = maze
        self._init_pellet_grid(maze)

        height = len(maze)
        width = len(maze[0])
        center_x = width // 2
        center_y = height // 2
        self.player = Player(center_y, center_x)
        self.player.find_player_spawn(self.game, maze)
        self.ghosts = [
            Ghost(0, 0, (255, 0, 0), "Blinky"),
            Ghost(0, width - 1, (255, 182, 193), "Pinky"),
            Ghost(height - 1, 0, (0, 255, 255), "Inky"),
            Ghost(height - 1, width - 1, (255, 165, 0), "Clyde"),
        ]
        self.get_42_patterns()

    def play_sound(self, name: str) -> None:
        self.sound.play_sound_with_duck(name)

    def update(self, maze: list[list[int]], dt: float) -> None:
        assert self.player is not None
        px, py = self.player.grid_x, self.player.grid_y
        pellet = self.pellets[py][px]

        if pellet != 0:
            self.pellets[py][px] = 0
            if pellet in (1, 2):
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
                    (pm.PUNCH if ability == ABILITY_PUNCH else pm.KICK),
                    fright_duration,
                )

        elif pellet == 3:
            self.play_sound("eat_super_pacgum")
            self.score_management.add_super_pacgum()
            curr_state = getattr(
                getattr(self.game, "state_manager", None), "current_state", None
            )
            if curr_state and hasattr(curr_state, "on_special_pellet_eaten"):
                curr_state.on_special_pellet_eaten()

        # Check 30%, 60%, 90% milestones to spawn special mystery pellets
        if self.initial_total_pellets > 0:
            cleared = self.initial_total_pellets - self.total_pellets
            for milestone in (30, 60, 90):
                threshold = int(
                    self.initial_total_pellets * (milestone / 100.0)
                )
                if (
                    cleared >= threshold
                    and milestone not in self.spawned_milestones
                ):
                    self.spawned_milestones.add(milestone)
                    spawned = self.spawn_special_pellet()
                    if spawned:
                        curr_state = getattr(
                            getattr(self.game, "state_manager", None),
                            "current_state",
                            None,
                        )
                        if curr_state and hasattr(
                            curr_state, "on_special_pellet_spawned"
                        ):
                            curr_state.on_special_pellet_spawned()

        dt_ms = dt * 1000.0

        if self.player.powered_mode is not None:
            self.player.powered_timer = max(
                0.0,
                self.player.powered_timer - dt,
            )
            if self.player.powered_timer == 0.0:
                self.player.end_powered_mode()

        self.player.update_animation(dt_ms)

        for ghost in self.ghosts:
            if ghost.is_eaten:
                if (
                    ghost.grid_x,
                    ghost.grid_y,
                ) == (
                    ghost.spawn_x,
                    ghost.spawn_y,
                ):
                    if ghost.respawn_timer < 0:
                        ghost.respawn_timer = 5.0
                    else:
                        ghost.respawn_timer -= dt
                        if ghost.respawn_timer <= 0:
                            ghost.reset()

            if ghost.is_edible:
                ghost.frightened_timer = max(0.0, ghost.frightened_timer - dt)
                if ghost.frightened_timer == 0.0:
                    ghost.is_edible = False
            ghost.update_animation(dt_ms)

    def init_pellets(self) -> None:
        self._init_pellet_grid(self.maze)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw pellets, player, and Ghosts."""
        assert self.player is not None
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
                    ability = self.super_gum_abilities.get(
                        (x, y),
                        ABILITY_NONE,
                    )
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
                elif self.pellets[y][x] == 3:
                    pulse = int(math.sin(pygame.time.get_ticks() * 0.008) * 3)
                    r = max(6, CELL_SIZE // 3) + pulse
                    pygame.draw.circle(
                        screen, (255, 0, 255), (px, py), r + 2, width=2
                    )
                    pygame.draw.circle(screen, (220, 20, 220), (px, py), r)
                    pygame.draw.circle(
                        screen, (255, 240, 255), (px, py), max(2, r // 2)
                    )

        self.player.draw(screen)

        for ghost in self.ghosts:
            ghost.draw(screen)

    def reset_positions(self) -> None:
        """Reset Pacman and Ghosts to start positions."""
        self.init_pellets()
        if self.player:
            self.player.reset_location()

        for ghost in self.ghosts:
            ghost.reset()

    def get_42_patterns(self) -> bool:
        pattern_4_cells = []
        pattern_2_cells = []
        height = len(self.maze)
        width = len(self.maze[0])

        if 5 > height or 7 > width:
            for ghost in self.ghosts:
                ghost.prison_cells = None
            return False

        for y in range(height):
            for x in range(width):
                if self.maze[y][x] == 15:
                    if x < width // 2:
                        pattern_4_cells.append((y, x))
                    else:
                        pattern_2_cells.append((y, x))
        for ghost in self.ghosts:
            if ghost.name in ("Blinky", "Pinky"):
                ghost.prison_cells = pattern_4_cells
            else:
                ghost.prison_cells = pattern_2_cells

        return True
