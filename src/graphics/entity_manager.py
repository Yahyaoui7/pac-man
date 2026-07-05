import math
from typing import Optional

import pygame

from src.logic.config import CELL_SIZE, PADDING, TOP_BAR_HEIGHT, GameConfig
from src.logic.config import NORTH, EAST, SOUTH, WEST


class EntityManager:
    def __init__(self, config: GameConfig) -> None:
        """Initialize systems and settings from configuration."""
        self.config: GameConfig = config
        self.player: Optional[Player] = None
        self.ghosts: list[Ghost] = []
        self.pellets: list[list[int]] = []
        self.total_pellets: int = 0

    def load_level_entities(self, maze: list[list[int]]) -> None:
        """Setup maze grid, pellets, and spawn entities."""
        height = len(maze)
        width = len(maze[0])

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
                elif (x, y) == center or maze[y][x] == 15:
                    self.pellets[y][x] = 0
                else:
                    self.pellets[y][x] = 1
                    self.total_pellets += 1

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


class Entity:
    def __init__(
        self,
        y: int,
        x: int,
        speed: int,
    ) -> None:
        self.grid_y = y
        self.grid_x = x

        self.x = x * CELL_SIZE + CELL_SIZE // 2
        self.y = y * CELL_SIZE + CELL_SIZE // 2

        self.speed = speed

        self.direction: Optional[str] = None
        self.next_direction: Optional[str] = None

        self.row_direction = 0
        self.col_direction = 0


class Player(Entity):
    def __init__(self, y: int, x: int) -> None:
        super().__init__(y, x, speed=3)

        self.lives = 3
        self.score = 0
        self.is_invincible = False

    def draw(self, screen: pygame.Surface) -> None:

        x = PADDING // 2 + self.x
        y = TOP_BAR_HEIGHT + PADDING // 2 + self.y

        pygame.draw.circle(
            screen,
            "yellow",
            (int(x), int(y)),
            CELL_SIZE // 3,
        )

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

        self.spawn_x = x
        self.spawn_y = y

        self.is_edible = False
        self.is_eaten = False

    def draw(self, screen: pygame.Surface) -> None:
        colors = {
            "Blinky": "red",
            "Pinky": "pink",
            "Inky": "cyan",
            "Clyde": "orange",
        }

        x = PADDING // 2 + self.x
        y = TOP_BAR_HEIGHT + PADDING // 2 + self.y

        pygame.draw.circle(
            screen,
            colors.get(self.name, "white"),
            (int(x), int(y)),
            CELL_SIZE // 3,
        )
