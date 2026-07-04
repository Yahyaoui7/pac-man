# EntityManager that will control the :
# Pacman
# Ghosts
# Pellets
# Walls

import pygame
import random
TOP_BAR_HEIGHT = 30
CELL_SIZE = 30
PADDING = 20
NORTH = 1 << 0
EAST = 1 << 1
SOUTH = 1 << 2
WEST = 1 << 3


class Player:
    def __init__(self, row: int, col: int, cell_size: int, screen) -> None:
        self.row = row
        self.col = col
        self.cell_size = cell_size
        self.screen = screen

        self.x = col * cell_size + cell_size // 2
        self.y = row * cell_size + cell_size // 2

        self.lives = 3
        self.score = 0
        self.speed = 5

        self.direction = None
        self.next_direction = None

        self.row_direction = 0
        self.col_direction = 0

    def is_valid_spawn(self, row, col, maze):
        cell = maze.maze[row][col]
        return cell != (NORTH | EAST | SOUTH | WEST)

    def find_player_spawn(self, curr_level, maze):
        middle_row = curr_level.height // 2
        middle_col = curr_level.width // 2

        for radius in range(max(curr_level.width, curr_level.height)):
            for row in range(middle_row - radius, middle_row + radius + 1):
                for col in range(middle_col - radius, middle_col + radius + 1):
                    if (
                        0 <= row < curr_level.height
                        and 0 <= col < curr_level.width
                    ):
                        if self.is_valid_spawn(row, col, maze):
                            return row, col

        return 0, 0

    def draw_player(self):
        if self.player is None:
            return

        x = PADDING // 2 + self.player.x
        y = PADDING // 2 + self.player.y + TOP_BAR_HEIGHT

        pygame.draw.circle(
            self.screen,
            "yellow",
            (x, y),
            CELL_SIZE // 3,
        )


class Ghost:
    def __init__(
        self, row: int, col: int, name: str, cell_size: int, screen
    ) -> None:
        self.row = row
        self.col = col
        self.cell_size = cell_size
        self.screen = screen

        self.x = col * cell_size + cell_size // 2
        self.y = row * cell_size + cell_size // 2

        self.speed = 5
        self.name = name

        self.direction = None
        self.next_direction = None

        self.row_direction = 0
        self.col_direction = 0

        self.is_edible = False
        self.is_eaten = False

        self.spawn_row = row
        self.spawn_col = col
        self.spawn_x = self.x
        self.spawn_y = self.y

    def draw_ghosts(self):
        colors = {
            "Blinky": "red",
            "Pinky": "pink",
            "Inky": "cyan",
            "Clyde": "orange",
        }

        for ghost in self.ghosts:
            x = PADDING // 2 + ghost.x
            y = PADDING // 2 + ghost.y + TOP_BAR_HEIGHT

            pygame.draw.circle(
                self.screen,
                colors.get(ghost.name, "white"),
                (x, y),
                CELL_SIZE // 3,
            )

    def update_random_ghosts(self, movement):
        for ghost in self.ghosts:
            if movement.is_centered(ghost):
                movement.update_cell_position(ghost)

                possible_directions = []

                for direction in self.directions:
                    if movement.can_move(ghost.row, ghost.col, direction):
                        possible_directions.append(direction)

                if possible_directions:
                    movement.set_direction(
                        ghost, random.choice(possible_directions)
                    )

            movement.update_entity(ghost)

    def reset_ghost(self):
        self.row = self.spawn_row
        self.col = self.spawn_col
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.direction = None
        self.next_direction = None
        self.is_eaten = False
