# main game loop

# GameStarter
# │
# ├── Renderer
# ├── InputManager
# ├── LevelManager
# ├── EntityManager
# ├── AudioManager
# ├── UISystem
# └── CollisionSystem

from mazegenerator import MazeGenerator
from src.graphics.renderer import HomeState, StateManager
from .logic.config import GameConfig
from .logic.inputmanager import InputManager
from .logic.entities import Player, Ghost
from .logic.movement import MovementSystem
import random
import pygame
import sys

sys.setrecursionlimit(99999999)
TOP_BAR_HEIGHT = 30
CELL_SIZE = 30
PADDING = 20
NORTH = 1 << 0
EAST = 1 << 1
SOUTH = 1 << 2
WEST = 1 << 3


class GameStarter:

    def __init__(self, config: GameConfig):
        self.running = True
        self.config = config
        self.maze = None
        self.screen = None
        self.curr_level = None
        self.state_manager = StateManager(self)
        self.player = None
        self.ghosts = []
        self.directions = ["LEFT", "RIGHT", "UP", "DOWN"]

    def display(self):

        self.screen = pygame.display.set_mode(
            (
                self.curr_level.width * CELL_SIZE + PADDING,
                self.curr_level.height * CELL_SIZE + PADDING + 50,
            )
        )
        pygame.display.flip()

    def draw_maze(self, maze):

        for row, cells in enumerate(maze):

            for col, cell in enumerate(cells):

                x = PADDING // 2 + col * CELL_SIZE

                y = PADDING // 2 + row * CELL_SIZE + TOP_BAR_HEIGHT

                if cell & NORTH:
                    pygame.draw.line(
                        self.screen,
                        "blue",
                        (x, y),
                        (x + CELL_SIZE, y),
                        2,
                    )

                if cell & EAST:
                    pygame.draw.line(
                        self.screen,
                        "blue",
                        (x + CELL_SIZE, y),
                        (x + CELL_SIZE, y + CELL_SIZE),
                        2,
                    )

                if cell & SOUTH:
                    pygame.draw.line(
                        self.screen,
                        "blue",
                        (x, y + CELL_SIZE),
                        (x + CELL_SIZE, y + CELL_SIZE),
                        2,
                    )

                if cell & WEST:
                    pygame.draw.line(
                        self.screen,
                        "blue",
                        (x, y),
                        (x, y + CELL_SIZE),
                        2,
                    )

    def run(self):

        self.curr_level = self.config.levels[0]

        self.curr_level.height = min(self.curr_level.height, 32)
        self.curr_level.width = min(self.curr_level.width, 60)

        self.maze = MazeGenerator(
            size=(self.curr_level.width, self.curr_level.height),
            entry_cell=(0, 0),
            exit_cell=(0, 0),
            perfect=False,
            seed=self.curr_level.seed,
        )

        pygame.init()

        clock = pygame.time.Clock()

        self.display()

        input_manager = InputManager()

        self.state_manager.change_state(HomeState(self))

        #
        player_row, player_col = self.find_player_spawn()
        self.player = Player(player_row, player_col, CELL_SIZE)

        movement = MovementSystem(self.maze.maze)
        self.ghosts = [
            Ghost(0, 0, "Blinky", CELL_SIZE),
            Ghost(0, self.curr_level.width - 1, "Pinky", CELL_SIZE),
            Ghost(self.curr_level.height - 1, 0, "Inky", CELL_SIZE),
            Ghost(
                self.curr_level.height - 1,
                self.curr_level.width - 1,
                "Clyde",
                CELL_SIZE,
            ),
        ]

        while self.running:

            events = pygame.event.get()

            input_state = input_manager.update(events)

            if input_state.quit_requested:
                self.running = False

            self.state_manager.update(input_state)

            self.state_manager.draw(self.screen)

            pygame.display.flip()

            clock.tick(60)

        pygame.quit()

        if input_state.pause_pressed:
            print("Pause")

            if input_state.move_left:
                self.player.next_direction = "LEFT"
            elif input_state.move_right:
                self.player.next_direction = "RIGHT"
            elif input_state.move_up:
                self.player.next_direction = "UP"
            elif input_state.move_down:
                self.player.next_direction = "DOWN"

            self.update_random_ghosts(movement)

            movement.update_entity(self.player)
            self.screen.fill("black")
            self.draw_maze(self.maze.maze)
            self.draw_player()
            self.draw_ghosts()

            pygame.display.flip()
            clock.tick(25)

        pygame.quit()

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

    def is_valid_spawn(self, row, col):
        cell = self.maze.maze[row][col]
        return cell != (NORTH | EAST | SOUTH | WEST)

    def find_player_spawn(self):
        middle_row = self.curr_level.height // 2
        middle_col = self.curr_level.width // 2

        for radius in range(max(self.curr_level.width, self.curr_level.height)):
            for row in range(middle_row - radius, middle_row + radius + 1):
                for col in range(middle_col - radius, middle_col + radius + 1):
                    if (
                        0 <= row < self.curr_level.height
                        and 0 <= col < self.curr_level.width
                    ):
                        if self.is_valid_spawn(row, col):
                            return row, col

        return 0, 0

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
                    movement.set_direction(ghost, random.choice(possible_directions))

            movement.update_entity(ghost)
