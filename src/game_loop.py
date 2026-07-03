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
import pygame
import sys

sys.setrecursionlimit(99999999)
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

    def display(self):

        self.screen = pygame.display.set_mode(
            (
                self.curr_level.width * CELL_SIZE + PADDING,
                self.curr_level.height * CELL_SIZE + PADDING + 20,
            )
        )
        pygame.display.flip()

    def draw_maze(self, maze):

        for row, cells in enumerate(maze):

            for col, cell in enumerate(cells):

                x = PADDING // 2 + col * CELL_SIZE

                y = PADDING // 2 + row * CELL_SIZE + 40

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
        self.curr_level = self.config.levels[6]
        pygame.init()

        clock = pygame.time.Clock()

        self.display()

        input_manager = InputManager()

        self.state_manager.change_state(HomeState(self))

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

        # def run(self):
        #     self.curr_level = self.config.levels[6]
        # if self.curr_level.height > 33:
        #     self.curr_level.height = 33
        # if self.curr_level.width > 60:
        #     self.curr_level.width = 60
        # if self.curr_level.height == 33:
        #     hight = 32
        # self.maze = MazeGenerator(
        #     size=(self.curr_level.width, hight),
        #     entry_cell=(0, 0),
        #     exit_cell=(0, 0),
        #     perfect=False,
        #     seed=self.curr_level.seed,
        # )
        # pygame.init()
        # clock = pygame.time.Clock()

        # self.display()
        # self.draw_maze(self.maze.maze)
        # input_manager = InputManager()

        # state_manager = StateManager()
        # state_manager.change_state(HomeState(state_manager))

        # while self.running:

        #     events = pygame.event.get()
        #     input_state = input_manager.update(events)

        #     if input_state.quit_requested:
        #         self.running = False

        #     state_manager.update(input_state)

        #     self.screen.fill("black")
        #     state_manager.draw(self.screen)

        #     pygame.display.flip()
        #     clock.tick(60)

        pygame.quit()
