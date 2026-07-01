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
from .logic.config import GameConfig
from .logic.inputmanager import InputManager
import pygame

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

    def display(self):

        self.screen = pygame.display.set_mode(
            (
                self.curr_level.width * CELL_SIZE + PADDING,
                self.curr_level.height * CELL_SIZE + PADDING,
            )
        )
        pygame.display.flip()

    def draw_maze(self, maze):

        for row, cells in enumerate(maze):

            for col, cell in enumerate(cells):

                x = PADDING // 2 + col * CELL_SIZE
                y = PADDING // 2 + row * CELL_SIZE

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
        self.draw_maze(self.maze.maze)
        input_manager = InputManager()

        while self.running:

            events = pygame.event.get()

            input_state = input_manager.update(events)
            if input_state.quit_requested:
                self.running = False

            if input_state.quit_requested:
                self.running = False

            if input_state.pause_pressed:
                print("Pause")

            if input_state.move_left:
                print("Move Left")

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
