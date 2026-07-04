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
from src.logic.level_manager import LevelManager
from .logic.config import GameConfig
from .logic.inputmanager import InputManager
import random
import pygame
import sys

from src.logic.entities import Player, Ghost
from src.logic.movement import MovementSystem

sys.setrecursionlimit(99999999)

CELL_SIZE = 30
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
        self.level_manager = LevelManager()

    def resize_window(self, width: int, height: int) -> None:
        """Resize the window dynamically if width/height changed."""
        if self.screen is None or self.screen.get_size() != (width, height):
            self.screen = pygame.display.set_mode((width, height))

    def run(self) -> None:
        """Run the main game loop at 60 FPS."""
        pygame.init()
        pygame.display.set_caption("NEON PAC-MAN")

        clock = pygame.time.Clock()
        input_manager = InputManager()

        # Initialize to Main Menu
        self.state_manager.change_state(HomeState(self))

        while self.running:
            events = pygame.event.get()
            input_state = input_manager.update(events)

            if input_state.quit_requested:
                self.running = False

            # Update the current state
            self.state_manager.update(input_state, events)

            # Draw the current state
            if self.screen:
                self.screen.fill((0, 0, 0))
                self.state_manager.draw(self.screen)
                pygame.display.flip()

            clock.tick(60)

        pygame.quit()

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
        self.player = Player()
        self.player.find_player_spawn(self.curr_level, self.maze)

        movement = MovementSystem(self.maze.maze)
        # self.ghosts = [
        #     Ghost(0, 0, "Blinky", CELL_SIZE),
        #     Ghost(0, self.curr_level.width - 1, "Pinky", CELL_SIZE),
        #     Ghost(self.curr_level.height - 1, 0, "Inky", CELL_SIZE),
        #     Ghost(
        #         self.curr_level.height - 1,
        #         self.curr_level.width - 1,
        #         "Clyde",
        #         CELL_SIZE,
        #     ),
        # ]

        while self.running:

            events = pygame.event.get()

            input_state = input_manager.update(events)

            if input_state.quit_requested:
                self.running = False

            self.state_manager.update(input_state)

            self.state_manager.draw(self.screen)

            pygame.display.flip()

            clock.tick(60)

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

            # self.update_random_ghosts(movement)

            movement.update_entity(self.player)
            self.screen.fill("black")
            self.draw_maze(self.maze.maze)
            self.player.draw_player()
            # self.draw_ghosts()

            pygame.display.flip()
            clock.tick(25)

        pygame.quit()
