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
