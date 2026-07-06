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


from src.graphics.entity_manager import EntityManager
from src.graphics.renderer import HomeState, StateManager
from src.logic.level_manager import LevelManager
from src.sounds.soud_manager import SoundManager
from .logic.inputmanager import InputManager
from .logic.config import GameConfig

import pygame
import sys

sys.setrecursionlimit(99999999)


class GameStarter:

    def __init__(self, config: GameConfig):
        self.running = True
        self.config = config
        self.screen = None
        self.state_manager = StateManager(self)
        self.level_manager = LevelManager(config)
        self.sound_manager = SoundManager()
        self.entity_manager = EntityManager(config)
        self.lives: int = config.lives

    def resize_window(self, width: int, height: int) -> None:
        """Resize the window dynamically if width/height changed."""
        if self.screen is None or self.screen.get_size() != (width, height):
            self.screen = pygame.display.set_mode((width, height))

    def recalculate_cell_size(self, width: int, height: int) -> None:
        """Dynamically resize cells so the maze fits nicely on screen."""
        max_screen_width = 1200
        max_screen_height = 750
        self.cell_size = min(
            30,
            max_screen_width // width,
            (max_screen_height - 100) // height,
        )
        self.cell_size = max(12, self.cell_size)

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
