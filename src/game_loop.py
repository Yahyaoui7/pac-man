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


from typing import Any

from src.graphics.entitys.entity_manager import EntityManager
from src.graphics.renderer import StateManager
from src.graphics.states.home import HomeState
from src.graphics.ui_helpers import _init_fonts
from src.logic.level_manager import LevelManager
from src.sounds.soud_manager import SoundManager
from .logic.inputmanager import InputManager
from .logic.config import CELL_SIZE, GameConfig
from .logic.score import ScoreManager, HighScoreManager

import pygame
import sys
import os

sys.setrecursionlimit(99999999)


class GameStarter:

    def __init__(self, config: GameConfig):
        self.running = True
        self.config = config
        self.screen: Any = None

        self.state_manager = StateManager(self)
        self.level_manager = LevelManager(config)
        self.sound_manager = SoundManager()
        self.score_management = ScoreManager(config)
        self.highscore_manager = HighScoreManager(".highscores.json")
        self.entity_manager: Any = None

        self.lives: int = config.lives

    def resize_window(self, width: int, height: int) -> None:
        """Resize the window dynamically if width/height changed."""
        if self.screen is None or self.screen.get_size() != (width, height):
            os.environ["SDL_VIDEO_WINDOW_POS"] = "center"
            self.screen = pygame.display.set_mode((width, height))

    def recalculate_cell_size(self, width: int, height: int) -> None:
        """Dynamically resize cells so the maze fits nicely on screen."""
        max_screen_width = 1000
        max_screen_height = 600
        self.cell_size = min(
            CELL_SIZE,
            max_screen_width // width,
            (max_screen_height - 100) // height,
        )
        self.cell_size = max(12, self.cell_size)

    def run(self) -> None:
        pygame.init()
        _init_fonts()

        self.screen = pygame.display.set_mode((1000, 600))
        pygame.display.set_caption("NEON PAC-MAN")

        self.entity_manager = EntityManager(
            self.config,
            self.score_management,
            self,
        )

        clock = pygame.time.Clock()
        input_manager = InputManager()

        self.state_manager.change_state(HomeState(self))

        while self.running:

            events = pygame.event.get()

            input_state = input_manager.update(events)

            if input_state.quit_requested:
                self.running = False

            self.state_manager.update(input_state, events)

            if self.screen:
                self.screen.fill((0, 0, 0))
                self.state_manager.draw(self.screen)
                pygame.display.flip()
            clock.tick(60)
        pygame.quit()
