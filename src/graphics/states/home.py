import pygame
from typing import Any, List

from src.UI.button import Button
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui
from src.logic.helpers import screen_center


class HomeState(State):
    """The Main Menu screen state."""

    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.play_button = Button((0, 0), "hold")
        self.instructions_button = Button((0, 0), "hold")
        self.scores_button = Button((0, 0), "hold")
        self.quit_button = Button((0, 0), "hold")

    def enter(self) -> None:
        self.game.resize_window(1200, 750)
        self.game.sound_manager.play_music("menu_intro", False)
        self.game.lives = self.game.config.lives

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        if not pygame.mixer.music.get_busy():
            self.game.sound_manager.play_music("menu_music")
        if self.play_button.update(input_state):
            self.game.score_management.reset()
            self.game.level_manager.current_level_index = 0
            self.game.curr_level = self.game.config.levels[0]
            self.game.curr_level.height = min(self.game.curr_level.height, 32)
            self.game.curr_level.width = min(self.game.curr_level.width, 60)
            self.game.sound_manager.play_music("game_intro", False)
            from src.graphics.states.playing import PlayingState

            self.game.state_manager.change_state(PlayingState(self.game))

        elif self.instructions_button.update(input_state):
            from src.graphics.states.instructions import InstructionsState

            self.game.state_manager.change_state(InstructionsState(self.game))

        elif self.scores_button.update(input_state):
            from src.graphics.states.high_score import HighScoreState

            self.game.state_manager.change_state(
                HighScoreState(self.game, self),
            )

        elif self.quit_button.update(input_state):
            self.game.running = False

    def draw(self, screen: pygame.Surface) -> None:
        x, y = screen_center(screen.get_width(), screen.get_height())

        self.play_button = Button(
            (x - 100, y - 200),
            "START GAME",
        )
        self.instructions_button = Button(
            (x - 100, y - 130),
            "INSTRUCTIONS",
        )
        self.scores_button = Button(
            (x - 100, y - 60),
            "HIGHSCORES",
        )
        self.quit_button = Button(
            (x - 100, y + 10),
            "EXIT",
        )

        screen.fill(ui.COLOR_BG_DARK)

        title_surf = ui.FONT_TITLE_LARGE.render(
            "PAC-MAN",
            True,
            ui.COLOR_NEON_YELLOW,
        )
        title_rect = title_surf.get_rect(center=(x, y - 300))
        screen.blit(title_surf, title_rect)

        subtitle_surf = ui.FONT_BTN.render(
            "NEON RETRO EDITION", True, ui.COLOR_NEON_CYAN
        )
        subtitle_rect = subtitle_surf.get_rect(center=(x, y - 250))
        screen.blit(subtitle_surf, subtitle_rect)

        self.play_button.draw(screen)
        self.instructions_button.draw(screen)
        self.scores_button.draw(screen)
        self.quit_button.draw(screen)
