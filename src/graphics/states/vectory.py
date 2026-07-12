import pygame
from typing import Optional

from src.UI.button import Button
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


class VictoryState(State):
    """The game completed victory screen."""

    def __init__(self, game):
        super().__init__(game)
        self.save_score_button: Optional[Button] = None
        self.home_button: Optional[Button] = None

    def enter(self):
        self.game.resize_window(1200, 750)
        self.game.sound_manager.play_music("victory_music", loop=False)
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        x, y = w // 2, h // 2

        self.save_score_button = Button(
            (x - 100, y + 40),
            "Save Score",
        )
        self.home_button = Button(
            (x - 100, y + 105),
            "Home Menu",
        )

    def update(self, input_state, events):
        if self.save_score_button and self.save_score_button.update(input_state):
            from src.graphics.states.name_input import NameInputState

            self.game.state_manager.change_state(NameInputState(self.game))
        elif self.home_button and self.home_button.update(input_state):
            from src.graphics.states.home import HomeState

            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen):
        screen.fill(ui.COLOR_BG_PANEL)

        ui.draw_text_centered(
            screen, ui.FONT_TITLE_LARGE, "YOU WIN!", 130, ui.COLOR_NEON_YELLOW
        )
        ui.draw_text_centered(
            screen,
            ui.FONT_SCORE,
            f"Final Score: {self.game.score_management.get_score()}",
            210,
            ui.COLOR_WHITE,
        )

        if self.save_score_button:
            self.save_score_button.draw(screen)
        if self.home_button:
            self.home_button.draw(screen)
