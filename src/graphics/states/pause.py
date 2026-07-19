from __future__ import annotations

import pygame
from typing import Any, List

from src.graphics.UI.button import Button, ButtonManager
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


class PauseState(State):
    """The paused overlay menu screen state."""

    def __init__(self, game: Any, previous_state: State) -> None:
        super().__init__(game)
        self.previous_state = previous_state
        self.resume_button: Button = Button((0, 0), "RESUME")
        self.home_button: Button = Button((0, 0), "Home MENU")
        self.buttons_manager = ButtonManager(
            [
                self.resume_button,
                self.home_button,
            ]
        )
        self.button_index = 0

    def enter(self) -> None:
        self.game.sound_manager.play_sound("pause")

        w = self.game.screen.get_width()
        h = self.game.screen.get_height()

        self.resume_button.rect.topleft = (w // 2 - 100, h // 2 - 40)
        self.home_button.rect.topleft = (w // 2 - 100, h // 2 + 25)

    def activate_selected_button(self) -> None:
        if self.button_index == 0:
            self.game.state_manager.pop_state()

        elif self.button_index == 1:

            from src.graphics.states.home import HomeState

            self.game.state_manager.change_state(HomeState(self.game))

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        self.button_index, clicked_index = self.buttons_manager.update(
            input_state,
            self.button_index,
        )
        if clicked_index is not None:
            self.button_index = clicked_index
        if input_state.confirm_pressed or clicked_index is not None:
            self.activate_selected_button()

    def draw(self, screen: pygame.Surface) -> None:
        self.previous_state.draw(screen)
        ui.draw_overlay(screen, 150)
        assert ui.FONT_TITLE is not None
        ui.draw_text_centered(
            screen,
            ui.FONT_TITLE,
            "GAME PAUSED",
            screen.get_height() // 2 - 100,
            ui.COLOR_NEON_CYAN,
        )
        if self.resume_button and self.home_button:
            self.resume_button.draw(screen)
            self.home_button.draw(screen)
