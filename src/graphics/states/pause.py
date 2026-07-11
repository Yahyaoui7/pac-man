from __future__ import annotations

import pygame
from typing import Any, List, Optional

from src.UI.button import Button
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


class PauseState(State):
    """The paused overlay menu screen state."""

    def __init__(self, game: Any, previous_state: PlayingState) -> None:
        super().__init__(game)
        self.previous_state = previous_state
        self.resume_button: Optional[Button] = None
        self.home_button: Optional[Button] = None

    def enter(self) -> None:
        self.game.sound_manager.play_sound("pause")
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        self.resume_button = Button(
            w // 2 - 100, h // 2 - 40, 200, 45, "RESUME", ui.FONT_BTN
        )
        self.home_button = Button(
            w // 2 - 100, h // 2 + 25, 200, 45, "Home MENU", ui.FONT_BTN
        )

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        if input_state.pause_pressed:
            self.game.state_manager.pop_state()
            return
        if self.resume_button and self.resume_button.update(input_state):
            self.game.state_manager.pop_state()
        elif self.home_button and self.home_button.update(input_state):
            from src.graphics.states.home import HomeState
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        self.previous_state.draw(screen)
        ui.draw_overlay(screen, 150)
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
