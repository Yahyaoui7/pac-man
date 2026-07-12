import pygame
from typing import Any, List, Optional

from src.UI.button import Button
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


class HighScoreState(State):
    """The top 10 highscores display screen."""

    def __init__(self, game: Any, previous_state=None) -> None:
        super().__init__(game)
        self.previous_state = previous_state
        self.home_button: Optional[Button] = None

    def enter(self) -> None:
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        self.home_button = Button((w // 2 - 100, h - 90), "Home MENU", ui.FONT_BTN)

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        if self.home_button and self.home_button.update(input_state):
            from src.graphics.states.home import HomeState

            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        if self.previous_state:
            self.previous_state.draw(screen)
        else:
            screen.fill(ui.COLOR_BG_PANEL)

        ui.draw_overlay(screen, 180)
        ui.draw_text_centered(
            screen, ui.FONT_TITLE, "HIGH SCORES", 90, ui.COLOR_NEON_CYAN
        )

        highscores = self.game.highscore_manager.get_top_scores()

        if not highscores:
            ui.draw_text_centered(
                screen, ui.FONT_SCORE, "No scores yet", 180, ui.COLOR_WHITE
            )
        else:
            start_y = 150
            for index, item in enumerate(highscores):
                score_text = f"{index + 1}. {item['name']}  -  {item['score']}"
                score_surf = ui.FONT_SCORE.render(score_text, True, ui.COLOR_WHITE)
                score_rect = score_surf.get_rect(
                    center=(screen.get_width() // 2, start_y + index * 35)
                )
                screen.blit(score_surf, score_rect)

        if self.home_button:
            self.home_button.draw(screen)
