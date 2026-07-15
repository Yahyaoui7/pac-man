from __future__ import annotations

import pygame
from typing import Optional

from src.UI.button import Button
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


class GameOverState(State):
    """The Game Over display screen and Name Input handler."""

    def __init__(self, game, previous_state: State):
        super().__init__(game)
        self.previous_state = previous_state
        self.name_button: Optional[Button] = None
        self.home_button: Optional[Button] = None
        self.hi_score_button: Optional[Button] = None

    def enter(self):
        self.game.sound_manager.play_music("game_over_music", loop=False)
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        center_x, center_y = w // 2, h // 2

        self.name_button = Button(
            (center_x - 100, center_y + 20), "Enter Your Name", ui.FONT_BTN
        )
        self.hi_score_button = Button(
            (center_x - 100, center_y + 85), "Highest Score", ui.FONT_BTN
        )
        self.home_button = Button(
            (center_x - 100, center_y + 150), "Home Menu", ui.FONT_BTN
        )

    def update(self, input_state, events):
        if self.name_button and self.name_button.update(input_state):
            from src.graphics.states.name_input import NameInputState

            self.game.state_manager.change_state(NameInputState(self.game))
        elif self.hi_score_button and self.hi_score_button.update(input_state):
            from src.graphics.states.high_score import HighScoreState

            self.game.state_manager.change_state(
                HighScoreState(self.game, self),
            )
        elif self.home_button and self.home_button.update(input_state):
            from src.graphics.states.home import HomeState

            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        self.previous_state.draw(screen)
        ui.draw_overlay(screen, 150)

        center_x = screen.get_width() // 2
        center_y = screen.get_height() // 2

        # Title
        ui.draw_text_centered(
            screen,
            ui.FONT_TITLE,
            "GAME OVER",
            center_y - 170,
            ui.COLOR_NEON_CYAN,
        )

        # Losing cause
        lives = self.game.lives
        cause = "You were caught by a ghost!" if lives == 0 else "Time's Up!"
        losing_surf = ui.FONT_LOSING.render(cause, True, (255, 80, 80))
        losing_rect = losing_surf.get_rect(center=(center_x, center_y - 125))
        screen.blit(losing_surf, losing_rect)

        # Scores
        score = self.game.score_management.get_score()
        top_scores = self.game.highscore_manager.get_top_scores()
        high_score = top_scores[0]["score"] if top_scores else 0

        score_label = ui.FONT_BTN.render("Your Score:", True, (220, 220, 220))
        score_value = ui.FONT_BTN.render(
            str(score),
            True,
            ui.COLOR_NEON_YELLOW,
        )
        score_label_rect = score_label.get_rect(
            center=(center_x - 40, center_y - 65),
        )
        score_value_rect = score_value.get_rect(
            midleft=(score_label_rect.right + 10, score_label_rect.centery)
        )
        screen.blit(score_label, score_label_rect)
        screen.blit(score_value, score_value_rect)

        high_label = ui.FONT_BTN.render(
            "Highest Score:",
            True,
            (220, 220, 220),
        )
        high_value = ui.FONT_BTN.render(
            str(high_score),
            True,
            ui.COLOR_NEON_YELLOW,
        )
        high_label_rect = high_label.get_rect(
            center=(center_x - 40, center_y - 25),
        )
        high_value_rect = high_value.get_rect(
            midleft=(high_label_rect.right + 10, high_label_rect.centery)
        )
        screen.blit(high_label, high_label_rect)
        screen.blit(high_value, high_value_rect)

        # Buttons
        if self.hi_score_button and self.home_button and self.name_button:
            self.name_button.draw(screen)
            self.hi_score_button.draw(screen)
            self.home_button.draw(screen)
