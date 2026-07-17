from typing import Any
import pygame

from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


class NameInputState(State):
    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.player_name = ""
        self.final_score = 0
        self.title_text = "ENTER YOUR NAME"
        self.max_name_length = 10

    def enter(self) -> None:
        self.player_name = ""
        self.final_score = self.game.score_management.get_score()

    def update(
        self, input_state: Any, events: list[pygame.event.Event]
    ) -> None:
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.player_name = self.player_name[:-1]
                elif event.key == pygame.K_RETURN:
                    self.confirm_name()
                elif event.unicode.isalnum():
                    if len(self.player_name) < self.max_name_length:
                        self.player_name += event.unicode

    def confirm_name(self) -> None:
        self.game.highscore_manager.add_score(
            self.player_name,
            self.final_score,
        )
        from src.graphics.states.high_score import HighScoreState

        self.game.state_manager.change_state(HighScoreState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(ui.COLOR_BG_PANEL)

        assert ui.FONT_TITLE is not None
        assert ui.FONT_INPUT is not None

        ui.draw_text_centered(
            screen, ui.FONT_TITLE, self.title_text, 120, ui.COLOR_WHITE
        )
        ui.draw_text_centered(
            screen,
            ui.FONT_INPUT,
            f"Final Score: {self.final_score}",
            190,
            ui.COLOR_NEON_YELLOW,
        )

        # Name input
        input_surf = ui.FONT_INPUT.render(
            self.player_name + "_", True, ui.COLOR_NEON_CYAN
        )
        input_rect = input_surf.get_rect(center=(screen.get_width() // 2, 260))
        screen.blit(input_surf, input_rect)

        ui.draw_text_centered(
            screen,
            ui.FONT_INPUT,
            "Press ENTER to save",
            330,
            (128, 128, 128),
        )
