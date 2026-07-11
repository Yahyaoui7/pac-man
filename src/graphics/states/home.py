import pygame
from typing import Any, List

from src.UI.button import Button
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


class HomeState(State):
    """The Main Menu screen state."""

    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.play_button = Button(200, 180, 200, 50, "START GAME", ui.FONT_BTN)
        self.instructions_button = Button(
            200, 245, 200, 50, "INSTRUCTIONS", ui.FONT_BTN
        )
        self.scores_button = Button(200, 310, 200, 50, "HIGHSCORES", ui.FONT_BTN)
        self.quit_button = Button(200, 375, 200, 50, "EXIT", ui.FONT_BTN)

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
        screen.fill(ui.COLOR_BG_DARK)

        title_surf = ui.FONT_TITLE_LARGE.render("PAC-MAN", True, ui.COLOR_NEON_YELLOW)
        title_rect = title_surf.get_rect(center=(300, 80))
        screen.blit(title_surf, title_rect)

        subtitle_surf = ui.FONT_BTN.render(
            "NEON RETRO EDITION", True, ui.COLOR_NEON_CYAN
        )
        subtitle_rect = subtitle_surf.get_rect(center=(300, 125))
        screen.blit(subtitle_surf, subtitle_rect)

        self.play_button.draw(screen)
        self.instructions_button.draw(screen)
        self.scores_button.draw(screen)
        self.quit_button.draw(screen)
