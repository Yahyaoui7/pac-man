import pygame
from typing import Any, List

from src.UI.button import Button, ButtonManager
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui
from src.logic.helpers import screen_center


class HomeState(State):
    """The Main Menu screen state."""

    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.play_button = Button((0, 0), "START GAME")
        self.instructions_button = Button((0, 0), "INSTRUCTIONS")
        self.scores_button = Button((0, 0), "HIGHSCORES")
        self.quit_button = Button((0, 0), "EXIT")
        self.button_index = 0
        self.buttons = [
            self.play_button,
            self.instructions_button,
            self.scores_button,
            self.quit_button,
        ]
        self.button_manager = ButtonManager(self.buttons)

    def enter(self) -> None:
        self.game.resize_window(1000, 600)
        self.game.sound_manager.play_music("menu_intro", False)
        self.game.lives = self.game.config.lives

    def activate_selected_button(self) -> None:

        if self.button_index == 0:

            self.game.score_management.reset()
            self.game.level_manager.current_level_index = 0
            self.game.curr_level = self.game.config.levels[0]
            self.game.curr_level.height = min(self.game.curr_level.height, 32)
            self.game.curr_level.width = min(self.game.curr_level.width, 60)
            self.game.sound_manager.play_music("game_intro", False)

            from src.graphics.states.playing import PlayingState

            self.game.state_manager.change_state(PlayingState(self.game))

        elif self.button_index == 1:
            from src.graphics.states.instructions import InstructionsState

            self.game.state_manager.change_state(InstructionsState(self.game))

        elif self.button_index == 2:
            from src.graphics.states.high_score import HighScoreState

            self.game.state_manager.change_state(
                HighScoreState(
                    self.game,
                    self,
                )
            )

        elif self.button_index == 3:
            self.game.running = False

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        self.button_index, clicked_index = self.button_manager.update(
            input_state,
            self.button_index,
        )
        if clicked_index is not None:
            self.button_index = clicked_index
        if input_state.confirm_pressed or clicked_index is not None:
            self.activate_selected_button()
        if not pygame.mixer.music.get_busy():
            self.game.sound_manager.play_music("menu_music")

    def layout_buttons(self, x: int, y: int) -> None:

        self.play_button.rect.topleft = (x - 100, y - 100)
        self.instructions_button.rect.topleft = (x - 100, y - 30)
        self.scores_button.rect.topleft = (x - 100, y + 40)
        self.quit_button.rect.topleft = (x - 100, y + 110)

    def draw(self, screen: pygame.Surface) -> None:
        x, y = screen_center(screen.get_width(), screen.get_height())

        self.layout_buttons(x, y)
        screen.fill(ui.COLOR_BG_DARK)

        assert ui.FONT_TITLE_LARGE is not None
        assert ui.FONT_BTN is not None

        title_surf = ui.FONT_TITLE_LARGE.render(
            "PAC-MAN",
            True,
            ui.COLOR_NEON_YELLOW,
        )
        title_rect = title_surf.get_rect(center=(x, y - 200))
        screen.blit(title_surf, title_rect)

        subtitle_surf = ui.FONT_BTN.render(
            "NEON RETRO EDITION", True, ui.COLOR_NEON_CYAN
        )
        subtitle_rect = subtitle_surf.get_rect(center=(x, y - 150))
        screen.blit(subtitle_surf, subtitle_rect)

        self.play_button.draw(screen)
        self.instructions_button.draw(screen)
        self.scores_button.draw(screen)
        self.quit_button.draw(screen)
