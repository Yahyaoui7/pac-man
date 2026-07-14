from typing import Optional

from src.UI.button import Button, ButtonManager
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


class VictoryState(State):
    """The game completed victory screen."""

    def __init__(self, game):
        super().__init__(game)
        self.save_score_button: Optional[Button] = Button((0, 0), "Save Score")
        self.home_button: Optional[Button] = Button((0, 0), "Home Menu")
        self.buttons_manager = ButtonManager(
            [
                self.home_button,
                self.save_score_button,
            ]
        )
        self.button_index = 0

    def enter(self):
        self.game.resize_window(1000, 600)
        self.game.sound_manager.play_music("victory_music", loop=False)
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        x, y = w // 2, h // 2

        self.save_score_button.rect.topleft = (x - 100, y + 40)
        self.home_button.rect.topleft = (x - 100, y + 105)

    def activate_selected_button(self):
        if self.button_index == 0:
            from src.graphics.states.home import HomeState

            self.game.state_manager.change_state(HomeState(self.game))

        elif self.button_index == 1:
            from src.graphics.states.name_input import NameInputState

            self.game.state_manager.change_state(NameInputState(self.game))

    def update(self, input_state, events):
        self.button_index, clicked_index = self.buttons_manager.update(
            input_state,
            self.button_index,
        )
        if clicked_index is not None:
            self.button_index = clicked_index
        if input_state.confirm_pressed or clicked_index is not None:
            self.activate_selected_button()

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
