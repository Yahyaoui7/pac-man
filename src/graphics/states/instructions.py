import pygame
from typing import Any, List

from src.graphics.UI.button import Button
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui
from src.graphics.states.home import HomeState
from src.logic.helpers import screen_center


class InstructionsState(State):
    """The game rules and controls instructions screen."""

    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.back_button = Button((0, 0), "BACK")

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:

        if (
            self.back_button
            and self.back_button.update(input_state)
            or self.back_button
            and input_state.confirm_pressed
            or self.back_button
            and input_state.pause_pressed
        ):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(ui.COLOR_BG_DARK)

        screen_w = screen.get_width()
        screen_h = screen.get_height()
        x, y = screen_center(screen_w, screen_h)

        panel_width = 600
        panel_left = x - (panel_width // 2)
        panel_right = x + (panel_width // 2)

        pygame.draw.rect(
            screen,
            ui.COLOR_WHITE,
            screen.get_rect().inflate(-400, -20),
            width=2,
            border_radius=12,
        )

        assert ui.FONT_TITLE is not None
        assert ui.FONT_TEXT is not None

        title_surf = ui.FONT_TITLE.render(
            "HOW TO PLAY",
            True,
            ui.COLOR_NEON_CYAN,
        )
        screen.blit(title_surf, (x - title_surf.get_width() // 2, 45))

        controls_start_x = panel_left + 60
        bubbles_start_x = panel_left + 170
        controls_y = 120

        screen.blit(
            ui.FONT_TEXT.render("Move:", True, ui.COLOR_WHITE),
            (controls_start_x, controls_y),
        )
        movement_keys = ["UP", "LEFT", "DOWN", "RIGHT"]
        for i, key in enumerate(movement_keys):
            bubble = ui.draw_bubble(key)
            screen.blit(bubble, (bubbles_start_x + i * 80, controls_y - 8))

        controls_y += 50
        screen.blit(
            ui.FONT_TEXT.render("or:", True, ui.COLOR_WHITE),
            (controls_start_x + 30, controls_y),
        )
        wasd_keys = ["W", "A", "S", "D"]
        for i, key in enumerate(wasd_keys):
            bubble = ui.draw_bubble(key)
            screen.blit(bubble, (bubbles_start_x + i * 70, controls_y - 8))

        controls_y += 50
        screen.blit(
            ui.FONT_TEXT.render("Pause:", True, ui.COLOR_WHITE),
            (controls_start_x, controls_y),
        )
        screen.blit(ui.draw_bubble("ESC"), (bubbles_start_x, controls_y - 8))

        pygame.draw.line(
            screen,
            (50, 50, 50),
            (panel_left + 40, controls_y + 55),
            (panel_right - 40, controls_y + 55),
            1,
        )

        cheats = [
            ("I", "Invincibility"),
            ("F", "Freeze Ghosts"),
            ("B", "Speed Boost"),
            ("L", "Extra Life"),
            ("K", "Skip Level"),
            ("H", "Ghost Hunter"),
        ]

        start_x = panel_left + 50
        start_y = 335
        row_gap = 48
        col_gap = 270
        for i, (key, desc) in enumerate(cheats):
            col = i % 2
            row = i // 2

            px = start_x + col * col_gap
            py = start_y + row * row_gap

            screen.blit(ui.draw_bubble(key), (px, py))
            screen.blit(
                ui.FONT_TEXT.render(desc, True, ui.COLOR_WHITE),
                (px + 70, py + 8),
            )

        # Back button
        self.back_button.rect.midbottom = (x, screen_h - 45)
        self.back_button.draw(screen)
