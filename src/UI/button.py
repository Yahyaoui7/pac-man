from typing import List, Optional, Tuple, Any

import pygame

from src.logic.config import BUTTON_SIZE
from src.graphics import ui_helpers as ui

button_font: Optional[pygame.font.Font] = None


def get_button_font() -> pygame.font.Font:
    global button_font
    if button_font is None:
        button_font = pygame.font.Font(None, 36)
    return button_font


class Button:

    def __init__(
        self,
        cord: Tuple[int, int],
        text: str,
        font: Optional[pygame.font.Font] = None,
        size: Tuple[int, int] = BUTTON_SIZE,
    ) -> None:
        if font is None:
            font = get_button_font()
        width, height = size
        x, y = cord
        self.rect = pygame.Rect(x, y, width, height)

        self.text = text
        self.font = font

        self.color = (70, 70, 70)
        self.hover_color = (120, 120, 120)
        self.text_color = (255, 255, 255)

        self.hovered = False
        self.selected = False

    def update(self, input_state: Any) -> bool:
        """Returns True once when the button is clicked."""
        self.hovered = self.rect.collidepoint(input_state.mouse_pos)

        return bool(self.hovered and input_state.mouse_clicked)

    def draw(self, screen: pygame.Surface) -> None:

        fill = (30, 30, 40)

        active = self.hovered or self.selected

        border = ui.COLOR_NEON_YELLOW if active else ui.COLOR_NEON_CYAN
        pygame.draw.rect(
            screen,
            fill,
            self.rect,
            border_radius=10,
        )

        pygame.draw.rect(
            screen,
            border,
            self.rect,
            width=5,
            border_radius=10,
        )

        text = self.font.render(self.text, True, self.text_color)
        text_rect = text.get_rect(center=self.rect.center)

        screen.blit(text, text_rect)


class ButtonManager:
    def __init__(self, buttons: List[Button]) -> None:
        self.buttons = buttons

    def update(
        self, input_state: Any, button_index: int
    ) -> Tuple[int, Optional[int]]:
        """Update selection from keyboard/mouse input.

        Returns (button_index, clicked_index):
          - button_index: the currently selected button's index (keyboard
            navigation wraps around; a mouse hover moves selection to that
            button so the two stay in sync).
          - clicked_index: index of the button clicked this frame, or None.
        """
        num_buttons = len(self.buttons)

        if input_state.move_up_pressed:
            button_index = (button_index - 1) % num_buttons
        elif input_state.move_down_pressed:
            button_index = (button_index + 1) % num_buttons

        # Hovering a button with the mouse takes over keyboard selection,
        # so the next arrow-key press continues from that button.
        for i, button in enumerate(self.buttons):
            if button.rect.collidepoint(input_state.mouse_pos):
                button_index = i
                break

        clicked_index = None
        for i, button in enumerate(self.buttons):
            button.selected = i == button_index
            if button.update(input_state):
                clicked_index = i

        return button_index, clicked_index
