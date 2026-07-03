"""UI Menu elements like TextInput for player name entry."""

import pygame


class TextInput:
    """A text input field for capturing player names on Game Over/Victory."""

    def __init__(
        self, x: int, y: int, width: int, height: int, font: pygame.font.Font
    ) -> None:
        """Initialize the text input position and state."""
        self.rect: pygame.Rect = pygame.Rect(x, y, width, height)
        self.font: pygame.font.Font = font
        self.text: str = ""
        self.active: bool = True
        self.color_active: tuple[int, int, int] = (255, 238, 0)  # Neon Yellow
        self.color_inactive: tuple[int, int, int] = (100, 100, 100)
        self.text_color: tuple[int, int, int] = (255, 255, 255)

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Process a pygame event. Returns True when ENTER is pressed."""
        if not self.active:
            return False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                # Only allow submit if name is not empty
                if self.text.strip():
                    return True
            else:
                # Alphanumeric and spaces only, max 10 characters
                char = event.unicode
                if (char.isalnum() or char == " ") and len(self.text) < 10:
                    self.text += char

        return False

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the text input field with a glowing border and cursor."""
        color = self.color_active if self.active else self.color_inactive

        # Draw background and border
        pygame.draw.rect(screen, (20, 20, 20), self.rect)
        pygame.draw.rect(screen, color, self.rect, 2)

        # Render the text
        # Add a flashing cursor at the end
        show_cursor = (pygame.time.get_ticks() // 500) % 2 == 0 and self.active
        cursor = "_" if show_cursor else ""
        display_text = self.text + cursor
        text_surface = self.font.render(display_text, True, self.text_color)
        text_rect = text_surface.get_rect(midleft=(self.rect.x + 10, self.rect.centery))

        screen.blit(text_surface, text_rect)
