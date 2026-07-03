import pygame


class Button:

    def __init__(self, x, y, width, height, text, font):

        self.rect = pygame.Rect(x, y, width, height)

        self.text = text
        self.font = font

        self.color = (70, 70, 70)
        self.hover_color = (120, 120, 120)
        self.text_color = (255, 255, 255)

        self.hovered = False

    def update(self, input_state):
        """Returns True once when the button is clicked."""

        self.hovered = self.rect.collidepoint(input_state.mouse_pos)

        return self.hovered and input_state.mouse_clicked

    def draw(self, screen):

        color = self.hover_color if self.hovered else self.color

        pygame.draw.rect(screen, color, self.rect)

        text = self.font.render(self.text, True, self.text_color)
        text_rect = text.get_rect(center=self.rect.center)

        screen.blit(text, text_rect)
