import pygame


class Button:
    def __init__(self, x, y, width, height, text, font):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = font

        self.color = (70, 70, 70)
        self.hover_color = (100, 100, 100)
        self.text_color = (255, 255, 255)

    def draw(self, screen):
        mouse_pos = pygame.mouse.get_pos()

        color = self.hover_color if self.rect.collidepoint(mouse_pos) else self.color

        pygame.draw.rect(screen, color, self.rect)

        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def is_clicked(self, input_state):
        return input_state.mouse_clicked and self.rect.collidepoint(
            input_state.mouse_pos
        )
