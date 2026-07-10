from dataclasses import dataclass
import pygame


@dataclass
class InputState:
    quit_requested: bool = False

    move_up: bool = False
    move_down: bool = False
    move_left: bool = False
    move_right: bool = False

    pause_pressed: bool = False
    action_pressed: bool = False

    mouse_pos: tuple[int, int] = (0, 0)
    mouse_pressed: bool = False
    mouse_clicked: bool = False


class InputManager:

    def __init__(self) -> None:
        self.state = InputState()

    def update(self, events: list[pygame.event.Event]) -> InputState:

        self.state.quit_requested = False
        self.state.pause_pressed = False
        self.state.action_pressed = False
        self.state.mouse_clicked = False

        keys = pygame.key.get_pressed()

        self.state.move_up = keys[pygame.K_UP]
        self.state.move_down = keys[pygame.K_DOWN]
        self.state.move_left = keys[pygame.K_LEFT]
        self.state.move_right = keys[pygame.K_RIGHT]

        self.state.mouse_pos = pygame.mouse.get_pos()
        self.state.mouse_pressed = pygame.mouse.get_pressed()[0]

        for event in events:

            if event.type == pygame.QUIT or keys[pygame.K_q]:
                self.state.quit_requested = True

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    self.state.pause_pressed = True
                elif event.key == pygame.K_SPACE:
                    self.state.action_pressed = True

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.state.mouse_clicked = True

        return self.state
