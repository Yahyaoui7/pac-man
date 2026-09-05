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

    move_up_pressed: bool = False
    move_down_pressed: bool = False
    move_left_pressed: bool = False
    move_right_pressed: bool = False

    confirm_pressed: bool = False
    cancel_pressed: bool = False

    invinciblity: bool = False
    ghost_freez: bool = False
    speed_boost: bool = False
    extra_life: bool = False
    skip_level: bool = False
    ai_player: bool = False
    ghost_hunter: bool = False


class InputManager:

    def __init__(self) -> None:
        self.state = InputState()

    def update(self, events: list[pygame.event.Event]) -> InputState:

        self.state.quit_requested = False
        self.state.pause_pressed = False
        self.state.action_pressed = False
        self.state.mouse_clicked = False
        self.state.move_up_pressed = False
        self.state.move_down_pressed = False
        self.state.move_left_pressed = False
        self.state.move_right_pressed = False

        self.state.confirm_pressed = False
        self.state.cancel_pressed = False
        keys = pygame.key.get_pressed()
        ctrl_pressed = keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]

        self.state.move_up = keys[pygame.K_UP] or (
            keys[pygame.K_w] and not ctrl_pressed
        )
        self.state.move_down = keys[pygame.K_DOWN] or (
            keys[pygame.K_s] and not ctrl_pressed
        )
        self.state.move_left = keys[pygame.K_LEFT] or (
            keys[pygame.K_a] and not ctrl_pressed
        )
        self.state.move_right = keys[pygame.K_RIGHT] or (
            keys[pygame.K_d] and not ctrl_pressed
        )

        self.state.mouse_pos = pygame.mouse.get_pos()
        self.state.mouse_pressed = pygame.mouse.get_pressed()[0]

        self.state.invinciblity = False
        self.state.ghost_freez = False
        self.state.speed_boost = False
        self.state.extra_life = False
        self.state.skip_level = False
        self.state.ai_player = False
        self.state.ghost_hunter = False

        for event in events:

            if event.type == pygame.QUIT or keys[pygame.K_q]:
                self.state.quit_requested = True

            elif event.type == pygame.KEYDOWN:
                ctrl_mask = (
                    pygame.KMOD_CTRL | pygame.KMOD_LCTRL | pygame.KMOD_RCTRL
                )
                is_ctrl = bool(event.mod & ctrl_mask)

                if event.key in (pygame.K_UP, pygame.K_w) and not is_ctrl:
                    self.state.move_up_pressed = True

                elif event.key in (pygame.K_DOWN, pygame.K_s) and not is_ctrl:
                    self.state.move_down_pressed = True

                elif event.key in (pygame.K_LEFT, pygame.K_a) and not is_ctrl:
                    self.state.move_left_pressed = True

                elif event.key in (pygame.K_RIGHT, pygame.K_d) and not is_ctrl:
                    self.state.move_right_pressed = True

                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    self.state.confirm_pressed = True
                    self.state.action_pressed = True

                elif event.key == pygame.K_ESCAPE:
                    self.state.pause_pressed = True
                    self.state.cancel_pressed = True

                elif event.key == pygame.K_a and is_ctrl:
                    self.state.ai_player = True

                elif event.key == pygame.K_k:
                    self.state.skip_level = True
                elif event.key == pygame.K_l:
                    self.state.extra_life = True
                elif event.key == pygame.K_f:
                    self.state.ghost_freez = True
                elif event.key == pygame.K_b:
                    self.state.speed_boost = True
                elif event.key == pygame.K_i:
                    self.state.invinciblity = True
                elif event.key == pygame.K_h:
                    self.state.ghost_hunter = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.state.mouse_clicked = True

        return self.state
