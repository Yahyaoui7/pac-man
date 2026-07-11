"""Implements the State Pattern for game screens and UI rendering."""

import math
from src.logic.utils import expired, after

import pygame
import pygame.draw as dr
from typing import Any, List, Optional
from src.UI.button import Button


from src.logic.config import CELL_SIZE, PADDING
from src.logic.config import EAST, NORTH, SOUTH, WEST
from src.logic.helpers import cell_to_screen, pixel_to_screen
from src.logic.movement import MovementSystem
from src.graphics import ui_helpers as ui


class State:
    """Base class for all screen states."""

    def __init__(self, game: Any) -> None:
        self.game = game

    def enter(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        pass

    def draw(self, screen: pygame.Surface) -> None:
        pass


class StateManager:
    """Manages switching and updating the active screen state."""

    def __init__(self, game: Any) -> None:
        self.game = game
        self.stack: list[State] = []

    @property
    def current(self):
        if self.stack:
            return self.stack[-1]
        return None

    def change_state(self, state: State):
        while self.stack:
            self.stack.pop().exit()
        self.stack.append(state)
        state.enter()

    def push_state(self, state: State) -> None:
        self.stack.append(state)
        state.enter()

    def pop_state(self) -> None:
        if not self.stack:
            return
        state = self.stack.pop()
        state.exit()

    def update(self, input_state, events):
        if self.current:
            self.current.update(input_state, events)

    def draw(self, screen: pygame.Surface) -> None:
        if self.current:
            self.current.draw(screen)


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
            self.game.state_manager.change_state(PlayingState(self.game))

        elif self.instructions_button.update(input_state):
            self.game.state_manager.change_state(InstructionsState(self.game))

        elif self.scores_button.update(input_state):
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


class InstructionsState(State):
    """The game rules and controls instructions screen."""

    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.back_button = Button(200, 420, 200, 45, "BACK", ui.FONT_BTN)

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        if self.back_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill(ui.COLOR_BG_DARK)

        title_surf = ui.FONT_TITLE.render("HOW TO PLAY", True, ui.COLOR_NEON_CYAN)
        screen.blit(title_surf, (200, 40))

        lines = [
            "- Use ARROWS or WASD keys to move Pacman.",
            "- Eat all Pacgums (small dots) to clear the level.",
            "- Eat Super Pacgums (corner pellets) to make ghosts edible.",
            "- Some Super Pacgums grant special abilities!",
            "- Press SPACE to use your ability (Punch or Kick).",
            "- Avoid Ghosts. If they touch you, you lose a life.",
            "- Press ESC to pause the game.",
            "",
            "--- CHEAT MODES (For Peer Review) ---",
            "- Press [ I ] to Toggle Invincibility (No life lost)",
            "- Press [ F ] to Toggle Ghost Freeze (Stop ghosts)",
            "- Press [ S ] to Toggle Speed Boost (Double speed)",
            "- Press [ L ] to Add an Extra Life",
            "- Press [ K ] to Skip current level instantly",
        ]

        y_offset = 110
        for line in lines:
            color = (
                ui.COLOR_NEON_YELLOW
                if "CHEAT" in line or line.startswith("- Press")
                else ui.COLOR_WHITE
            )
            line_surf = ui.FONT_TEXT.render(line, True, color)
            screen.blit(line_surf, (50, y_offset))
            y_offset += 24

        self.back_button.draw(screen)


class PlayingState(State):
    """The active gameplay state handling movements, collisions, timers."""

    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.player_invincible_until = 0
        self.msg_timer: float = 0.0
        self.msg_text: str = ""

    def enter(self) -> None:
        self.game.level_manager.load_level(
            self.game.level_manager.current_level_index,
        )
        self.maze = self.game.level_manager.current_maze.maze
        self.game.entity_manager.load_level_entities(self.maze)

        width = self.game.level_manager.get_current_level_config().width
        height = self.game.level_manager.get_current_level_config().height

        self.game.recalculate_cell_size(width, height)
        self.game.resize_window(
            width * CELL_SIZE + PADDING,
            height * CELL_SIZE + PADDING + 60,
        )

        curr_idx = self.game.level_manager.current_level_index
        self.movement = MovementSystem(self.maze)
        self.msg_text = f"LEVEL {curr_idx + 1}"
        self.msg_timer = 2.0

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        if not pygame.mixer.music.get_busy():
            self.game.sound_manager.play_music("game_music", False)

        if self.game.lives <= 0:
            self.game.state_manager.change_state(
                GameOverState(self.game, self),
            )
            return
        if input_state.pause_pressed:
            self.game.state_manager.push_state(PauseState(self.game, self))
            return

        self._handle_input(input_state)
        self._update_entities()
        self._check_level_end()

    def _handle_input(self, input_state) -> None:
        player = self.game.entity_manager.player
        if input_state.move_left:
            player.next_direction = "LEFT"
        elif input_state.move_right:
            player.next_direction = "RIGHT"
        elif input_state.move_up:
            player.next_direction = "UP"
        elif input_state.move_down:
            player.next_direction = "DOWN"
        elif input_state.skip_level:
            self.game.entity_manager.total_pellets = 0
        if input_state.action_pressed:
            player.use_ability()

    def _update_entities(self) -> None:
        em = self.game.entity_manager
        self.movement.update_entity(em.player)

        for ghost in em.ghosts:
            if ghost.is_eaten:
                self.movement.update_ghost_to_target(
                    ghost,
                    ghost.spawn_y,
                    ghost.spawn_x,
                )
            elif ghost.is_edible:
                self.movement.update_runaway_ghost(ghost, em.player)
            else:
                self.movement.update_bfs_ghost(ghost, em.player)

        self.check_collision(em.player, em.ghosts)
        em.update(self.maze, 1 / 60.0)

    def _check_level_end(self) -> None:
        self.game.level_manager.update_time(1 / 60.0)
        if self.game.level_manager.is_time_out():
            if self.game.lives <= 0:
                self.game.state_manager.change_state(GameOverState(self.game, self))
            else:
                level_cfg = self.game.level_manager.get_current_level_config()
                self.game.level_manager.remaining_time = float(
                    level_cfg.level_max_time,
                )
                self.game.entity_manager.reset_positions()
                self.msg_text = "TIME'S UP! TRY AGAIN"
                self.msg_timer = 2.0
            return

        if self.game.entity_manager.total_pellets <= 0:
            self.game.sound_manager.play_sound("level_complete")
            self.game.score_management.add_time_bonus(
                int(self.game.level_manager.remaining_time)
            )
            next_lvl = self.game.level_manager.current_level_index + 1
            if next_lvl >= len(self.game.config.levels):
                self.game.state_manager.change_state(VictoryState(self.game))
            else:
                self.game.level_manager.current_level_index = next_lvl
                self.game.state_manager.change_state(PlayingState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        # HUD bar
        pygame.draw.rect(screen, ui.COLOR_BG_PANEL, (0, 0, screen.get_width(), 40))
        pygame.draw.line(
            screen, ui.COLOR_NEON_CYAN, (0, 40), (screen.get_width(), 40), 2
        )

        score_surf = ui.FONT_HUD.render(
            f"SCORE: {self.game.score_management.get_score()}",
            True,
            ui.COLOR_NEON_YELLOW,
        )
        lvl_num = self.game.level_manager.current_level_index + 1
        level_surf = ui.FONT_HUD.render(f"LEVEL: {lvl_num}", True, ui.COLOR_WHITE)
        lives_surf = ui.FONT_HUD.render(f"LIVES: {self.game.lives}", True, ui.COLOR_RED)
        time_rem = max(0, int(self.game.level_manager.remaining_time))
        time_surf = ui.FONT_HUD.render(f"TIME: {time_rem}s", True, ui.COLOR_GREEN)

        screen.blit(score_surf, (15, 10))
        screen.blit(level_surf, (screen.get_width() // 3, 10))
        screen.blit(lives_surf, (2 * screen.get_width() // 3, 10))
        screen.blit(time_surf, (screen.get_width() - 110, 10))

        # Floating message
        if self.msg_timer > 0:
            self.msg_timer -= 1 / 60
        else:
            self.msg_text = ""

        player = self.game.entity_manager.player
        if self.msg_timer > 0:
            px, py = pixel_to_screen(player.x, player.y)
            text_surface = ui.FONT_HUD.render(self.msg_text, True, ui.COLOR_WHITE)
            screen.blit(
                text_surface,
                (px - text_surface.get_width() // 2, py - 40),
            )

        # Maze walls
        c = CELL_SIZE
        for row, cells in enumerate(self.maze):
            for col, cell in enumerate(cells):
                x, y = cell_to_screen(row, col)

                if cell & NORTH:
                    dr.line(screen, "blue", (x, y), (x + c, y), 2)
                if cell & EAST:
                    dr.line(screen, "blue", (x + c, y), (x + c, y + c), 2)
                if cell & SOUTH:
                    dr.line(screen, "blue", (x, y + c), (x + c, y + c), 2)
                if cell & WEST:
                    dr.line(screen, "blue", (x, y), (x, y + c), 2)

        self.game.entity_manager.draw(screen)

    def check_collision(self, player, ghosts):
        radius = CELL_SIZE // 3

        for ghost in ghosts:
            dx = player.x - ghost.x
            dy = player.y - ghost.y
            distance = math.hypot(dx, dy)

            if distance <= radius * 2 and not ghost.is_eaten:
                if ghost.is_edible:
                    ghost.is_eaten = True
                    self.game.sound_manager.play_sound("eat_ghost")
                    self.game.score_management.add_ghost()
                    self.msg_text = f"+{self.game.config.points_per_ghost}"
                    self.msg_timer = 2
                    player.trigger_attack()
                else:
                    if expired(self.player_invincible_until):
                        if self.game.lives > 1:
                            self.game.sound_manager.play_sound("player_death")
                        self.game.lives -= 1
                        self.player_invincible_until = after(1500)
                        player.reset_location()
                        self.msg_text = "Be careful!"
                        self.msg_timer = 1.0


class PauseState(State):
    """The paused overlay menu screen state."""

    def __init__(self, game: Any, previous_state: PlayingState) -> None:
        super().__init__(game)
        self.previous_state = previous_state
        self.resume_button: Optional[Button] = None
        self.home_button: Optional[Button] = None

    def enter(self) -> None:
        self.game.sound_manager.play_sound("pause")
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        self.resume_button = Button(
            w // 2 - 100, h // 2 - 40, 200, 45, "RESUME", ui.FONT_BTN
        )
        self.home_button = Button(
            w // 2 - 100, h // 2 + 25, 200, 45, "Home MENU", ui.FONT_BTN
        )

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        if input_state.pause_pressed:
            self.game.state_manager.pop_state()
            return
        if self.resume_button and self.resume_button.update(input_state):
            self.game.state_manager.pop_state()
        elif self.home_button and self.home_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        self.previous_state.draw(screen)
        ui.draw_overlay(screen, 150)
        ui.draw_text_centered(
            screen,
            ui.FONT_TITLE,
            "GAME PAUSED",
            screen.get_height() // 2 - 100,
            ui.COLOR_NEON_CYAN,
        )
        if self.resume_button and self.home_button:
            self.resume_button.draw(screen)
            self.home_button.draw(screen)


class GameOverState(State):
    """The Game Over display screen and Name Input handler."""

    def __init__(self, game, previous_state: PlayingState):
        super().__init__(game)
        self.previous_state = previous_state
        self.name_button: Optional[Button] = None
        self.home_button: Optional[Button] = None
        self.hi_score_button: Optional[Button] = None

    def enter(self):
        self.game.sound_manager.play_music("game_over_music", loop=False)
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        center_x, center_y = w // 2, h // 2

        self.name_button = Button(
            center_x - 100, center_y + 20, 200, 45, "Enter Your Name", ui.FONT_BTN
        )
        self.hi_score_button = Button(
            center_x - 100, center_y + 85, 200, 45, "Highest Score", ui.FONT_BTN
        )
        self.home_button = Button(
            center_x - 100, center_y + 150, 200, 45, "Home Menu", ui.FONT_BTN
        )

    def update(self, input_state, events):
        if self.name_button and self.name_button.update(input_state):
            self.game.state_manager.change_state(NameInputState(self.game))
        elif self.hi_score_button and self.hi_score_button.update(input_state):
            self.game.state_manager.change_state(HighScoreState(self.game, self))
        elif self.home_button and self.home_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        self.previous_state.draw(screen)
        ui.draw_overlay(screen, 150)

        center_x = screen.get_width() // 2
        center_y = screen.get_height() // 2

        # Title
        ui.draw_text_centered(
            screen, ui.FONT_TITLE, "GAME OVER", center_y - 170, ui.COLOR_NEON_CYAN
        )

        # Losing cause
        lives = self.game.lives
        cause = "You were caught by a ghost!" if lives == 0 else "Time's Up!"
        losing_surf = ui.FONT_LOSING.render(cause, True, (255, 80, 80))
        losing_rect = losing_surf.get_rect(center=(center_x, center_y - 125))
        screen.blit(losing_surf, losing_rect)

        # Scores
        score = self.game.score_management.get_score()
        top_scores = self.game.highscore_manager.get_top_scores()
        high_score = top_scores[0]["score"] if top_scores else 0

        score_label = ui.FONT_BTN.render("Your Score:", True, (220, 220, 220))
        score_value = ui.FONT_BTN.render(str(score), True, ui.COLOR_NEON_YELLOW)
        score_label_rect = score_label.get_rect(center=(center_x - 40, center_y - 65))
        score_value_rect = score_value.get_rect(
            midleft=(score_label_rect.right + 10, score_label_rect.centery)
        )
        screen.blit(score_label, score_label_rect)
        screen.blit(score_value, score_value_rect)

        high_label = ui.FONT_BTN.render("Highest Score:", True, (220, 220, 220))
        high_value = ui.FONT_BTN.render(str(high_score), True, ui.COLOR_NEON_YELLOW)
        high_label_rect = high_label.get_rect(center=(center_x - 40, center_y - 25))
        high_value_rect = high_value.get_rect(
            midleft=(high_label_rect.right + 10, high_label_rect.centery)
        )
        screen.blit(high_label, high_label_rect)
        screen.blit(high_value, high_value_rect)

        # Buttons
        if self.hi_score_button and self.home_button and self.name_button:
            self.name_button.draw(screen)
            self.hi_score_button.draw(screen)
            self.home_button.draw(screen)


class HighScoreState(State):
    """The top 10 highscores display screen."""

    def __init__(self, game: Any, previous_state=None) -> None:
        super().__init__(game)
        self.previous_state = previous_state
        self.home_button: Optional[Button] = None

    def enter(self) -> None:
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        self.home_button = Button(
            w // 2 - 100, h - 90, 200, 45, "Home MENU", ui.FONT_BTN
        )

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        if self.home_button and self.home_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        if self.previous_state:
            self.previous_state.draw(screen)
        else:
            screen.fill(ui.COLOR_BG_PANEL)

        ui.draw_overlay(screen, 180)
        ui.draw_text_centered(
            screen, ui.FONT_TITLE, "HIGH SCORES", 90, ui.COLOR_NEON_CYAN
        )

        highscores = self.game.highscore_manager.get_top_scores()

        if not highscores:
            ui.draw_text_centered(
                screen, ui.FONT_SCORE, "No scores yet", 180, ui.COLOR_WHITE
            )
        else:
            start_y = 150
            for index, item in enumerate(highscores):
                score_text = f"{index + 1}. {item['name']}  -  {item['score']}"
                score_surf = ui.FONT_SCORE.render(score_text, True, ui.COLOR_WHITE)
                score_rect = score_surf.get_rect(
                    center=(screen.get_width() // 2, start_y + index * 35)
                )
                screen.blit(score_surf, score_rect)

        if self.home_button:
            self.home_button.draw(screen)


class VictoryState(State):
    """The game completed victory screen."""

    def __init__(self, game):
        super().__init__(game)
        self.save_score_button: Optional[Button] = None
        self.home_button: Optional[Button] = None

    def enter(self):
        self.game.sound_manager.play_music("victory_music", loop=False)
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        center_x, center_y = w // 2, h // 2

        self.save_score_button = Button(
            center_x - 100, center_y + 40, 200, 45, "Save Score", ui.FONT_BTN
        )
        self.home_button = Button(
            center_x - 100, center_y + 105, 200, 45, "Home Menu", ui.FONT_BTN
        )

    def update(self, input_state, events):
        if self.save_score_button and self.save_score_button.update(input_state):
            self.game.state_manager.change_state(NameInputState(self.game))
        elif self.home_button and self.home_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

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


class NameInputState(State):
    def __init__(self, game):
        super().__init__(game)
        self.player_name = ""
        self.final_score = 0
        self.title_text = "ENTER YOUR NAME"
        self.max_name_length = 10

    def enter(self) -> None:
        self.player_name = ""
        self.final_score = self.game.score_management.get_score()

    def update(self, input_state, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.player_name = self.player_name[:-1]
                elif event.key == pygame.K_RETURN:
                    self.confirm_name()
                elif event.unicode.isalnum():
                    if len(self.player_name) < self.max_name_length:
                        self.player_name += event.unicode

    def confirm_name(self):
        self.game.highscore_manager.add_score(self.player_name, self.final_score)
        self.game.state_manager.change_state(HighScoreState(self.game))

    def draw(self, screen):
        screen.fill(ui.COLOR_BG_PANEL)

        ui.draw_text_centered(
            screen, ui.FONT_TITLE, self.title_text, 120, ui.COLOR_WHITE
        )
        ui.draw_text_centered(
            screen,
            ui.FONT_INPUT,
            f"Final Score: {self.final_score}",
            190,
            ui.COLOR_NEON_YELLOW,
        )

        # Name input
        input_surf = ui.FONT_INPUT.render(
            self.player_name + "_", True, ui.COLOR_NEON_CYAN
        )
        input_rect = input_surf.get_rect(center=(screen.get_width() // 2, 260))
        screen.blit(input_surf, input_rect)

        ui.draw_text_centered(
            screen,
            ui.FONT_INPUT,
            "Press ENTER to save",
            330,
            (128, 128, 128),
        )
