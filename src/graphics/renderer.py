"""Implements the State Pattern for game screens and UI rendering."""

import math
from src.logic.utils import expired, after

import pygame
import pygame.draw as dr
from typing import Any, List, Optional
from src.UI.button import Button


from src.logic.config import CELL_SIZE, PADDING, TOP_BAR_HEIGHT
from src.logic.config import EAST, NORTH, SOUTH, WEST
from src.logic.movement import MovementSystem


class State:
    """Base class for all screen states."""

    def __init__(self, game: Any) -> None:
        """Initialize with a reference to the main game object."""
        self.game = game

    def enter(self) -> None:
        """Called when this state becomes active."""
        pass

    def exit(self) -> None:
        """Called when leaving this state."""
        pass

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        """Process logic and events."""
        pass

    def draw(self, screen: pygame.Surface) -> None:
        """Render to the screen."""
        pass


class StateManager:
    """Manages switching and updating the active screen state."""

    def __init__(self, game: Any) -> None:
        """Initialize the state manager."""
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
        """Forward draw calls to the active state."""
        if self.current:
            self.current.draw(screen)


class HomeState(State):
    """The Main Menu screen state."""

    def __init__(self, game: Any) -> None:
        """Initialize buttons for the main menu."""
        super().__init__(game)
        self.font_title = pygame.font.Font(None, 64)
        self.font_btn = pygame.font.Font(None, 36)

        # Center buttons on a 600x500 screen
        self.play_button = Button(
            200,
            180,
            200,
            50,
            "START GAME",
            self.font_btn,
        )
        self.instructions_button = Button(
            200, 245, 200, 50, "INSTRUCTIONS", self.font_btn
        )
        self.scores_button = Button(
            200,
            310,
            200,
            50,
            "HIGHSCORES",
            self.font_btn,
        )
        self.quit_button = Button(200, 375, 200, 50, "EXIT", self.font_btn)

    def enter(self) -> None:
        """Ensure screen size is set for the main menu."""
        self.game.resize_window(600, 500)

        self.game.sound_manager.play_music("menu_intro", False)

        self.game.lives = self.game.config.lives
        # self.game.sound_manager.play_music("menu")

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        """Check button clicks."""
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
        """Render the main menu elements."""
        screen.fill((5, 5, 10))

        title_surf = self.font_title.render("PAC-MAN", True, (255, 238, 0))
        title_rect = title_surf.get_rect(center=(300, 80))
        screen.blit(title_surf, title_rect)

        subtitle_surf = self.font_btn.render(
            "NEON RETRO EDITION",
            True,
            (0, 238, 255),
        )
        subtitle_rect = subtitle_surf.get_rect(center=(300, 125))
        screen.blit(subtitle_surf, subtitle_rect)

        # Draw buttons
        self.play_button.draw(screen)
        self.instructions_button.draw(screen)
        self.scores_button.draw(screen)
        self.quit_button.draw(screen)


class InstructionsState(State):
    """The game rules and controls instructions screen."""

    def __init__(self, game: Any) -> None:
        """Initialize buttons and text fonts."""
        super().__init__(game)
        self.font_title = pygame.font.Font(None, 48)
        self.font_text = pygame.font.Font(None, 24)
        self.font_btn = pygame.font.Font(None, 36)
        self.back_button = Button(200, 420, 200, 45, "BACK", self.font_btn)

    def update(
        self,
        input_state: Any,
        events: List[pygame.event.Event],
    ) -> None:
        """Check back button click."""
        if self.back_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        """Render instructions text."""
        screen.fill((5, 5, 10))

        title_surf = self.font_title.render("HOW TO PLAY", True, (0, 238, 255))
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
            if "CHEAT" in line or line.startswith("- Press"):
                color = (255, 238, 0)
            else:
                color = (255, 255, 255)
            line_surf = self.font_text.render(line, True, color)
            screen.blit(line_surf, (50, y_offset))
            y_offset += 24

        self.back_button.draw(screen)


class PlayingState(State):
    """The active gameplay state handling movements, collisions, timers."""

    def __init__(self, game: Any) -> None:
        """Initialize playing parameters."""
        super().__init__(game)

        self.font_hud = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 48)
        self.player_invincible_until = 0
        self.msg_timer: float = 0.0
        self.msg_text: str = ""

    def enter(self) -> None:

        # self.game.sound_manager.play_music("game_intro", False)
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

        if input_state.move_left:
            self.game.entity_manager.player.next_direction = "LEFT"
        elif input_state.move_right:
            self.game.entity_manager.player.next_direction = "RIGHT"
        elif input_state.move_up:
            self.game.entity_manager.player.next_direction = "UP"
        elif input_state.move_down:
            self.game.entity_manager.player.next_direction = "DOWN"

        if input_state.action_pressed:
            self.game.entity_manager.player.use_ability()

        self.movement.update_entity(self.game.entity_manager.player)

        for ghost in self.game.entity_manager.ghosts:

            if ghost.is_eaten:

                self.movement.update_ghost_to_target(
                    ghost,
                    ghost.spawn_y,
                    ghost.spawn_x,
                )

            elif ghost.is_edible:
                self.movement.update_runaway_ghost(
                    ghost,
                    self.game.entity_manager.player,
                )
            else:
                self.movement.update_bfs_ghost(
                    ghost, self.game.entity_manager.player
                )
        self.game.entity_manager.update(
            self.game.level_manager.current_maze.maze, 1 / 60.0
        )
        self.check_collision(
            self.game.entity_manager.player,
            self.game.entity_manager.ghosts,
        )
        self.game.level_manager.update_time(1 / 60.0)
        if self.game.level_manager.is_time_out():

            if self.game.lives <= 0:
                self.game.state_manager.change_state(
                    GameOverState(self.game, self)
                )
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

        pygame.draw.rect(screen, (10, 10, 20), (0, 0, screen.get_width(), 40))
        pygame.draw.line(
            screen, (0, 238, 255), (0, 40), (screen.get_width(), 40), 2
        )

        score_surf = self.font_hud.render(
            f"SCORE: {self.game.score_management.get_score()}",
            True,
            (255, 238, 0),
        )
        lvl_num = self.game.level_manager.current_level_index + 1
        level_surf = self.font_hud.render(
            f"LEVEL: {lvl_num}", True, (255, 255, 255)
        )
        lives_surf = self.font_hud.render(
            f"LIVES: {self.game.lives}", True, (255, 0, 0)
        )

        time_rem = max(0, int(self.game.level_manager.remaining_time))
        time_surf = self.font_hud.render(
            f"TIME: {time_rem}s", True, (0, 255, 0)
        )

        screen.blit(score_surf, (15, 10))
        screen.blit(level_surf, (screen.get_width() // 3, 10))
        screen.blit(lives_surf, (2 * screen.get_width() // 3, 10))
        screen.blit(time_surf, (screen.get_width() - 110, 10))
        if self.msg_timer > 0:
            self.msg_timer -= 1 / 60
        else:
            self.msg_text = ""

        player = self.game.entity_manager.player
        if self.msg_timer > 0:
            text_surface = self.font_hud.render(self.msg_text, True, "white")
            screen.blit(
                text_surface,
                (
                    PADDING // 2 + player.x - text_surface.get_width() // 2,
                    TOP_BAR_HEIGHT + PADDING // 2 + player.y - 40,
                ),
            )

        c = CELL_SIZE

        for row, cells in enumerate(self.maze):

            for col, cell in enumerate(cells):
                x = PADDING // 2 + col * CELL_SIZE
                y = PADDING // 2 + row * CELL_SIZE + TOP_BAR_HEIGHT

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

        self.font_hud.render(self.msg_text, True, "white")


class PauseState(State):
    """The paused overlay menu screen state."""

    def __init__(self, game: Any, previous_state: PlayingState) -> None:
        super().__init__(game)
        self.previous_state = previous_state
        self.font_title = pygame.font.Font(None, 48)
        self.font_btn = pygame.font.Font(None, 36)
        self.resume_button: Optional[Button] = None
        self.home_button: Optional[Button] = None

    def enter(self) -> None:
        self.game.sound_manager.play_sound("pause")

        w = self.game.screen.get_width()
        h = self.game.screen.get_height()

        self.resume_button = Button(
            w // 2 - 100, h // 2 - 40, 200, 45, "RESUME", self.font_btn
        )
        self.home_button = Button(
            w // 2 - 100, h // 2 + 25, 200, 45, "Home MENU", self.font_btn
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

        overlay = pygame.Surface(
            (screen.get_width(), screen.get_height()), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        title_surf = self.font_title.render("GAME PAUSED", True, (0, 238, 255))
        title_rect = title_surf.get_rect(
            center=(screen.get_width() // 2, screen.get_height() // 2 - 100)
        )
        screen.blit(title_surf, title_rect)

        if self.resume_button and self.home_button:
            self.resume_button.draw(screen)
            self.home_button.draw(screen)


class GameOverState(State):
    """The Game Over display screen and Name Input handler."""

    def __init__(self, game, previous_state: PlayingState):
        super().__init__(game)
        self.previous_state = previous_state
        self.font_title = pygame.font.Font(None, 48)
        self.losing_cause = pygame.font.Font(None, 42)

        self.font_btn = pygame.font.Font(None, 36)

        self.name_button: Optional[Button] = None
        self.home_button: Optional[Button] = None
        self.hi_score_button: Optional[Button] = None

    def enter(self):
        self.game.sound_manager.play_music("game_over_music", loop=False)
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()

        center_x = w // 2
        center_y = h // 2
        self.name_button = Button(
            center_x - 100,
            center_y + 20,
            200,
            45,
            "Enter Your Name",
            self.font_btn,
        )

        self.hi_score_button = Button(
            center_x - 100,
            center_y + 85,
            200,
            45,
            "Highest Score",
            self.font_btn,
        )

        self.home_button = Button(
            center_x - 100,
            center_y + 150,
            200,
            45,
            "Home Menu",
            self.font_btn,
        )

    def update(self, input_state, events):
        if self.name_button and self.name_button.update(input_state):
            self.game.state_manager.change_state(NameInputState(self.game))

        elif self.hi_score_button and self.hi_score_button.update(input_state):
            self.game.state_manager.change_state(
                HighScoreState(self.game, self)
            )
        elif self.home_button and self.home_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        lives = self.game.lives

        self.previous_state.draw(screen)

        # Overlay
        overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))

        center_x = screen.get_width() // 2
        center_y = screen.get_height() // 2

        # ---------- Title ----------
        title_surf = self.font_title.render(
            "GAME OVER",
            True,
            (0, 238, 255),
        )
        title_rect = title_surf.get_rect(center=(center_x, center_y - 170))
        screen.blit(title_surf, title_rect)

        # ---------- Losing Cause ----------
        cause = "You were caught by a ghost!" if lives == 0 else "Time's Up!"

        losing_surf = self.losing_cause.render(
            cause,
            True,
            (255, 80, 80),
        )
        losing_rect = losing_surf.get_rect(center=(center_x, center_y - 125))
        screen.blit(losing_surf, losing_rect)

        # ---------- Scores ----------
        score = self.game.score_management.get_score()
        top_scores = self.game.highscore_manager.get_top_scores()

        if top_scores:
            high_score = top_scores[0]["score"]
        else:
            high_score = 0

        score_label = self.font_btn.render(
            "Your Score:",
            True,
            (220, 220, 220),
        )
        score_value = self.font_btn.render(
            str(score),
            True,
            (255, 238, 0),
        )

        score_label_rect = score_label.get_rect(
            center=(center_x - 40, center_y - 65)
        )
        score_value_rect = score_value.get_rect(
            midleft=(score_label_rect.right + 10, score_label_rect.centery)
        )

        screen.blit(score_label, score_label_rect)
        screen.blit(score_value, score_value_rect)

        high_label = self.font_btn.render(
            "Highest Score:",
            True,
            (220, 220, 220),
        )
        high_value = self.font_btn.render(
            str(high_score),
            True,
            (255, 238, 0),
        )

        high_label_rect = high_label.get_rect(
            center=(center_x - 40, center_y - 25)
        )
        high_value_rect = high_value.get_rect(
            midleft=(high_label_rect.right + 10, high_label_rect.centery)
        )

        screen.blit(high_label, high_label_rect)
        screen.blit(high_value, high_value_rect)

        # ---------- Buttons ----------
        if self.hi_score_button and self.home_button and self.name_button:
            self.name_button.draw(screen)
            self.hi_score_button.draw(screen)
            self.home_button.draw(screen)


class HighScoreState(State):
    """The top 10 highscores display screen."""

    def __init__(self, game: Any, previous_state=None) -> None:
        super().__init__(game)

        self.previous_state = previous_state

        self.font_title = pygame.font.Font(None, 48)
        self.font_score = pygame.font.Font(None, 32)
        self.font_btn = pygame.font.Font(None, 36)

        self.home_button: Optional[Button] = None

    def enter(self) -> None:
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()

        self.home_button = Button(
            w // 2 - 100,
            h - 90,
            200,
            45,
            "Home MENU",
            self.font_btn,
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
            screen.fill((10, 10, 20))

        overlay = pygame.Surface(
            (screen.get_width(), screen.get_height()),
            pygame.SRCALPHA,
        )
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        title_surf = self.font_title.render(
            "HIGH SCORES",
            True,
            (0, 238, 255),
        )
        title_rect = title_surf.get_rect(center=(screen.get_width() // 2, 90))
        screen.blit(title_surf, title_rect)

        highscores = self.game.highscore_manager.get_top_scores()

        if not highscores:
            empty_surf = self.font_score.render(
                "No scores yet",
                True,
                "white",
            )
            empty_rect = empty_surf.get_rect(
                center=(screen.get_width() // 2, 180)
            )
            screen.blit(empty_surf, empty_rect)
        else:
            start_y = 150

            for index, item in enumerate(highscores):
                name = item["name"]
                score = item["score"]

                score_text = f"{index + 1}. {name}  -  {score}"

                score_surf = self.font_score.render(
                    score_text,
                    True,
                    "white",
                )

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

        self.font_title = pygame.font.Font(None, 64)
        self.font_score = pygame.font.Font(None, 38)
        self.font_btn = pygame.font.Font(None, 36)

        self.save_score_button: Optional[Button] = None
        self.home_button: Optional[Button] = None

    def enter(self):
        # If you do not have this sound, remove this line.
        self.game.sound_manager.play_music("victory_music", loop=False)

        w = self.game.screen.get_width()
        h = self.game.screen.get_height()

        center_x = w // 2
        center_y = h // 2

        self.save_score_button = Button(
            center_x - 100,
            center_y + 40,
            200,
            45,
            "Save Score",
            self.font_btn,
        )

        self.home_button = Button(
            center_x - 100,
            center_y + 105,
            200,
            45,
            "Home Menu",
            self.font_btn,
        )

    def update(self, input_state, events):
        if self.save_score_button and self.save_score_button.update(
            input_state
        ):
            self.game.state_manager.change_state(NameInputState(self.game))

        elif self.home_button and self.home_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen):
        screen.fill((10, 10, 20))

        title_surface = self.font_title.render(
            "YOU WIN!",
            True,
            "yellow",
        )

        score_surface = self.font_score.render(
            f"Final Score: {self.game.score_management.get_score()}",
            True,
            "white",
        )

        screen.blit(
            title_surface,
            (
                screen.get_width() // 2 - title_surface.get_width() // 2,
                130,
            ),
        )

        screen.blit(
            score_surface,
            (
                screen.get_width() // 2 - score_surface.get_width() // 2,
                210,
            ),
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
        self.font_title = pygame.font.Font(None, 48)
        self.font_input = pygame.font.Font(None, 36)

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
        self.game.highscore_manager.add_score(
            self.player_name, self.final_score
        )
        self.game.state_manager.change_state(HighScoreState(self.game))

    def draw(self, screen):
        screen.fill((10, 10, 20))

        title_surface = self.font_title.render(
            self.title_text,
            True,
            "white",
        )

        score_surface = self.font_input.render(
            f"Final Score: {self.final_score}",
            True,
            "yellow",
        )

        input_surface = self.font_input.render(
            self.player_name + "_",
            True,
            "cyan",
        )

        help_surface = self.font_input.render(
            "Press ENTER to save",
            True,
            "gray",
        )

        screen.blit(
            title_surface,
            (
                screen.get_width() // 2 - title_surface.get_width() // 2,
                120,
            ),
        )

        screen.blit(
            score_surface,
            (
                screen.get_width() // 2 - score_surface.get_width() // 2,
                190,
            ),
        )

        screen.blit(
            input_surface,
            (
                screen.get_width() // 2 - input_surface.get_width() // 2,
                260,
            ),
        )

        screen.blit(
            help_surface,
            (
                screen.get_width() // 2 - help_surface.get_width() // 2,
                330,
            ),
        )
