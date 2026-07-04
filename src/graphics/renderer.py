"""Implements the State Pattern for game screens and UI rendering."""

import pygame
from typing import Any, List, Optional
from src.UI.button import Button

TOP_BAR_HEIGHT = 30
CELL_SIZE = 30
PADDING = 20
NORTH = 1 << 0
EAST = 1 << 1
SOUTH = 1 << 2
WEST = 1 << 3


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

    def update(self, input_state: Any, events: List[pygame.event.Event]) -> None:
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
        self.current: Optional[State] = None

    def change_state(self, state: State) -> None:
        """Exit the current state and enter the new one."""
        if self.current:
            self.current.exit()
        self.current = state
        self.current.enter()

    def update(self, input_state: Any, events: List[pygame.event.Event]) -> None:
        """Forward updates to the active state."""
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
        self.play_button = Button(200, 180, 200, 50, "START GAME", self.font_btn)
        self.instructions_button = Button(
            200, 245, 200, 50, "INSTRUCTIONS", self.font_btn
        )
        self.scores_button = Button(200, 310, 200, 50, "HIGHSCORES", self.font_btn)
        self.quit_button = Button(200, 375, 200, 50, "EXIT", self.font_btn)

    def enter(self) -> None:
        """Ensure screen size is set for the main menu."""
        self.game.resize_window(600, 500)

    def update(self, input_state: Any, events: List[pygame.event.Event]) -> None:
        """Check button clicks."""
        if self.play_button.update(input_state):
            # Reset score, lives, and start playing level 0
            self.game.score = 0
            self.game.lives = self.game.config.lives
            self.game.level_manager.current_level_index = 0
            self.game.curr_level = self.game.config.levels[0]

            self.game.curr_level.height = min(self.game.curr_level.height, 32)
            self.game.curr_level.width = min(self.game.curr_level.width, 60)

            self.game.state_manager.change_state(PlayingState(self.game))

        elif self.instructions_button.update(input_state):
            self.game.state_manager.change_state(InstructionsState(self.game))

        elif self.scores_button.update(input_state):
            self.game.state_manager.change_state(HighScoreState(self.game))

        elif self.quit_button.update(input_state):
            self.game.running = False

    def draw(self, screen: pygame.Surface) -> None:
        """Render the main menu elements."""
        screen.fill((5, 5, 10))

        title_surf = self.font_title.render("PAC-MAN", True, (255, 238, 0))
        title_rect = title_surf.get_rect(center=(300, 80))
        screen.blit(title_surf, title_rect)

        subtitle_surf = self.font_btn.render("NEON RETRO EDITION", True, (0, 238, 255))
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

    def update(self, input_state: Any, events: List[pygame.event.Event]) -> None:
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
        self.msg_timer: float = 0.0
        self.msg_text: str = ""

    def enter(self) -> None:
        """Load the level, generate the maze, and set window size."""
        self.game.level_manager.load_level(
            self.game.level_manager.current_level_index,
        )

        width = self.game.level_manager.get_current_level_config().width
        height = self.game.level_manager.get_current_level_config().height
        self.game.recalculate_cell_size(width, height)
        self.game.resize_window(
            width * self.game.cell_size + self.game.padding,
            height * self.game.cell_size + self.game.padding + 60,
        )

        curr_idx = self.game.level_manager.current_level_index
        self.msg_text = f"LEVEL {curr_idx + 1}"
        self.msg_timer = 2.0  # Show message for 2 seconds

    def draw_maze(self, maze, screen: pygame.Surface):

        for row, cells in enumerate(maze):

            for col, cell in enumerate(cells):

                x = PADDING // 2 + col * CELL_SIZE

                y = PADDING // 2 + row * CELL_SIZE + TOP_BAR_HEIGHT

                if cell & NORTH:
                    pygame.draw.line(
                        screen,
                        "blue",
                        (x, y),
                        (x + CELL_SIZE, y),
                        2,
                    )

                if cell & EAST:
                    pygame.draw.line(
                        screen,
                        "blue",
                        (x + CELL_SIZE, y),
                        (x + CELL_SIZE, y + CELL_SIZE),
                        2,
                    )

                if cell & SOUTH:
                    pygame.draw.line(
                        screen,
                        "blue",
                        (x, y + CELL_SIZE),
                        (x + CELL_SIZE, y + CELL_SIZE),
                        2,
                    )

                if cell & WEST:
                    pygame.draw.line(
                        screen,
                        "blue",
                        (x, y),
                        (x, y + CELL_SIZE),
                        2,
                    )

    def draw(self, screen: pygame.Surface) -> None:
        """Render HUD, Maze, and Entities."""
        screen.fill((0, 0, 0))

        # 1. Draw Maze
        self.draw_maze(self.game.level_manager.current_maze.maze, screen)

        # 3. Draw HUD (Top area)
        pygame.draw.rect(screen, (10, 10, 20), (0, 0, screen.get_width(), 40))
        pygame.draw.line(screen, (0, 238, 255), (0, 40), (screen.get_width(), 40), 2)

        score_surf = self.font_hud.render(
            f"SCORE: {self.game.score}", True, (255, 238, 0)
        )
        lvl_num = self.game.level_manager.current_level_index + 1
        level_surf = self.font_hud.render(f"LEVEL: {lvl_num}", True, (255, 255, 255))
        lives_surf = self.font_hud.render(
            f"LIVES: {self.game.lives}", True, (255, 0, 0)
        )

        time_rem = max(0, int(self.game.level_manager.remaining_time))
        time_surf = self.font_hud.render(f"TIME: {time_rem}s", True, (0, 255, 0))

        screen.blit(score_surf, (15, 10))
        screen.blit(level_surf, (screen.get_width() // 3, 10))
        screen.blit(lives_surf, (2 * screen.get_width() // 3, 10))
        screen.blit(time_surf, (screen.get_width() - 110, 10))

        # 5. Draw Alert/Status overlay message
        if self.msg_timer > 0:
            msg_surf = self.font_msg.render(self.msg_text, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(
                center=(screen.get_width() // 2, screen.get_height() // 2)
            )
            bg_rect = msg_rect.inflate(30, 15)
            pygame.draw.rect(screen, (10, 10, 30), bg_rect)
            pygame.draw.rect(screen, (0, 238, 255), bg_rect, 2)
            screen.blit(msg_surf, msg_rect)


class PauseState(State):
    """The paused overlay menu screen state."""

    def __init__(self, game: Any) -> None:
        """Initialize buttons centered relative to window."""
        super().__init__(game)
        self.font_title = pygame.font.Font(None, 48)
        self.font_btn = pygame.font.Font(None, 36)
        self.resume_button: Optional[Button] = None
        self.home_button: Optional[Button] = None

    def enter(self) -> None:
        """Center buttons on the current window size."""
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()
        self.resume_button = Button(
            w // 2 - 100, h // 2 - 40, 200, 45, "RESUME", self.font_btn
        )
        self.home_button = Button(
            w // 2 - 100, h // 2 + 25, 200, 45, "MAIN MENU", self.font_btn
        )

    def update(self, input_state: Any, events: List[pygame.event.Event]) -> None:
        """Check button clicks or escape to resume."""
        if input_state.pause_pressed:
            self.game.state_manager.change_state(PlayingState(self.game))
            return

        if self.resume_button and self.resume_button.update(input_state):
            self.game.state_manager.change_state(PlayingState(self.game))
        elif self.home_button and self.home_button.update(input_state):
            self.game.state_manager.change_state(HomeState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        """Render transparent overlay on top of current gameplay."""
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

    pass


class HighScoreState(State):
    """The top 10 highscores display screen."""

    pass


class VictoryState(State):
    """The game completed victory screen with Name Input."""

    pass
