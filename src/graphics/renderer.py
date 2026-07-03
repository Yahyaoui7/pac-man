"""Implements the State Pattern for game screens and UI rendering."""

import pygame
from typing import Any, List, Optional
from src.UI.button import Button


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
            self.game.state_manager.change_state(PlayingState(self.game))

        elif self.instructions_button.update(input_state):
            self.game.state_manager.change_state(InstructionsState(self.game))

        elif self.scores_button.update(input_state):
            self.game.state_manager.change_state(HighScoreState(self.game))

        elif self.quit_button.update(input_state):
            self.game.running = False

    def draw(self, screen: pygame.Surface) -> None:
        """Render the main menu elements."""
        screen.fill((5, 5, 10))  # Slate dark

        # Draw glowing retro title
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


class HighScoreState(State):
    """The top 10 highscores display screen."""

    pass


class PlayingState(State):
    """The active gameplay state handling movements, collisions, timers."""

    def __init__(self, game: Any) -> None:
        """Initialize playing parameters."""
        super().__init__(game)
        self.font_hud = pygame.font.Font(None, 28)
        self.font_msg = pygame.font.Font(None, 48)
        self.msg_timer: float = 0.0
        self.msg_text: str = ""


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


class VictoryState(State):
    """The game completed victory screen with Name Input."""

    pass
