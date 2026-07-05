"""Implements the State Pattern for game screens and UI rendering."""

import pygame
from typing import Any, List, Optional
from src.UI.button import Button


from mazegenerator import MazeGenerator

from src.logic.movement import MovementSystem
from src.logic.entities import Player, Ghost

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

    def update(
        self, input_state: Any, events: List[pygame.event.Event]
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
        self.current: Optional[State] = None

    def change_state(self, state: State) -> None:
        """Exit the current state and enter the new one."""
        if self.current:
            self.current.exit()
            self.current.game.sound_manager.stop_music()
        self.current = state
        self.current.enter()

    def update(
        self, input_state: Any, events: List[pygame.event.Event]
    ) -> None:
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
        self.play_button = Button(
            200, 180, 200, 50, "START GAME", self.font_btn
        )
        self.instructions_button = Button(
            200, 245, 200, 50, "INSTRUCTIONS", self.font_btn
        )
        self.scores_button = Button(
            200, 310, 200, 50, "HIGHSCORES", self.font_btn
        )
        self.quit_button = Button(200, 375, 200, 50, "EXIT", self.font_btn)

    def enter(self) -> None:
        """Ensure screen size is set for the main menu."""
        self.game.resize_window(600, 500)
        self.game.sound_manager.play_music("menu")

    def update(
        self, input_state: Any, events: List[pygame.event.Event]
    ) -> None:
        """Check button clicks."""
        if self.play_button.update(input_state):
            # Reset score, lives, and start playing level 0
            self.game.score = 0
            self.game.lives = self.game.config.lives
            self.game.level_manager.current_level_index = 0
            self.game.curr_level = self.game.config.levels[1]

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

        subtitle_surf = self.font_btn.render(
            "NEON RETRO EDITION", True, (0, 238, 255)
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
        self, input_state: Any, events: List[pygame.event.Event]
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
        """Load the level and initialize gameplay."""
        # self.game.sound_manager.play_music("game")
        self.game.level_manager.load_level(
            self.game.level_manager.current_level_index,
        )

        level = self.game.level_manager.get_current_level_config()

        self.game.recalculate_cell_size(level.width, level.height)

        self.game.resize_window(
            level.width * self.game.cell_size + self.game.padding,
            level.height * self.game.cell_size + self.game.padding + 60,
        )

        self.maze = self.game.level_manager.current_maze.maze
        self.movement = MovementSystem(self.maze)

        player_row, player_col = self.find_player_spawn()

        self.game.player = Player(
            player_row,
            player_col,
            self.game.cell_size,
        )

        self.game.ghosts = [
            Ghost(0, 0, self.game.cell_size, "Blinky"),
            Ghost(0, level.width - 1, self.game.cell_size, "Pinky"),
            Ghost(level.height - 1, 0, self.game.cell_size, "Inky"),
            Ghost(
                level.height - 1,
                level.width - 1,
                self.game.cell_size,
                "Clyde",
            ),
        ]

        self.msg_text = (
            f"LEVEL {self.game.level_manager.current_level_index + 1}"
        )
        self.msg_timer = 2.0

    def update(
        self, input_state: Any, events: List[pygame.event.Event]
    ) -> None:
        if input_state.pause_pressed:
            self.game.state_manager.change_state(PauseState(self.game, self))
            return

        if input_state.move_left:
            self.game.player.next_direction = "LEFT"
        elif input_state.move_right:
            self.game.player.next_direction = "RIGHT"
        elif input_state.move_up:
            self.game.player.next_direction = "UP"
        elif input_state.move_down:
            self.game.player.next_direction = "DOWN"

        self.movement.update_entity(self.game.player)

        for ghost in self.game.ghosts:
            if ghost.is_edible:
                self.movement.update_runaway_ghost(ghost, self.game.player)
            else:
                self.movement.update_bfs_ghost(ghost, self.game.player)

    def draw(self, screen: pygame.Surface) -> None:
        self.draw_hud(screen)
        self.draw_maze(screen)
        self.draw_player(screen)
        self.draw_ghosts(screen)

    def draw_maze(self, screen: pygame.Surface) -> None:
        for row, cells in enumerate(self.maze):

            for col, cell in enumerate(cells):
                x = PADDING // 2 + col * CELL_SIZE
                y = PADDING // 2 + row * CELL_SIZE + TOP_BAR_HEIGHT

                if cell & NORTH:
                    pygame.draw.line(
                        screen, "blue", (x, y), (x + CELL_SIZE, y), 2
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
                        screen, "blue", (x, y), (x, y + CELL_SIZE), 2
                    )

    def draw_player(self, screen: pygame.Surface) -> None:
        player = self.game.player
        if player is None:
            return

        x = PADDING // 2 + player.x
        y = TOP_BAR_HEIGHT + PADDING // 2 + player.y

        pygame.draw.circle(
            screen,
            "yellow",
            (int(x), int(y)),
            CELL_SIZE // 3,
        )

    def draw_ghosts(self, screen: pygame.Surface) -> None:
        colors = {
            "Blinky": "red",
            "Pinky": "pink",
            "Inky": "cyan",
            "Clyde": "orange",
        }

        for ghost in self.game.ghosts:
            x = PADDING // 2 + ghost.x
            y = TOP_BAR_HEIGHT + PADDING // 2 + ghost.y

            if ghost.is_edible:
                color = "blue"
            else:
                color = colors.get(ghost.name, "white")
            pygame.draw.circle(
                screen,
                color,
                (int(x), int(y)),
                CELL_SIZE // 3,
            )

    def draw_hud(self, screen: pygame.Surface) -> None:
        level_number = self.game.level_manager.current_level_index + 1

        hud_text = (
            f"Score: {self.game.score}   "
            f"Lives: {self.game.lives}   "
            f"Level: {level_number}"
        )

        hud_surface = self.font_hud.render(hud_text, True, "white")
        screen.blit(hud_surface, (10, 5))

    def is_valid_spawn(self, row: int, col: int) -> bool:
        cell = self.maze[row][col]
        return cell != (NORTH | EAST | SOUTH | WEST)

    def find_player_spawn(self) -> tuple[int, int]:
        middle_row = self.game.curr_level.height // 2
        middle_col = self.game.curr_level.width // 2

        for radius in range(
            max(self.game.curr_level.width, self.game.curr_level.height)
        ):
            for row in range(middle_row - radius, middle_row + radius + 1):
                for col in range(middle_col - radius, middle_col + radius + 1):
                    if (
                        0 <= row < self.game.curr_level.height
                        and 0 <= col < self.game.curr_level.width
                        and self.is_valid_spawn(row, col)
                    ):
                        return row, col

        return 0, 0


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
        w = self.game.screen.get_width()
        h = self.game.screen.get_height()

        self.resume_button = Button(
            w // 2 - 100, h // 2 - 40, 200, 45, "RESUME", self.font_btn
        )
        self.home_button = Button(
            w // 2 - 100, h // 2 + 25, 200, 45, "MAIN MENU", self.font_btn
        )

    def update(
        self, input_state: Any, events: List[pygame.event.Event]
    ) -> None:
        if input_state.pause_pressed:
            self.game.state_manager.change_state(self.previous_state)
            return

        if self.resume_button and self.resume_button.update(input_state):
            self.game.state_manager.change_state(self.previous_state)

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

    pass


class HighScoreState(State):
    """The top 10 highscores display screen."""

    pass


class VictoryState(State):
    """The game completed victory screen with Name Input."""

    pass
