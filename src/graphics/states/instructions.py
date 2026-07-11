import pygame
from typing import Any, List

from src.UI.button import Button
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui


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
            from src.graphics.states.home import HomeState
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
