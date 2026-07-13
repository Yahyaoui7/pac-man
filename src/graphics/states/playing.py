import math

import pygame
import pygame.draw as dr
from typing import Any, List

from src.graphics.renderer import State
from src.graphics import ui_helpers as ui
from src.logic.config import CELL_SIZE, PADDING, EAST, NORTH, SOUTH, WEST
from src.logic.helpers import cell_to_screen, pixel_to_screen, expired, after
from src.logic.movement import MovementSystem


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
            from src.graphics.states.game_over import GameOverState

            self.game.state_manager.change_state(
                GameOverState(self.game, self),
            )
            return
        if input_state.pause_pressed:
            from src.graphics.states.pause import PauseState

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
                from src.graphics.states.game_over import GameOverState

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
                from src.graphics.states.vectory import VictoryState

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
                    ghost.is_edible = False
                    ghost.frightened_timer = 0.0
                    ghost.respawn_timer = -1.0

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
