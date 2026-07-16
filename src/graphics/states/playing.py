import math

import pygame
import pygame.draw as dr
import random
from typing import Any, List

from src.graphics.renderer import State
from src.graphics import ui_helpers as ui
from src.logic.config import (
    CELL_SIZE,
    PADDING,
    TOP_BAR_HEIGHT,
    EAST,
    NORTH,
    SOUTH,
    WEST,
)
from src.logic.helpers import cell_to_screen, pixel_to_screen, expired, after
from src.logic.movement import MovementSystem


class PlayingState(State):
    """The active gameplay state handling movements, collisions, timers."""

    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.player_invincible_until = 0
        self.msg_timer: float = 0.0
        self.msg_text: str = ""

        self.active_cheats: set[str] = set()
        self.player_speed: float = 0.0

    def enter(self) -> None:
        self.game.level_manager.load_level(
            self.game.level_manager.current_level_index,
        )
        self.maze = self.game.level_manager.current_maze.maze
        self.game.entity_manager.load_level_entities(self.maze)

        height = len(self.maze)
        width = len(self.maze[0]) if height else 0
        self.game.recalculate_cell_size(width, height)
        self.game.resize_window(
            width * CELL_SIZE + PADDING,
            height * CELL_SIZE + PADDING + 60,
        )

        curr_idx = self.game.level_manager.current_level_index
        self.movement = MovementSystem(self.maze)
        self.msg_text = f"LEVEL {curr_idx + 1}"
        self.msg_timer = 2.0

        self.player_speed = self.game.entity_manager.player.speed
        self.active_cheats = set()
        print(self.maze)

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
        elif input_state.extra_life:
            self.game.lives = min(
                self.game.config.lives,
                self.game.lives + 1,
            )
        elif input_state.invinciblity:
            self._toggle_cheat("invincible")
        elif input_state.speed_boost:
            self._toggle_cheat("speed boost")
        elif input_state.ghost_freez:
            self._toggle_cheat("ghost freeze")

        if input_state.action_pressed:
            player.use_ability()

    def _toggle_cheat(self, name: str) -> None:
        """Flip a named cheat on/off and apply its side effects."""
        if name in self.active_cheats:
            self.active_cheats.discard(name)
            turning_on = False
        else:
            self.active_cheats.add(name)
            turning_on = True

        if name == "invincible":
            self.player_invincible_until = 999999999999 if turning_on else 0
        elif name == "speed boost":
            player = self.game.entity_manager.player
            player.speed = (
                self.player_speed * 2 if turning_on else self.player_speed
            )

    def _update_entities(self) -> None:
        em = self.game.entity_manager
        self.movement.update_entity(em.player)

        if "ghost freeze" not in self.active_cheats:
            for ghost in em.ghosts:
                if ghost.going_to_prison:
                    if ghost.prison_target is not None:
                        target_y, target_x = ghost.prison_target

                        self.movement.update_ghost_to_target(
                            ghost, target_y, target_x, em.pattern_42_cells
                        )

                        if (
                            ghost.grid_y == target_y
                            and ghost.grid_x == target_x
                        ):
                            ghost.going_to_prison = False
                            ghost.in_prison = True
                            ghost.respawn_timer = after(5000)

                elif ghost.in_prison:
                    if expired(ghost.respawn_timer):
                        ghost.reset()
                        ghost.is_eaten = False
                        ghost.is_edible = False
                        ghost.going_to_prison = False
                        ghost.in_prison = False
                        ghost.prison_target = None
                        ghost.respawn_timer = 0.0

                        continue

                    self.movement.move_inside_prison(
                        ghost, self.game.entity_manager, em.pattern_42_cells
                    )
                elif ghost.is_eaten:
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

                self.game.state_manager.change_state(
                    GameOverState(self.game, self),
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
                from src.graphics.states.vectory import VictoryState

                self.game.state_manager.change_state(VictoryState(self.game))
            else:
                self.game.level_manager.current_level_index = next_lvl
                self.game.state_manager.change_state(PlayingState(self.game))

    def draw(self, screen: pygame.Surface) -> None:
        self._draw_maze_panel(screen)
        self._draw_maze_walls(screen)
        self.game.entity_manager.draw(screen)
        self._draw_hud(screen)
        self._draw_message(screen)
        self._draw_cheat_banner(screen)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, screen: pygame.Surface) -> None:
        w = screen.get_width()
        hud_h = TOP_BAR_HEIGHT + PADDING // 2  # matches the maze's top offset

        # Vertical gradient panel instead of a flat fill.
        top_c = ui.COLOR_HUD_TOP
        bot_c = ui.COLOR_HUD_BOTTOM
        for y in range(hud_h):
            t = y / max(hud_h - 1, 1)
            color = (
                int(top_c[0] + (bot_c[0] - top_c[0]) * t),
                int(top_c[1] + (bot_c[1] - top_c[1]) * t),
                int(top_c[2] + (bot_c[2] - top_c[2]) * t),
            )
            pygame.draw.line(screen, color, (0, y), (w, y))

        pygame.draw.line(
            screen,
            ui.COLOR_DIM_CYAN,
            (0, hud_h + 1),
            (w, hud_h + 1),
            3,
        )
        pygame.draw.line(screen, ui.COLOR_NEON_CYAN, (0, hud_h), (w, hud_h), 2)

        bracket = 12
        for x, direction in ((6, 1), (w - 6, -1)):
            pygame.draw.line(
                screen,
                ui.COLOR_NEON_YELLOW,
                (x, 4),
                (x + direction * bracket, 4),
                2,
            )
            pygame.draw.line(
                screen,
                ui.COLOR_NEON_YELLOW,
                (x, 4),
                (x, 4 + bracket),
                2,
            )

        seg_w = w / 4
        cy = hud_h // 2

        score_text = f"{self.game.score_management.get_score():,}"
        self._draw_hud_stat(
            screen,
            seg_w * 0.5,
            cy,
            "\u2605",
            score_text,
            ui.COLOR_NEON_YELLOW,
            seg_w,
        )

        lvl_num = self.game.level_manager.current_level_index + 1
        self._draw_hud_stat(
            screen, seg_w * 1.5, cy, "LV", str(lvl_num), ui.COLOR_WHITE, seg_w
        )

        self._draw_lives(screen, seg_w * 2.5, cy, seg_w)

        level_cfg = self.game.level_manager.get_current_level_config()
        time_rem = max(0, int(self.game.level_manager.remaining_time))
        frac = time_rem / max(level_cfg.level_max_time, 1)
        if frac < 0.2:
            time_color = ui.COLOR_RED
        elif frac < 0.5:
            time_color = ui.COLOR_NEON_YELLOW
        else:
            time_color = ui.COLOR_GREEN
        self._draw_hud_stat(
            screen,
            seg_w * 3.5,
            cy,
            "\u23f1",
            f"{time_rem}s",
            time_color,
            seg_w,
        )

        # Thin time-remaining drain bar right under the HUD.
        bar_h = 3
        pygame.draw.rect(screen, (30, 30, 40), (0, hud_h - bar_h, w, bar_h))
        pygame.draw.rect(
            screen,
            time_color,
            (
                0,
                hud_h - bar_h,
                int(w * frac),
                bar_h,
            ),
        )

    def _draw_hud_stat(
        self,
        screen: pygame.Surface,
        cx: float,
        cy: float,
        icon: str,
        value: str,
        color,
        seg_w: float,
    ) -> None:
        text = f"{icon} {value}"
        font = ui.get_scaled_font(text, max_width=seg_w - 8, base_size=26)
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(cx, cy))
        screen.blit(surf, rect)

    def _draw_lives(
        self, screen: pygame.Surface, cx: float, cy: float, seg_w: float
    ) -> None:
        lives = self.game.lives
        max_icons = 4
        shown = max(min(lives, max_icons), 0)
        icon_r = 6
        spacing = icon_r * 2 + 5
        total_w = shown * spacing
        start_x = cx - total_w / 2 + icon_r

        for i in range(shown):
            x = start_x + i * spacing
            self._draw_pacman_icon(
                screen,
                (x, cy),
                icon_r,
                ui.COLOR_NEON_YELLOW,
            )

        if lives > max_icons:
            extra = f"+{lives - max_icons}"
            font = ui.get_scaled_font(
                extra,
                max_width=seg_w * 0.3,
                base_size=20,
            )
            surf = font.render(extra, True, ui.COLOR_WHITE)
            screen.blit(
                surf,
                (start_x + total_w + 6, cy - surf.get_height() // 2),
            )
        elif lives <= 0:
            surf = ui.get_scaled_font("0", seg_w - 8, base_size=22).render(
                "0", True, ui.COLOR_RED
            )
            screen.blit(surf, surf.get_rect(center=(cx, cy)))

    @staticmethod
    def _draw_pacman_icon(
        screen: pygame.Surface,
        center,
        radius: int,
        color,
    ) -> None:
        x, y = center
        mouth = 40
        start = math.radians(mouth / 2)
        end = math.radians(360 - mouth / 2)
        steps = 10
        points = [center]
        for i in range(steps + 1):
            a = start + (end - start) * i / steps
            points.append((x + radius * math.cos(a), y - radius * math.sin(a)))
        pygame.draw.polygon(screen, color, points)

    # ------------------------------------------------------------------
    # Maze
    # ------------------------------------------------------------------

    def _draw_maze_panel(self, screen: pygame.Surface) -> None:
        """Dark rounded backdrop with a glowing frame behind the maze."""
        rows = len(self.maze)
        cols = len(self.maze[0]) if rows else 0
        x0, y0 = cell_to_screen(0, 0)
        panel_rect = pygame.Rect(
            x0 - 6, y0 - 6, cols * CELL_SIZE + 12, rows * CELL_SIZE + 12
        )

        pygame.draw.rect(
            screen,
            ui.COLOR_BG_DARK,
            panel_rect,
            border_radius=10,
        )
        pygame.draw.rect(
            screen, ui.COLOR_DIM_CYAN, panel_rect, width=3, border_radius=10
        )
        pygame.draw.rect(
            screen, ui.COLOR_NEON_CYAN, panel_rect, width=1, border_radius=10
        )

    def _draw_maze_walls(self, screen: pygame.Surface) -> None:
        c = CELL_SIZE

        for row, cells in enumerate(self.maze):
            for col, cell in enumerate(cells):
                x, y = cell_to_screen(row, col)

                if cell & NORTH:
                    self._draw_wall_segment(screen, (x, y), (x + c, y))
                if cell & EAST:
                    self._draw_wall_segment(screen, (x + c, y), (x + c, y + c))
                if cell & SOUTH:
                    self._draw_wall_segment(screen, (x, y + c), (x + c, y + c))
                if cell & WEST:
                    self._draw_wall_segment(screen, (x, y), (x, y + c))

    @staticmethod
    def _draw_wall_segment(screen: pygame.Surface, p1, p2) -> None:
        """Two-tone glowing wall: a soft dim outer stroke plus a bright core,
        with rounded joints so segments read as continuous walls."""
        glow_w, core_w = 6, 2
        dr.line(screen, ui.COLOR_DIM_CYAN, p1, p2, glow_w)
        dr.circle(screen, ui.COLOR_DIM_CYAN, p1, glow_w // 2)
        dr.circle(screen, ui.COLOR_DIM_CYAN, p2, glow_w // 2)

        dr.line(screen, ui.COLOR_NEON_CYAN, p1, p2, core_w)
        dr.circle(screen, ui.COLOR_NEON_CYAN, p1, core_w // 2 + 1)
        dr.circle(screen, ui.COLOR_NEON_CYAN, p2, core_w // 2 + 1)

    # ------------------------------------------------------------------
    # Floating message
    # ------------------------------------------------------------------

    def _draw_message(self, screen: pygame.Surface) -> None:
        if self.msg_timer > 0:
            self.msg_timer -= 1 / 60
        else:
            self.msg_text = ""

        if self.msg_timer <= 0:
            return

        player = self.game.entity_manager.player
        px, py = pixel_to_screen(player.x, player.y)
        text_surface = ui.FONT_HUD.render(self.msg_text, True, ui.COLOR_WHITE)
        text_rect = text_surface.get_rect(center=(px, py - 40))

        bubble_rect = text_rect.inflate(16, 10)
        bubble = pygame.Surface(bubble_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            bubble,
            (0, 0, 0, 160),
            bubble.get_rect(),
            border_radius=8,
        )
        pygame.draw.rect(
            bubble,
            (*ui.COLOR_NEON_CYAN, 200),
            bubble.get_rect(),
            width=1,
            border_radius=8,
        )
        screen.blit(bubble, bubble_rect.topleft)
        screen.blit(text_surface, text_rect)

    # ------------------------------------------------------------------
    # Cheat banner
    # ------------------------------------------------------------------

    def _draw_cheat_banner(self, screen: pygame.Surface) -> None:
        if not self.active_cheats:
            return

        label = " + ".join(sorted(self.active_cheats)).upper()
        text = f"CHEATS ACTIVE: {label}"

        font = ui.get_scaled_font(
            text,
            max_width=screen.get_width() - 24,
            base_size=18,
        )
        surf = font.render(text, True, ui.COLOR_NEON_YELLOW)
        rect = surf.get_rect(
            midbottom=(screen.get_width() // 2, screen.get_height() - 8)
        )

        bubble_rect = rect.inflate(20, 10)
        bubble = pygame.Surface(bubble_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(
            bubble,
            (0, 0, 0, 170),
            bubble.get_rect(),
            border_radius=8,
        )
        pygame.draw.rect(
            bubble,
            (*ui.COLOR_NEON_YELLOW, 200),
            bubble.get_rect(),
            width=1,
            border_radius=8,
        )
        screen.blit(bubble, bubble_rect.topleft)
        screen.blit(surf, rect)

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

                    ghost.going_to_prison = True
                    ghost.in_prison = False

                    if self.game.entity_manager.pattern_42_cells:
                        ghost.prison_target = random.choice(
                            self.game.entity_manager.pattern_42_cells
                        )
                    else:
                        ghost.prison_target = (ghost.spawn_y, ghost.spawn_x)

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
