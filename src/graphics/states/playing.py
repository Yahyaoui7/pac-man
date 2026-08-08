import math
import json

import pygame
import pygame.draw as dr
from typing import Any, List

from AI_arena.ghosts.ghost_controller import CNNGhostController
from AI_arena.player.player_controller import CNNPlayerController
from src.graphics.renderer import State
from src.graphics import ui_helpers as ui
from src.logic.movement import MovementSystem
from src.logic.helpers import cell_to_screen, pixel_to_screen, expired, after

from src.logic.config import (
    CELL_SIZE,
    PADDING,
    TOP_BAR_HEIGHT,
    EAST,
    NORTH,
    SOUTH,
    WEST,
)


class PlayingState(State):
    """The active gameplay state handling movements, collisions, timers."""

    def __init__(self, game: Any) -> None:
        super().__init__(game)
        self.player_invincible_until = 0
        self.msg_timer: float = 0.0
        self.msg_text: str = ""

        self.active_cheats: set[str] = set()
        self.player_speed: float = 0.0
        self.ghost_controller: CNNGhostController | None = None
        self.ghost_decision_sources: dict[str, str] = {}
        self.ghost_predictions: dict[str, str | None] = {}
        self.ghost_decision_cells: dict[str, tuple[int, int]] = {}

        self.player_controller: CNNPlayerController | None = None
        self.use_ai_player: bool = getattr(game, "use_ai_player", False)
        self.ai_player_decision: str | None = None
        self.ai_frame_counter: int = 0
        # Last cell (grid_x, grid_y) at which the AI made a decision.
        # Model is re-queried only when the player reaches a *new* cell center.
        self.ai_last_decision_cell: tuple[int, int] | None = None

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
        self.use_cnn_ghosts = False
        if self.use_cnn_ghosts:
            try:
                self.ghost_controller = CNNGhostController()
                if hasattr(self.ghost_controller, "init_observation"):
                    self.ghost_controller.init_observation(self.maze)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                print(f"Ghost CNN unavailable; using scripted movement: {exc}")
                self.ghost_controller = None
        else:
            self.ghost_controller = None

        try:
            self.player_controller = CNNPlayerController()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"Player RL model unavailable: {exc}")
            self.player_controller = None
        self.ai_last_decision_cell = None
        self.ai_frame_counter = 0

        self.msg_text = f"LEVEL {curr_idx + 1}"
        self.msg_timer = 2.0

        self.player_speed = self.game.entity_manager.player.speed
        self.active_cheats = set()

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

        for event in events:
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_p, pygame.K_a):
                self.use_ai_player = not self.use_ai_player

        self._handle_input(input_state)
        self._update_entities()
        self._check_level_end()

    def _handle_input(self, input_state: Any) -> None:
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
            player.speed = self.player_speed * 2 if turning_on else self.player_speed

    def _update_entities(self) -> None:
        em = self.game.entity_manager

        if self.use_ai_player and self.player_controller is not None:
            self.ai_frame_counter += 1
            self.movement.update_cell_position(em.player)
            current_cell = (em.player.grid_x, em.player.grid_y)

            # Only ask the model when the player has arrived at the center
            # of a cell it hasn't been decided on yet.  This mirrors the
            # training environment (one decision per cell crossing) and
            # prevents the LEFT/RIGHT flicker seen when querying every frame.
            if (
                self.movement.is_centered(em.player)
                and current_cell != self.ai_last_decision_cell
            ):
                self.ai_last_decision_cell = current_cell
                action = self.player_controller.get_action(
                    self.maze,
                    em.pellets,
                    em.player,
                    em.ghosts,
                    self.movement,
                    sample=True,
                )
                if action:
                    em.player.next_direction = action
                    self.ai_player_decision = action
                    diag = self.player_controller.last_diagnostics
                    probs_str = " | ".join(
                        f"{d}:{p*100:.0f}%"
                        for d, p in diag.get("probabilities", {}).items()
                    )
                    print(
                        f"🤖 [PLAYER AI] Frame {self.ai_frame_counter:04d} "
                        f"Node ({em.player.grid_x:02d},{em.player.grid_y:02d}) "
                        f"-> Choice: {action:<5s} | Probs: [{probs_str}]"
                    )

        self.movement.update_entity(em.player)

        if "ghost freeze" not in self.active_cheats:
            controller = self.ghost_controller
            decision_names: set[str] = set()
            if controller is not None:
                for ghost in em.ghosts:
                    if (
                        self.movement.is_centered(ghost)
                        and not ghost.going_to_prison
                        and not ghost.in_prison
                        and not ghost.is_eaten
                    ):
                        self.movement.update_cell_position(ghost)
                        cell = (ghost.grid_x, ghost.grid_y)
                        if self.ghost_decision_cells.get(ghost.name) != cell:
                            decision_names.add(ghost.name)

            if controller is not None and decision_names:
                self.ghost_predictions = controller.predict(
                    self.maze,
                    em.pellets,
                    em.player,
                    em.ghosts,
                    self.movement,
                )
                for name in decision_names:
                    ghost = next(g for g in em.ghosts if g.name == name)
                    self.ghost_decision_cells[name] = (
                        ghost.grid_x,
                        ghost.grid_y,
                    )
                # self._print_model_decision(decision_names)

            for gst in em.ghosts:
                if gst.going_to_prison:
                    self.ghost_decision_sources[gst.name] = "PRISON"
                    if gst.prison_target is not None:
                        target_y, target_x = gst.prison_target

                        self.movement.update_ghost_to_target(
                            gst,
                            target_y,
                            target_x,
                        )
                        if (gst.grid_x, gst.grid_y) == (target_x, target_y):
                            gst.going_to_prison = False
                            gst.in_prison = True
                            gst.respawn_timer = after(10000)

                elif gst.in_prison:
                    self.ghost_decision_sources[gst.name] = "WAIT"
                    if expired(gst.respawn_timer):
                        gst.reset()
                        gst.is_eaten = False
                        gst.is_edible = False
                        gst.going_to_prison = False
                        gst.in_prison = False
                        gst.prison_target = None
                        gst.respawn_timer = 0.0

                        continue

                    self.movement.move_inside_prison(gst)
                elif gst.is_eaten:
                    self.ghost_decision_sources[gst.name] = "RESPAWN"
                    self.movement.update_ghost_to_target(
                        gst,
                        gst.spawn_y,
                        gst.spawn_x,
                    )

                else:
                    predicted_direction = self.ghost_predictions.get(gst.name)
                    if controller is not None and predicted_direction is not None:
                        self.ghost_decision_sources[gst.name] = "CNN"
                        if gst.name in decision_names:
                            self.movement.update_cnn_ghost(
                                gst,
                                predicted_direction,
                            )
                        else:
                            self.movement.update_entity(gst)
                    elif gst.is_edible:
                        self.ghost_decision_sources[gst.name] = "RUNAWAY"
                        self.movement.update_runaway_ghost(gst, em.player)
                    else:
                        self.ghost_decision_sources[gst.name] = "BFS"
                        self.movement.update_bfs_ghost(gst, em.player)
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
        self._draw_ai_banner(screen)

    def _print_model_decision(self, decision_names: set[str]) -> None:
        """Print one JSON record for each distinct CNN decision state."""
        if self.ghost_controller is None:
            return

        em = self.game.entity_manager
        player = em.player
        diagnostics = self.ghost_controller.last_diagnostics
        ghosts = []
        for ghost in em.ghosts:
            if ghost.name not in decision_names:
                continue
            diagnostic = diagnostics.get(ghost.name)
            if diagnostic is None:
                continue
            bfs_result = self.movement.get_bfs_next_move(
                (ghost.grid_x, ghost.grid_y),
                (player.grid_x, player.grid_y),
            )
            bfs_direction = (
                bfs_result[0][0] if bfs_result is not None and bfs_result[0] else None
            )
            ghosts.append(
                {
                    "name": ghost.name,
                    "position": [ghost.grid_x, ghost.grid_y],
                    "mode": "FRIGHTENED" if ghost.is_edible else "CHASE",
                    "delta_to_player": [
                        player.grid_x - ghost.grid_x,
                        player.grid_y - ghost.grid_y,
                    ],
                    "manhattan_distance": (
                        abs(player.grid_x - ghost.grid_x)
                        + abs(player.grid_y - ghost.grid_y)
                    ),
                    "chosen": diagnostic["chosen"],
                    "confidence": round(diagnostic["confidence"], 4),
                    "probabilities": {
                        direction: round(probability, 4)
                        for direction, probability in diagnostic[
                            "probabilities"
                        ].items()
                    },
                    "legal_actions": diagnostic["legal"],
                    "bfs_teacher_direction": bfs_direction,
                    "matches_bfs": diagnostic["chosen"] == bfs_direction,
                }
            )

        record = {
            "event": "cnn_ghost_decision",
            "time_ms": pygame.time.get_ticks(),
            "player": {
                "position": [player.grid_x, player.grid_y],
                "live_direction": player.direction,
                "model_direction_input": player.direction,
                "powered": any(ghost.is_edible for ghost in em.ghosts),
            },
            "ghosts": ghosts,
        }
        print(json.dumps(record, separators=(",", ":")), flush=True)

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
        color: Any,
        seg_w: float,
    ) -> None:
        text = f"{icon} {value}"
        font = ui.get_scaled_font(text, max_width=int(seg_w - 8), base_size=26)
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
                max_width=int(seg_w * 0.3),
                base_size=20,
            )
            surf = font.render(extra, True, ui.COLOR_WHITE)
            screen.blit(
                surf,
                (start_x + total_w + 6, cy - surf.get_height() // 2),
            )
        elif lives <= 0:
            font = ui.get_scaled_font("0", int(seg_w - 8), base_size=22)
            surf = font.render("0", True, ui.COLOR_RED)
            screen.blit(surf, surf.get_rect(center=(cx, cy)))

    @staticmethod
    def _draw_pacman_icon(
        screen: pygame.Surface,
        center: tuple[float, float],
        radius: int,
        color: Any,
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
    def _draw_wall_segment(
        screen: pygame.Surface, p1: tuple[int, int], p2: tuple[int, int]
    ) -> None:
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
        assert ui.FONT_HUD is not None
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

    def _draw_ai_banner(self, screen: pygame.Surface) -> None:
        if not self.use_ai_player:
            return

        decision = self.ai_player_decision or "EVALUATING"
        diag = (
            getattr(self.player_controller, "last_diagnostics", {})
            if self.player_controller
            else {}
        )
        probs = diag.get("probabilities", {})

        probs_text = (
            " | ".join(
                f"{d}:{probs.get(d, 0.0)*100:.0f}%"
                for d in ("UP", "DOWN", "LEFT", "RIGHT")
            )
            if probs
            else "EVALUATING..."
        )

        line1 = f"🤖 SUPERVISED AI | MOVE: {decision}"
        line2 = f"PROBS: {probs_text}"

        font1 = ui.get_scaled_font(
            line1, max_width=screen.get_width() - 24, base_size=15
        )
        font2 = ui.get_scaled_font(
            line2, max_width=screen.get_width() - 24, base_size=13
        )

        surf1 = font1.render(line1, True, ui.COLOR_NEON_CYAN)
        surf2 = font2.render(line2, True, ui.COLOR_WHITE)

        w = max(surf1.get_width(), surf2.get_width()) + 20
        h = surf1.get_height() + surf2.get_height() + 10
        cx = screen.get_width() // 2
        bottom_y = (
            screen.get_height() - 26 if self.active_cheats else screen.get_height() - 8
        )

        bubble_rect = pygame.Rect(0, 0, w, h)
        bubble_rect.midbottom = (cx, bottom_y)

        bubble = pygame.Surface(bubble_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(bubble, (0, 0, 0, 200), bubble.get_rect(), border_radius=8)
        pygame.draw.rect(
            bubble,
            (*ui.COLOR_NEON_CYAN, 220),
            bubble.get_rect(),
            width=1,
            border_radius=8,
        )

        screen.blit(bubble, bubble_rect.topleft)
        screen.blit(surf1, surf1.get_rect(midtop=(cx, bubble_rect.top + 4)))
        screen.blit(
            surf2, surf2.get_rect(midtop=(cx, bubble_rect.top + 4 + surf1.get_height()))
        )

    def give_target(self, ghost: Any) -> None:
        if ghost.name == "Blinky":
            ghost.prison_target = min(
                ghost.prison_cells,
                key=lambda cell: cell[0],
            )

        elif ghost.name == "Pinky":
            ghost.prison_target = max(
                ghost.prison_cells,
                key=lambda cell: cell[0],
            )

        elif ghost.name == "Inky":
            ghost.prison_target = min(
                ghost.prison_cells,
                key=lambda cell: cell[0],
            )

        elif ghost.name == "Clyde":
            ghost.prison_target = max(
                ghost.prison_cells,
                key=lambda cell: cell[0],
            )

    def check_collision(self, player: Any, ghosts: List[Any]) -> None:
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

                    if ghost.prison_cells:
                        self.give_target(ghost)
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
