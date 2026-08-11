from collections import deque
import os
import random
from typing import Any

import pygame
import torch

from AI_arena.data.constants import ACTION_COUNT
from AI_arena.player.data.observation import format_player_observation

from AI_arena.player.constants import (
    DIRECTIONS,
    GHOST_RESPAWN_TICKS,
    GHOST_SPECS,
    LIVES,
    MAZE_HEIGHT_MAX,
    MAZE_HEIGHT_MIN,
    MAZE_STEP_MULTIPLIER,
    MAZE_WIDTH_MAX,
    MAZE_WIDTH_MIN,
    MAX_PHYSICS_TICKS,
)
from AI_arena.player.entity_factory import EntityFactory
from AI_arena.player.ghost_controller import GhostController
from AI_arena.player.rewards import RewardCalculator

from src.graphics.entitys.graphic_lib import PacmanMode, SpriteLibrary
from src.logic.config import CELL_SIZE
from src.logic.level_manager import LevelManager
from src.logic.movement import MovementSystem


class PacmanPlayerEnv:
    """Headless environment in which RL policy controls Pac-Man against 4 BFS ghosts."""

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int | None = None,
        stage: int = 1,
        device: str | torch.device = "cpu",
        use_bfs_shaping: bool = True,
        maze_w_min: int | None = None,  # ← NEW: curriculum override
        maze_w_max: int | None = None,  # ← NEW
        maze_h_min: int | None = None,  # ← NEW
        maze_h_max: int | None = None,  # ← NEW
    ) -> None:
        if not pygame.get_init():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
        if not pygame.display.get_surface():
            pygame.display.set_mode((1, 1))

        SpriteLibrary.instance().load(CELL_SIZE)
        SpriteLibrary.instance().load_ghosts(CELL_SIZE)

        self.stage = stage
        self.user_max_steps = max_steps
        self.max_steps = max_steps if max_steps is not None else 800
        self.step_count = 0
        self.seed = seed
        self.rng = random.Random(seed)

        self.maze: list[list[int]] | None = None
        self.movement: MovementSystem | None = None
        self.player = None
        self.ghosts: list = []
        self.pellets: list[list[int]] | None = None
        self.total_pellets = 0
        self.remaining_pellets = 0

        self.visited_tiles: set[tuple[int, int]] = set()
        self.last_action: int | None = None
        self.device = torch.device(device)

        self.last_cell: tuple[int, int] | None = None
        self.prev_prev_cell: tuple[int, int] | None = None

        self.cell_history: deque[tuple[int, int]] = deque(maxlen=6)
        self.pellet_history: deque[bool] = deque(maxlen=6)
        self.region_pellets: dict[tuple[int, int], int] = {}
        self.region_pellets_initial: dict[tuple[int, int], int] = {}  # ← NEW
        self.last_region: tuple[int, int] | None = None

        self.use_reverse_mask = False
        self.use_bfs_shaping = use_bfs_shaping
        self.bfs_shaping_gamma = 0.99

        self._pellet_dist_grid: list[list[int]] | None = None
        self._cached_potential: float = 0.0
        self._ghost_respawn_ticks: list[int] = [0] * 4
        self._osc_count = 0

        self.episode_event_counts: dict[str, int] = {}
        self.episode_reward_breakdown: dict[str, float] = {}
        self.ghost_confusion_prob = 0.90
        self.death_count = 0

        # ← NEW: spatial / temporal memory state
        self.visited_heatmap: list[list[float]] | None = None
        self.visited_recent: deque[tuple[int, int]] = deque(maxlen=8)
        self.prev_nearest_pellet_dist = -1
        self.prev_nearest_ghost_dist = -1
        self.prev_nearest_pp_dist = -1
        self.steps_since_pellet = 0
        self.last_positions: deque[tuple[int, int]] = deque(maxlen=3)
        self.just_died = 0.0
        self.same_action_count = 0

        # Curriculum overrides
        self._maze_w_min = maze_w_min
        self._maze_w_max = maze_w_max
        self._maze_h_min = maze_h_min
        self._maze_h_max = maze_h_max

        # ── Delegates ──
        self._reward_calc = RewardCalculator(stage=stage)
        self._ghost_ctrl = GhostController(movement=None, rng=self.rng)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def reset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset episode state and return (grid, features, valid_actions)."""
        self.step_count = 0
        self.visited_tiles = set()
        self.last_action = None
        self.last_cell = None
        self.prev_prev_cell = None
        self.cell_history.clear()
        self.pellet_history.clear()
        self.region_pellets = {}
        self.region_pellets_initial = {}
        self.last_region = None
        self._osc_count = 0
        self._ghost_respawn_ticks = [0] * 4
        self.death_count = 0

        self.episode_event_counts = {
            "pellet": 0,
            "super": 0,
            "osc": 0,
            "died": 0,
            "ghost_eaten": 0,
            "completed": 0,
            "truncated": 0,
        }
        self.episode_reward_breakdown = {
            "step": 0.0,
            "oscillation": 0.0,
            "pellet": 0.0,
            "super_pellet": 0.0,
            "ghost": 0.0,
            "complete": 0.0,
            "death": 0.0,
            "milestone": 0.0,
            "bfs": 0.0,
            "ghost_proximity": 0.0,
            "region_cleared": 0.0,
            "region_dirty": 0.0,
            "backtrack": 0.0,  # ← NEW
            "incomplete": 0.0,  # ← NEW
        }

        self._reward_calc.reset()

        # ← NEW: curriculum-friendly sizing
        mw_min = self._maze_w_min if self._maze_w_min is not None else MAZE_WIDTH_MIN
        mw_max = self._maze_w_max if self._maze_w_max is not None else MAZE_WIDTH_MAX
        mh_min = self._maze_h_min if self._maze_h_min is not None else MAZE_HEIGHT_MIN
        mh_max = self._maze_h_max if self._maze_h_max is not None else MAZE_HEIGHT_MAX

        maze_w = self.rng.randint(mw_min, mw_max)
        maze_h = self.rng.randint(mh_min, mh_max)
        current_seed = self.rng.randint(1, 44444)

        maze_gen = LevelManager.build_maze(maze_w, maze_h, seed=current_seed)
        self.maze = maze_gen.maze
        self.movement = MovementSystem(self.maze)
        self._ghost_ctrl.movement = self.movement

        # Dynamic max steps based on maze size
        maze_size = maze_w * maze_h
        if self.user_max_steps is None:
            self.max_steps = int(maze_size * MAZE_STEP_MULTIPLIER)
        else:
            self.max_steps = self.user_max_steps

        # Create Player and Ghosts
        self.player = EntityFactory.create_player(self.maze)
        self.ghosts = EntityFactory.create_ghosts(self.maze, GHOST_SPECS)

        # Create Pellets
        self._create_pellets()

        # ── BFS potential shaping ──
        if self.use_bfs_shaping:
            self._pellet_dist_grid = self._compute_pellet_distance_grid()
            self._cached_potential = self._potential_at(
                self.player.grid_y, self.player.grid_x
            )

        start_cell = (self.player.grid_y, self.player.grid_x)
        self.visited_tiles.add(start_cell)
        self.last_cell = start_cell
        self.last_region = (start_cell[0] // 4, start_cell[1] // 4)
        h, w = len(self.maze), len(self.maze[0])
        self.visit_counts: list[list[int]] = [[0 for _ in range(w)] for _ in range(h)]
        self.visit_counts[start_cell[0]][start_cell[1]] = 1
        # ← NEW: init spatial memory
        h, w = len(self.maze), len(self.maze[0])
        self.visited_heatmap = [[0.0 for _ in range(w)] for _ in range(h)]
        self.visited_heatmap[start_cell[0]][start_cell[1]] = 1.0
        self.visited_recent.clear()
        self.visited_recent.append(start_cell)
        self.prev_nearest_pellet_dist = -1
        self.prev_nearest_ghost_dist = -1
        self.prev_nearest_pp_dist = -1
        self.steps_since_pellet = 0
        self.last_positions.clear()
        self.just_died = 0.0
        self.same_action_count = 0

        return self._get_observation()

    def step(
        self,
        action: int | torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        float,
        bool,
        dict[str, Any],
    ]:
        """Apply one action and advance physics until Pac-Man reaches a cell center."""
        if isinstance(action, torch.Tensor):
            action = int(action.item())

        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"Invalid action index {action}")

        # ← NEW: stuck / repetition detector
        if self.last_action is not None and action == self.last_action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0

        _, _, valid_actions = self._get_observation()
        if not bool(valid_actions[0, action]):
            legal = torch.where(valid_actions[0])[0].tolist()
            if legal:
                action = self.rng.choice(legal)

        assert self.player is not None
        self.player.next_direction = DIRECTIONS[action]
        self.last_action = action
        events = {
            "pellet_eaten": False,
            "super_pellet_eaten": False,
            "ghost_eaten": False,
            "pacman_died": False,
            "level_completed": False,
            "oscillating": False,
            "left_dirty_region": False,
            "cleared_region": False,
            "backtracked": False,  # ← NEW
        }

        start_cell = (self.player.grid_y, self.player.grid_x)

        # ← NEW: snapshot distances BEFORE moving (for delta features)
        if self.movement is not None and self.maze is not None:
            py0, px0 = self.player.grid_y, self.player.grid_x
            bfs0 = self.movement.bfs_distances((py0, px0))
            w0 = len(self.maze[0])

            # nearest normal pellet
            np_dists = [
                bfs0[gy * w0 + gx]
                for gy in range(len(self.maze))
                for gx in range(w0)
                if self.pellets[gy][gx] == 1
                and 0 <= gy * w0 + gx < len(bfs0)
                and bfs0[gy * w0 + gx] >= 0
            ]
            self.prev_nearest_pellet_dist = min(np_dists) if np_dists else -1

            # nearest active ghost
            ghost_dists = [
                bfs0[g.grid_y * w0 + g.grid_x]
                for g in self.ghosts
                if not g.in_prison
                and not g.is_edible
                and 0 <= g.grid_y * w0 + g.grid_x < len(bfs0)
                and bfs0[g.grid_y * w0 + g.grid_x] >= 0
            ]
            self.prev_nearest_ghost_dist = min(ghost_dists) if ghost_dists else -1

            # nearest power pellet
            pp_dists = [
                bfs0[gy * w0 + gx]
                for gy in range(len(self.maze))
                for gx in range(w0)
                if self.pellets[gy][gx] == 2
                and 0 <= gy * w0 + gx < len(bfs0)
                and bfs0[gy * w0 + gx] >= 0
            ]
            self.prev_nearest_pp_dist = min(pp_dists) if pp_dists else -1

        # Active ghost distance before step
        min_ghost_dist_before = -1
        if (
            self.stage > 1
            and self.movement is not None
            and self.player is not None
            and self.maze is not None
        ):
            py_b, px_b = self.player.grid_y, self.player.grid_x
            bfs_b = self.movement.bfs_distances((py_b, px_b))
            w_b = len(self.maze[0])
            active_dists = [
                bfs_b[g.grid_y * w_b + g.grid_x]
                for g in self.ghosts
                if not g.in_prison
                and not g.is_edible
                and 0 <= g.grid_y * w_b + g.grid_x < len(bfs_b)
                and bfs_b[g.grid_y * w_b + g.grid_x] >= 0
            ]
            if active_dists:
                min_ghost_dist_before = min(active_dists)

        potential_before = self._cached_potential if self.use_bfs_shaping else 0.0
        cell_changed = False

        for _ in range(MAX_PHYSICS_TICKS):
            prev_grid = (self.player.grid_x, self.player.grid_y)
            self._update_entities()

            if prev_grid != (self.player.grid_x, self.player.grid_y):
                tick_events = self._check_events()
                for key, val in tick_events.items():
                    if val:
                        events[key] = True
                if events["pacman_died"] or events["level_completed"]:
                    break

            current_cell = (self.player.grid_y, self.player.grid_x)
            if current_cell != start_cell and self.movement.is_centered(self.player):
                cell_changed = True
                break

        current_pos = (self.player.grid_y, self.player.grid_x)

        # Active ghost distance after step
        min_ghost_dist_after = -1
        threat_dist = float("inf")
        if (
            self.stage > 1
            and self.movement is not None
            and self.player is not None
            and self.maze is not None
        ):
            py_a, px_a = self.player.grid_y, self.player.grid_x
            bfs_a = self.movement.bfs_distances((py_a, px_a))
            w_a = len(self.maze[0])
            active_dists = [
                bfs_a[g.grid_y * w_a + g.grid_x]
                for g in self.ghosts
                if not g.in_prison
                and not g.is_edible
                and 0 <= g.grid_y * w_a + g.grid_x < len(bfs_a)
                and bfs_a[g.grid_y * w_a + g.grid_x] >= 0
            ]
            if active_dists:
                min_ghost_dist_after = min(active_dists)
                threat_dist = float(min_ghost_dist_after)

        # ── Anti-oscillation & backtracking tracking ──
        # ── Oscillation & loop detection ──
        if cell_changed:
            history = list(self.cell_history)

            # 1) Classic 2-cell flip A->B->A
            if self.prev_prev_cell is not None and current_pos == self.prev_prev_cell:
                events["oscillating"] = True
                self._osc_count += 1

            # 2) 4-cell loop A->B->C->D->A (only if no pellet eaten this step)
            elif len(history) >= 4:
                if current_pos == history[-4]:
                    if (
                        self.pellets
                        and self.pellets[current_pos[0]][current_pos[1]] == 0
                    ):
                        events["oscillating"] = True
                        self._osc_count += 1

            self.prev_prev_cell = self.last_cell
            self.last_cell = current_pos
            self.cell_history.append(current_pos)

            # 3) Mild inefficiency: revisiting a recent empty cell
            if (
                current_pos in self.visited_recent
                and self.pellets[current_pos[0]][current_pos[1]] == 0
            ):
                if threat_dist > 5 and (
                    self.player is None or self.player.powered_timer <= 0
                ):
                    events["backtracked"] = True

            # Increment visit count
            self.visit_counts[current_pos[0]][current_pos[1]] += 1

            self.visited_recent.append(current_pos)
            self.last_positions.append(current_pos)

            # Hunger counter
            self.steps_since_pellet += 1
            if events["pellet_eaten"] or events["super_pellet_eaten"]:
                self.steps_since_pellet = 0

            # ── Region Tracking (4x4) ──
            curr_region = (current_pos[0] // 4, current_pos[1] // 4)
            if self.last_region is not None and curr_region != self.last_region:
                rem_old = self.region_pellets.get(self.last_region, 0)
                if 0 < rem_old <= 2:
                    events["left_dirty_region"] = True
                elif rem_old == 0:
                    events["cleared_region"] = True
            self.last_region = curr_region

        if events["pacman_died"]:
            self.last_cell = None
            self.prev_prev_cell = None
            self.last_region = (current_pos[0] // 4, current_pos[1] // 4)
            # ← NEW: tell next observation that death just happened
            self.just_died = 1.0

        # ← NEW: decay death flag
        if self.just_died > 0:
            self.just_died = max(0.0, self.just_died - 0.05)

        # ── BFS potential shaping ──
        bfs_shaping = 0.0
        if self.use_bfs_shaping:
            if events["pellet_eaten"] or events["super_pellet_eaten"]:
                self._pellet_dist_grid = self._compute_pellet_distance_grid()
            potential_after = self._potential_at(*current_pos)
            bfs_shaping = self.bfs_shaping_gamma * potential_after - potential_before
            self._cached_potential = potential_after

        # ← NEW: mark truncation in events so reward calc can penalize incomplete
        truncated = self.step_count >= self.max_steps
        events["truncated"] = truncated

        reward, breakdown = self._reward_calc.calculate(
            events=events,
            bfs_shaping=bfs_shaping,
            total_pellets=self.total_pellets,
            remaining_pellets=self.remaining_pellets,
            step_count=self.step_count,
            max_steps=self.max_steps,
            player=self.player,
            ghosts=self.ghosts,
            movement=self.movement,
            maze=self.maze,
            threat_dist=threat_dist,
            min_ghost_dist_after=min_ghost_dist_after,
            min_ghost_dist_before=min_ghost_dist_before,
        )
        for key, val in breakdown.items():
            self.episode_reward_breakdown[key] += val

        if events["pellet_eaten"]:
            self.episode_event_counts["pellet"] += 1
        if events["super_pellet_eaten"]:
            self.episode_event_counts["super"] += 1
        if events["oscillating"]:
            self.episode_event_counts["osc"] += 1
        if events["pacman_died"]:
            self.episode_event_counts["died"] += 1
            self.death_count += 1
        if events.get("ghost_eaten", False):
            self.episode_event_counts["ghost_eaten"] += 1
        if events["level_completed"]:
            self.episode_event_counts["completed"] += 1

        self.step_count += 1

        terminated = bool(
            self.episode_event_counts["died"] > LIVES or events["level_completed"]
        )
        done = terminated or truncated

        pellets_eaten = self.total_pellets - self.remaining_pellets
        completion_pct = (
            (pellets_eaten / self.total_pellets * 100.0)
            if self.total_pellets > 0
            else 0.0
        )
        if truncated:
            self.episode_event_counts["truncated"] += 1

        min_ghost_dist = -1
        if self.stage > 1 and self.movement is not None and self.player is not None:
            py2, px2 = self.player.grid_y, self.player.grid_x
            bfs_from_player = self.movement.bfs_distances((py2, px2))
            w2 = len(self.maze[0]) if self.maze else 1
            active_dists = [
                bfs_from_player[g.grid_y * w2 + g.grid_x]
                for g in self.ghosts
                if not g.in_prison
                and 0 <= g.grid_y * w2 + g.grid_x < len(bfs_from_player)
                and bfs_from_player[g.grid_y * w2 + g.grid_x] >= 0
            ]
            if active_dists:
                min_ghost_dist = min(active_dists)

        info = {
            "step": self.step_count,
            "terminated": terminated,
            "truncated": truncated,
            "events": events,
            "pellets_eaten": pellets_eaten,
            "total_pellets": self.total_pellets,
            "remaining_pellets": self.remaining_pellets,
            "completion_pct": completion_pct,
            "stage": self.stage,
            "maze": (len(self.maze[0]), len(self.maze)) if self.maze else (0, 0),
            "min_ghost_dist": min_ghost_dist,
        }
        if done:
            info["episode_event_counts"] = dict(self.episode_event_counts)
            info["episode_reward_breakdown"] = dict(self.episode_reward_breakdown)
        return self._get_observation(), reward, done, info

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _create_pellets(self) -> None:
        if self.maze is None:
            raise RuntimeError("Maze must be created first.")

        height = len(self.maze)
        width = len(self.maze[0])

        pellets = [[0] * width for _ in range(height)]
        corners = [
            (0, 0),
            (0, width - 1),
            (height - 1, 0),
            (height - 1, width - 1),
        ]
        center = (width // 2, height // 2)
        player_spawn = (
            (self.player.grid_x, self.player.grid_y)
            if self.player is not None
            else None
        )

        total = 0
        for y in range(height):
            for x in range(width):
                if self.maze[y][x] == 15:
                    pellets[y][x] = 0
                elif (x, y) == center or (
                    player_spawn is not None and (x, y) == player_spawn
                ):
                    pellets[y][x] = 0
                elif (x, y) in corners:
                    pellets[y][x] = 2
                    total += 1
                else:
                    pellets[y][x] = 1
                    total += 1

        self.pellets = pellets
        self.total_pellets = total
        self.remaining_pellets = total

        reg_dict: dict[tuple[int, int], int] = {}
        for y in range(height):
            for x in range(width):
                if pellets[y][x] in (1, 2):
                    r = (y // 4, x // 4)
                    reg_dict[r] = reg_dict.get(r, 0) + 1
        self.region_pellets = reg_dict
        self.region_pellets_initial = dict(reg_dict)  # ← NEW

    def _update_entities(self) -> None:
        if self.movement is None or self.player is None or self.maze is None:
            return

        self.movement.update_entity(self.player)

        if self.player.powered_timer > 0:
            self.player.powered_timer -= 0.1
            if self.player.powered_timer <= 0:
                self.player.end_powered_mode()
                for ghost in self.ghosts:
                    ghost.is_edible = False

        self._ghost_ctrl.update(
            ghosts=self.ghosts,
            player=self.player,
            stage=self.stage,
            ghost_respawn_ticks=self._ghost_respawn_ticks,
            ghost_confusion_prob=self.ghost_confusion_prob,
        )

    def _check_events(self) -> dict[str, bool]:
        events = {
            "pellet_eaten": False,
            "super_pellet_eaten": False,
            "ghost_eaten": False,
            "pacman_died": False,
            "level_completed": False,
        }

        if self.player is None or self.pellets is None or self.maze is None:
            return events

        py, px = self.player.grid_y, self.player.grid_x
        h, w = len(self.maze), len(self.maze[0])

        if 0 <= py < h and 0 <= px < w:
            pellet_type = self.pellets[py][px]
            if pellet_type == 1:
                self.pellets[py][px] = 0
                self.remaining_pellets -= 1
                events["pellet_eaten"] = True
                reg = (py // 4, px // 4)
                if reg in self.region_pellets and self.region_pellets[reg] > 0:
                    self.region_pellets[reg] -= 1
            elif pellet_type == 2:
                self.pellets[py][px] = 0
                self.remaining_pellets -= 1
                events["super_pellet_eaten"] = True
                reg = (py // 4, px // 4)
                if reg in self.region_pellets and self.region_pellets[reg] > 0:
                    self.region_pellets[reg] -= 1
                self.player.start_powered_mode(mode=PacmanMode.PUNCH, duration=30.0)
                for ghost in self.ghosts:
                    ghost.is_edible = True

        if self.remaining_pellets <= 0:
            events["level_completed"] = True
        if self.stage == 1:
            return events

        for idx, ghost in enumerate(self.ghosts):
            if ghost.in_prison:
                continue

            dist_sq = (self.player.x - ghost.x) ** 2 + (self.player.y - ghost.y) ** 2
            if (ghost.grid_y == py and ghost.grid_x == px) or dist_sq <= (
                CELL_SIZE * 0.6
            ) ** 2:
                if ghost.is_edible:
                    events["ghost_eaten"] = True
                    ghost.is_edible = False
                    ghost.in_prison = True
                    ghost.reset()
                    ghost._tick_accumulator = 0.0
                    self._ghost_respawn_ticks[idx] = GHOST_RESPAWN_TICKS
                else:
                    events["pacman_died"] = True
                    self.player.reset_location()
                    for ghost in self.ghosts:
                        ghost.reset()
                    break

        return events

    def _compute_pellet_distance_grid(self) -> list[list[int]]:
        from collections import deque

        h, w = len(self.maze), len(self.maze[0])
        dist = [[-1] * w for _ in range(h)]
        q: deque[tuple[int, int]] = deque()
        for y in range(h):
            for x in range(w):
                if self.pellets[y][x] in (1, 2):
                    dist[y][x] = 0
                    q.append((y, x))
        while q:
            y, x = q.popleft()
            d = dist[y][x]
            for ny, nx in self.movement.get_neighbors(y, x):
                if dist[ny][nx] == -1:
                    dist[ny][nx] = d + 1
                    q.append((ny, nx))
        return dist

    def _potential_at(self, y: int, x: int) -> float:
        if self.remaining_pellets <= 0 or self._pellet_dist_grid is None:
            return 0.0
        d = self._pellet_dist_grid[y][x]
        if d < 0:
            sentinel = len(self.maze) + len(self.maze[0])
            return -float(sentinel)
        return -float(d)

    def _get_observation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.maze is None
            or self.player is None
            or self.pellets is None
            or self.movement is None
        ):
            raise RuntimeError("Environment has not been initialized.")

        # ← NEW: compute region state for extra features
        py, px = self.player.grid_y, self.player.grid_x
        curr_region = (py // 4, px // 4)
        region_total = self.region_pellets_initial.get(curr_region, 0)
        region_remaining = self.region_pellets.get(curr_region, 0)
        region_completion_frac = 1.0 - (region_remaining / max(region_total, 1))
        region_is_dirty = 1.0 if (0 < region_remaining <= 2) else 0.0

        grid, extra_features, valid_player_actions = format_player_observation(
            maze=self.maze,
            pellets=self.pellets,
            player=self.player,
            ghosts=self.ghosts,
            movement=self.movement,
            initial_pellet_count=self.total_pellets,
            device=self.device,
            visit_counts=self.visit_counts,
            prev_nearest_pellet_dist=self.prev_nearest_pellet_dist,
            prev_nearest_ghost_dist=self.prev_nearest_ghost_dist,
            prev_nearest_pp_dist=self.prev_nearest_pp_dist,
            steps_since_pellet=self.steps_since_pellet,
            last_positions=list(self.last_positions)[:-1],  # exclude current
            just_died=self.just_died,
            same_action_count=self.same_action_count,
            region_completion_frac=region_completion_frac,
            region_is_dirty=region_is_dirty,
        )

        if self.last_action is not None and self.use_reverse_mask:
            rev = self._reverse_action(self.last_action)
            if valid_player_actions[0, rev]:
                if valid_player_actions.sum().item() > 1:
                    valid_player_actions = valid_player_actions.clone()
                    valid_player_actions[0, rev] = False

        return grid, extra_features, valid_player_actions

    def _reverse_action(self, action: int) -> int:
        return {0: 1, 1: 0, 2: 3, 3: 2}[action]
