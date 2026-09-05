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
    ESCAPE_CONFIRM_STEPS,
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
        use_bfs_shaping: bool = False,
        maze_w_min: int | None = None,  # ← NEW: curriculum override
        maze_w_max: int | None = None,  # ← NEW
        maze_h_min: int | None = None,  # ← NEW
        maze_h_max: int | None = None,  # ← NEW
        start_pellets: tuple[int, ...] | None = None,  # ← completion curriculum
        ghost_speed_ratio: float = 0.5,
        ghost_confusion_prob: float = 0.0,
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
        self.cleared_regions: set[tuple[int, int]] = set()
        self.last_region: tuple[int, int] | None = None

        self.use_bfs_shaping = use_bfs_shaping
        self.bfs_shaping_gamma = 0.99

        self._pellet_dist_grid: list[list[int]] | None = None
        self._cached_potential: float = 0.0
        self._ghost_respawn_ticks: list[int] = [0] * 4
        self._osc_count = 0

        self.episode_event_counts: dict[str, int] = {}
        self.episode_reward_breakdown: dict[str, float] = {}
        self.episode_telemetry: dict[str, float] = {}
        self._in_corner_threat = False
        self._open_escape_deadline = -1
        self.ghost_speed_ratio = ghost_speed_ratio
        self.ghost_confusion_prob = ghost_confusion_prob
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

        # Completion curriculum: when set, each episode starts with one of
        # these pellet counts (chosen per reset) instead of the full map.
        self.start_pellets = start_pellets

        # ── Dead-End Trap Lookahead State ──
        self._dead_end_traps: dict[tuple[tuple[int, int], str], dict[str, Any]] = {}
        self._cell_to_trap: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

        # ── Delegates ──
        self._reward_calc = RewardCalculator(stage=stage)
        self._ghost_ctrl = GhostController(
            movement=None, rng=self.rng, ghost_speed_ratio=ghost_speed_ratio
        )

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
            "exploration": 0,
        }
        self.episode_telemetry = {
            "cornered_steps": 0.0,
            "cornered_entries": 0.0,
            "escape_success": 0.0,
            "escape_failure": 0.0,
            "deaths_cornered": 0.0,
            "min_ghost_dist_sum": 0.0,
            "min_ghost_dist_cnt": 0.0,
            "approach_steps": 0.0,
        }
        self._in_corner_threat = False
        self._open_escape_deadline = -1
        self.episode_reward_breakdown = {
            "step": 0.0,
            "oscillation": 0.0,
            "momentum": 0.0,
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
            "hunger": 0.0,
            "incomplete": 0.0,  # ← NEW
            "predictive_threat": 0.0,
            "evasion_skill": 0.0,
            "super_bait": 0.0,
            "zone_control": 0.0,
            "threat_mastery": 0.0,
            "ghost_lure": 0.0,
            "survival_truncation": 0.0,
            "exploration": 0.0,
            "zone_stagnation": 0.0,
            "bypassed_pellet": 0.0,
        }

        self._reward_calc.reset()
        # self.reward_calculator.reset()
        # ← NEW: curriculum-friendly sizing
        mw_min = self._maze_w_min if self._maze_w_min is not None else MAZE_WIDTH_MIN
        mw_max = self._maze_w_max if self._maze_w_max is not None else MAZE_WIDTH_MAX
        mh_min = self._maze_h_min if self._maze_h_min is not None else MAZE_HEIGHT_MIN
        mh_max = self._maze_h_max if self._maze_h_max is not None else MAZE_HEIGHT_MAX

        maze_w = self.rng.randint(mw_min, mw_max)
        maze_h = self.rng.randint(mh_min, mh_max)
        current_seed = self.rng.randint(1, 44444)
        # REMOVE
        # maze_w = 25
        # maze_h = 20
        # fixed_seeds = [20, 77, 1337, 42, 100]
        # current_seed = fixed_seeds[self.rng.randint(0, len(fixed_seeds) - 1)]

        maze_gen = LevelManager.build_maze(maze_w, maze_h, seed=current_seed)
        self.maze = maze_gen.maze
        self.movement = MovementSystem(self.maze)
        # MovementSystem owns an unseeded Random() used for frightened-ghost
        # flee targets — reseed it so identical seeds give identical ghosts.
        self.movement.rng.seed(current_seed)
        self._ghost_ctrl.movement = self.movement
        self._analyze_dead_ends()

        # Create Player and Ghosts
        self.player = EntityFactory.create_player(self.maze)
        self.ghosts = EntityFactory.create_ghosts(self.maze, GHOST_SPECS)
        if self.stage == 1:
            for g in self.ghosts:
                g.in_prison = True
                g.is_edible = False

        # Create Pellets
        n_pellets = None
        if self.start_pellets:
            n_pellets = int(self.rng.choice(list(self.start_pellets)))
        self._create_pellets(n_pellets)

        # Dynamic max steps based on maze size and actual pellet count
        maze_size = maze_w * maze_h
        if self.user_max_steps is None:
            base_steps = int(maze_size * MAZE_STEP_MULTIPLIER)
            if self.start_pellets is not None:
                # Curriculum mode: cap proportionally to actual pellets spawned so episodes end fast if agent gets lost
                n_actual = (
                    len(self.pellets)
                    if self.pellets
                    else (max(self.start_pellets) if self.start_pellets else 8)
                )
                tight_budget = n_actual * 65 + (maze_w + maze_h) * 4
                self.max_steps = min(base_steps, max(150, tight_budget))
            else:
                self.max_steps = base_steps
        else:
            self.max_steps = self.user_max_steps

        # ── BFS potential shaping ──
        if self.use_bfs_shaping:
            self._pellet_dist_grid = self._compute_pellet_distance_grid()
            self._cached_potential = self._potential_at(
                self.player.grid_y, self.player.grid_x
            )
            self._last_pellet_grid_frac = 1.0  # track fraction for throttled recompute

        start_cell = (self.player.grid_y, self.player.grid_x)
        self.visited_tiles.add(start_cell)
        self.last_cell = start_cell
        self.last_region = (start_cell[0] // 4, start_cell[1] // 4)
        self.cleared_regions.clear()
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
        explore: bool = False,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        float,
        bool,
        dict[str, Any],
    ]:
        """Apply one action and advance physics until Pac-Man reaches a cell center.

        `explore=True` marks ε-explorer steps: behavioral penalties that would
        punish pure exploration noise (oscillation) are skipped for them.
        """
        if self.just_died > 0:
            self.just_died = 0.0

        if isinstance(action, torch.Tensor):
            action = int(action.item())

        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"Invalid action index {action}")

        # Stuck / repetition detector
        if self.last_action is not None and action == self.last_action:
            self.same_action_count += 1
        else:
            self.same_action_count = 0

        assert self.player is not None
        self.player.next_direction = DIRECTIONS[action]
        prev_action = self.last_action
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
            "bypassed_pellet": False,  # ← NEW
        }

        start_cell = (self.player.grid_y, self.player.grid_x)
        start_y, start_x = start_cell

        had_adjacent_pellet = False
        if self.movement is not None and self.maze is not None and self.pellets is not None:
            h_m = len(self.maze)
            w_m = len(self.maze[0]) if h_m > 0 else 0
            for d_name, (dy_o, dx_o) in (("UP", (-1, 0)), ("DOWN", (1, 0)), ("LEFT", (0, -1)), ("RIGHT", (0, 1))):
                if self.movement.can_move(start_y, start_x, d_name):
                    ny_o, nx_o = start_y + dy_o, start_x + dx_o
                    if 0 <= ny_o < h_m and 0 <= nx_o < w_m and self.pellets[ny_o][nx_o] == 1:
                        had_adjacent_pellet = True
                        break

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
        min_ghost_dist_before = self.prev_nearest_ghost_dist if self.stage > 1 else -1

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
        OPPOSITE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}
        is_action_reversal = (
            prev_action is not None
            and action == OPPOSITE_ACTION.get(prev_action, -1)
        )

        if cell_changed:
            history = list(self.cell_history)

            # 2-cell direction reversal A -> B -> A is an oscillation ONLY if:
            # - tile B (self.last_cell) was NOT a dead end (>1 walkable exit)
            # - no pellet was eaten on this step
            # - not fleeing a close ghost (threat_dist > 3 or powered)
            if self.prev_prev_cell is not None and current_pos == self.prev_prev_cell:
                last_c_exits = 0
                if self.last_cell is not None and self.movement is not None:
                    last_c_exits = sum(
                        1
                        for d in DIRECTIONS
                        if self.movement.can_move(self.last_cell[0], self.last_cell[1], d)
                    )
                is_dead_end = last_c_exits <= 1
                is_fleeing = threat_dist <= 6 and (
                    self.player is not None and self.player.powered_timer <= 0
                )
                if not is_dead_end and not is_fleeing and not (
                    events["pellet_eaten"] or events["super_pellet_eaten"]
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
                elif rem_old == 0 and self.last_region not in self.cleared_regions and self.region_pellets_initial.get(self.last_region, 0) > 0:
                    events["cleared_region"] = True
                    self.cleared_regions.add(self.last_region)
            self.last_region = curr_region
        else:
            # If Pac-Man did not advance to a new cell center (e.g. wall bump or stopped mid-tile)
            # and flipped directions or toggled repeatedly on an empty cell:
            if is_action_reversal and not (
                events["pellet_eaten"] or events["super_pellet_eaten"]
            ):
                events["oscillating"] = True
                self._osc_count += 1

        if had_adjacent_pellet and not (
            events["pellet_eaten"] or events["super_pellet_eaten"] or events["pacman_died"]
        ):
            events["bypassed_pellet"] = True

        if events["pacman_died"]:
            spawn_cell = (self.player.grid_y, self.player.grid_x)
            self.last_cell = spawn_cell
            self.prev_prev_cell = None
            self.cell_history.clear()
            self.cell_history.append(spawn_cell)
            self.visited_recent.clear()
            self.visited_recent.append(spawn_cell)
            self.last_positions.clear()
            self.last_positions.append(spawn_cell)
            self.steps_since_pellet = 0
            self.same_action_count = 0
            self.last_action = None
            self.last_region = (spawn_cell[0] // 4, spawn_cell[1] // 4)
            self.just_died = 1.0

            # Re-initialize spatial memory & visit grids on death
            self.visited_tiles.clear()
            self.visited_tiles.add(spawn_cell)
            if self.maze is not None:
                h_m, w_m = len(self.maze), len(self.maze[0])
                self.visit_counts = [[0 for _ in range(w_m)] for _ in range(h_m)]
                self.visit_counts[spawn_cell[0]][spawn_cell[1]] = 1
                self.visited_heatmap = [[0.0 for _ in range(w_m)] for _ in range(h_m)]
                self.visited_heatmap[spawn_cell[0]][spawn_cell[1]] = 1.0

            if self.use_bfs_shaping:
                self._pellet_dist_grid = self._compute_pellet_distance_grid()
                self._cached_potential = self._potential_at(*spawn_cell)

            if self.movement is not None and self.maze is not None:
                bfs_sp = self.movement.bfs_distances(spawn_cell)
                w_sp = len(self.maze[0])
                np_dists = [
                    bfs_sp[gy * w_sp + gx]
                    for gy in range(len(self.maze))
                    for gx in range(w_sp)
                    if self.pellets[gy][gx] == 1
                    and 0 <= gy * w_sp + gx < len(bfs_sp)
                    and bfs_sp[gy * w_sp + gx] >= 0
                ]
                self.prev_nearest_pellet_dist = min(np_dists) if np_dists else -1

                ghost_dists = [
                    bfs_sp[g.grid_y * w_sp + g.grid_x]
                    for g in self.ghosts
                    if not g.in_prison
                    and not g.is_edible
                    and 0 <= g.grid_y * w_sp + g.grid_x < len(bfs_sp)
                    and bfs_sp[g.grid_y * w_sp + g.grid_x] >= 0
                ]
                self.prev_nearest_ghost_dist = min(ghost_dists) if ghost_dists else -1

                pp_dists = [
                    bfs_sp[gy * w_sp + gx]
                    for gy in range(len(self.maze))
                    for gx in range(w_sp)
                    if self.pellets[gy][gx] == 2
                    and 0 <= gy * w_sp + gx < len(bfs_sp)
                    and bfs_sp[gy * w_sp + gx] >= 0
                ]
                self.prev_nearest_pp_dist = min(pp_dists) if pp_dists else -1

        if self.just_died > 0:
            self.just_died = max(0.0, self.just_died - 0.05)

        # ── BFS potential shaping ──
        bfs_shaping = 0.0
        if self.use_bfs_shaping:
            if events["pacman_died"]:
                bfs_shaping = 0.0
            else:
                if events["pellet_eaten"] or events["super_pellet_eaten"]:
                    self._pellet_dist_grid = self._compute_pellet_distance_grid()
                potential_after = self._potential_at(*current_pos)
                bfs_shaping = self.bfs_shaping_gamma * potential_after - potential_before
                self._cached_potential = potential_after

        # ← NEW: mark truncation in events so reward calc can penalize incomplete
        truncated = self.step_count >= self.max_steps
        events["truncated"] = truncated

        # Check if an uneaten super pellet is within Manhattan distance 4 of player
        super_pellet_nearby = False
        if self.pellets is not None and self.player is not None:
            py_s, px_s = self.player.grid_y, self.player.grid_x
            h_p, w_p = len(self.pellets), len(self.pellets[0])
            for dy_s in range(-4, 5):
                for dx_s in range(-4, 5):
                    if abs(dy_s) + abs(dx_s) <= 4:
                        ny_s, nx_s = py_s + dy_s, px_s + dx_s
                        if 0 <= ny_s < h_p and 0 <= nx_s < w_p and self.pellets[ny_s][nx_s] == 2:
                            super_pellet_nearby = True
                            break
                if super_pellet_nearby:
                    break

        reward, breakdown = self._reward_calc.calculate(
            events=events,
            bfs_shaping=bfs_shaping,
            total_pellets=self.total_pellets,
            remaining_pellets=self.remaining_pellets,
            step_count=self.step_count,
            max_steps=self.max_steps,
            player=self.player,
            ghosts=self.ghosts,
            steps_since_pellet=self.steps_since_pellet,
            movement=self.movement,
            maze=self.maze,  # ← MUST pass maze for corner detection
            threat_dist=threat_dist,
            min_ghost_dist_after=min_ghost_dist_after,
            min_ghost_dist_before=min_ghost_dist_before,  # ← CRITICAL for evasion_skill
            same_action_count=self.same_action_count,
            explore_step=explore,
            super_pellet_nearby=super_pellet_nearby,
        )
        for key, val in breakdown.items():
            self.episode_reward_breakdown[key] += val

        if self.stage > 1 and self.maze is not None:
            self._update_telemetry(events, min_ghost_dist_before, min_ghost_dist_after)

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
            self.episode_event_counts["died"] >= max(1, LIVES - 1)
            or events["level_completed"]
        )
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        done = terminated or truncated

        pellets_eaten = self.total_pellets - self.remaining_pellets
        completion_pct = (
            (pellets_eaten / self.total_pellets * 100.0)
            if self.total_pellets > 0
            else 0.0
        )
        if truncated:
            self.episode_event_counts["truncated"] += 1

        min_ghost_dist = min_ghost_dist_after

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
            "max_steps": self.max_steps,
            "start_pellets": self.total_pellets,
        }
        if done:
            info["episode_event_counts"] = dict(self.episode_event_counts)
            info["episode_reward_breakdown"] = dict(self.episode_reward_breakdown)
            info["telemetry"] = dict(self.episode_telemetry)
        return self._get_observation(), reward, done, info, action

    def set_seed(self, seed: int) -> None:
        """Re-seed the episode RNG — used for deterministic evaluation runs."""
        self.seed = seed
        self.rng.seed(seed)

    def reset_fixed(self, maze_w: int, maze_h: int, maze_seed: int) -> tuple:
        """Reset to a specific, deterministic map.

        Pins the maze size and seed so the exact same layout is reproduced
        every time. The ghost/movement RNG is also re-seeded for full
        determinism. Used by the map-pool curriculum so the agent can practise
        the same map multiple times before moving on.
        """
        self._maze_w_min = maze_w
        self._maze_w_max = maze_w
        self._maze_h_min = maze_h
        self._maze_h_max = maze_h
        # Seed the RNG so that randint(maze_w, maze_w) and then
        # randint(maze_seed_min, maze_seed_max) produce our target values.
        # We bypass the RNG lottery by temporarily monkey-patching randint.
        _orig_randint = self.rng.randint

        _calls: list[int] = []

        def _fixed_randint(a: int, b: int) -> int:
            # First call → maze_w (already pinned via min==max, so a==b==maze_w)
            # Second call → maze_h (same)
            # Third call → current_seed (return our chosen seed)
            _calls.append(len(_calls))
            if len(_calls) <= 2:
                return _orig_randint(a, b)  # a == b so always returns the pinned dim
            if len(_calls) == 3:
                return maze_seed
            return _orig_randint(a, b)

        self.rng.randint = _fixed_randint  # type: ignore[method-assign]
        try:
            obs = self.reset()
        finally:
            self.rng.randint = _orig_randint  # type: ignore[method-assign]
            # Restore free-range sizing so normal reset() still works
            self._maze_w_min = None
            self._maze_w_max = None
            self._maze_h_min = None
            self._maze_h_max = None
        return obs

    def _update_telemetry(
        self,
        events: dict[str, bool],
        min_ghost_dist_before: int,
        min_ghost_dist_after: int,
    ) -> None:
        """Track leading indicators of trap-avoidance skill (cheap, no reward coupling).

        cornered+threatened = ≤1 open neighbour cell AND ≥1 hunting ghost within
        Manhattan distance 8. Exposure is luck; escaping it is the skill.
        """
        tel = self.episode_telemetry
        step_idx = self.step_count  # incremented later in step()

        if min_ghost_dist_after >= 0:
            tel["min_ghost_dist_sum"] += float(min_ghost_dist_after)
            tel["min_ghost_dist_cnt"] += 1.0

        powered = bool(self.player.powered_timer > 0)
        if (
            not powered
            and min_ghost_dist_before > 0
            and min_ghost_dist_after > 0
            and min_ghost_dist_after < min_ghost_dist_before
        ):
            tel["approach_steps"] += 1.0

        if events.get("pacman_died", False):
            # Death this step: attribute to the trap if we were cornered when
            # the step started, or within the escape-confirm window after one.
            if self._in_corner_threat or step_idx <= self._open_escape_deadline:
                tel["deaths_cornered"] += 1.0
                tel["escape_failure"] += 1.0
            self._in_corner_threat = False
            self._open_escape_deadline = -1
            return

        px, py = self.player.grid_x, self.player.grid_y
        threats = self._reward_calc.count_threatening(px, py, self.ghosts)
        in_danger = bool(
            threats > 0 and self._reward_calc.is_cornered(px, py, self.maze)
        )

        if in_danger:
            tel["cornered_steps"] += 1.0
            if not self._in_corner_threat:
                # Fresh entry into a trap; cancels any unresolved escape window.
                self._in_corner_threat = True
                tel["cornered_entries"] += 1.0
                self._open_escape_deadline = -1
        elif self._in_corner_threat:
            self._in_corner_threat = False
            self._open_escape_deadline = step_idx + ESCAPE_CONFIRM_STEPS
        elif 0 <= self._open_escape_deadline and step_idx >= self._open_escape_deadline:
            tel["escape_success"] += 1.0
            self._open_escape_deadline = -1

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _create_pellets(self, count: int | None = None) -> None:
        """Fill the maze with pellets.

        count=None → classic full map (normal pellets everywhere, super
        pellets in corners). count=N → completion curriculum: exactly N
        NORMAL pellets placed among the BFS-farthest walkable cells from the
        player spawn, so completing the level reduces to navigating to them.
        """
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

        if count is not None and count > 0:
            # Pick N walkable cells within a per-episode distance band
            # (near / mid / far relative to spawn). Normal pellets only, so
            # the completion signal isn't tangled with powered-mode hunting.
            # Tiered bands create a gradual difficulty ladder toward
            # long-range navigation instead of demanding it cold.
            py, px = self.player.grid_y, self.player.grid_x
            bfs = self.movement.bfs_distances((py, px))
            candidates = []
            for y in range(height):
                for x in range(width):
                    if pellets[y][x] in (1, 2):
                        d = bfs[y * width + x]
                        if d >= 0:
                            candidates.append((d, y, x))
            candidates.sort()
            bands = [(4, 9), (10, 17), (18, 10**9)]
            self.rng.shuffle(bands)
            chosen = None
            for lo, hi in bands:
                band_pool = [c for c in candidates if lo <= c[0] <= hi]
                if len(band_pool) >= count:
                    chosen = self.rng.sample(band_pool, count)
                    break
            if chosen is None:
                # No single band fits — spread uniformly over ALL candidates
                # (never fall back to far-only placement).
                chosen = self.rng.sample(candidates, min(count, len(candidates)))
            valid_sp_locs = [
                (y, x)
                for y in range(height)
                for x in range(width)
                if pellets[y][x] == 2
            ]
            pellets = [[0] * width for _ in range(height)]
            total = 0
            # Include 2 super pellets in curriculum mode so Pac-Man has defensive tools (only when count > 2)
            if valid_sp_locs and count > 2:
                sp_chosen = self.rng.sample(valid_sp_locs, min(2, len(valid_sp_locs)))
                for sy, sx in sp_chosen:
                    pellets[sy][sx] = 2
                    total += 1

            for _, gy, gx in chosen:
                if pellets[gy][gx] == 0:
                    pellets[gy][gx] = 1
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
                self.player.start_powered_mode(mode=PacmanMode.PUNCH, duration=45.0)
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

    def _get_min_ghost_distance(self):
        """Return minimum Manhattan distance to non-edible, non-prison ghosts."""
        if not self.ghosts:
            return -1

        px, py = self.player.grid_x, self.player.grid_y
        min_dist = float("inf")

        for ghost in self.ghosts:
            if ghost.in_prison or ghost.is_edible:
                continue
            dist = abs(ghost.grid_x - px) + abs(ghost.grid_y - py)
            min_dist = min(min_dist, dist)

        return min_dist if min_dist != float("inf") else -1

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
        from src.logic.config import EAST, NORTH, SOUTH, WEST

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        wall_bits = [NORTH, SOUTH, WEST, EAST]

        while q:
            y, x = q.popleft()
            d = dist[y][x]
            cell = self.maze[y][x]
            for i, (dy, dx) in enumerate(directions):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not (cell & wall_bits[i]):
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
            return -float(sentinel) ** 0.5
        return -float(d) ** 0.5

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

        valid_player_actions = self._apply_trap_lookahead_mask(valid_player_actions)
        extra_features[0, 12:16] = valid_player_actions[0].float()

        return grid, extra_features, valid_player_actions

    def _analyze_dead_ends(self) -> None:
        """Precompute all dead-end branches, their entrance bottleneck junctions, and depths."""
        if self.maze is None or self.movement is None:
            self._dead_end_traps = {}
            self._cell_to_trap = {}
            return

        h = len(self.maze)
        w = len(self.maze[0]) if h else 0
        dead_ends = [
            (y, x)
            for y in range(h)
            for x in range(w)
            if len(self.movement.get_neighbors(y, x)) == 1
        ]
        traps: dict[tuple[tuple[int, int], str], dict[str, Any]] = {}
        cell_to_trap: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}

        for de in dead_ends:
            curr = de
            visited = [curr]
            prev = None
            while True:
                nbrs = [n for n in self.movement.get_neighbors(curr[0], curr[1]) if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                deg = len(self.movement.get_neighbors(nxt[0], nxt[1]))
                visited.append(nxt)
                if deg >= 3:
                    junction = nxt
                    first_step = visited[-2]
                    d_name = self.movement.direction_to_next_cell(junction, first_step)
                    if d_name:
                        branch_cells = set(visited[:-1])
                        traps[(junction, d_name)] = {
                            "depth": len(visited) - 1,
                            "cells": branch_cells,
                            "junction": junction,
                        }
                        for d_idx, c in enumerate(reversed(visited[:-1])):
                            cell_to_trap[c] = (junction, d_idx + 1)
                    break
                prev = curr
                curr = nxt

        self._dead_end_traps = traps
        self._cell_to_trap = cell_to_trap

    def _apply_trap_lookahead_mask(
        self, valid_actions: torch.Tensor
    ) -> torch.Tensor:
        """Filter out candidate moves that enter sealed dead-ends with no escape."""
        if (
            self.stage <= 1
            or self.player is None
            or self.movement is None
            or self.maze is None
        ):
            return valid_actions
        # While powered, ghosts are edible, so dead-ends are safe
        if self.player.powered_timer > 0:
            return valid_actions

        py, px = self.player.grid_y, self.player.grid_x
        h = len(self.maze)
        w = len(self.maze[0]) if h else 0

        active_ghosts = [
            g for g in self.ghosts
            if not getattr(g, "in_prison", False) and not getattr(g, "is_edible", False)
        ]
        if not active_ghosts:
            return valid_actions

        orig_mask = valid_actions.clone()
        filtered_mask = valid_actions.clone()

        dir_offsets = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }

        speed = max(0.1, self.ghost_speed_ratio)

        for a_idx, d_name in enumerate(DIRECTIONS):
            if not orig_mask[0, a_idx]:
                continue

            dy, dx = dir_offsets[d_name]
            ny, nx = py + dy, px + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue

            # Case A: Moving from junction into a dead-end branch
            trap_key = ((py, px), d_name)
            if trap_key in self._dead_end_traps:
                trap_info = self._dead_end_traps[trap_key]
                depth = trap_info["depth"]
                t_pacman = 2.0 * depth
                j_dists = self.movement.bfs_distances((py, px))
                ghost_dists = [
                    j_dists[g.grid_y * w + g.grid_x]
                    for g in active_ghosts
                    if 0 <= g.grid_y * w + g.grid_x < len(j_dists)
                    and j_dists[g.grid_y * w + g.grid_x] >= 0
                ]
                if ghost_dists:
                    min_g_dist = min(ghost_dists)
                    t_ghost = min_g_dist / speed
                    if t_ghost <= t_pacman + 2.0:
                        filtered_mask[0, a_idx] = False

            # Case B: Already inside a dead-end branch
            elif (py, px) in self._cell_to_trap:
                junction, dist_to_junc = self._cell_to_trap[(py, px)]
                if (ny, nx) in self._cell_to_trap:
                    _, next_dist_to_junc = self._cell_to_trap[(ny, nx)]
                    if next_dist_to_junc > dist_to_junc:
                        j_dists = self.movement.bfs_distances(junction)
                        ghost_dists = [
                            j_dists[g.grid_y * w + g.grid_x]
                            for g in active_ghosts
                            if 0 <= g.grid_y * w + g.grid_x < len(j_dists)
                            and j_dists[g.grid_y * w + g.grid_x] >= 0
                        ]
                        if ghost_dists:
                            min_g_dist = min(ghost_dists)
                            t_ghost = min_g_dist / speed
                            t_pacman_escape = (next_dist_to_junc + 1) * 1.0
                            if t_ghost <= t_pacman_escape + 2.0:
                                filtered_mask[0, a_idx] = False

        if filtered_mask.any():
            return filtered_mask
        return orig_mask

    def _reverse_action(self, action: int) -> int:
        return {0: 1, 1: 0, 2: 3, 3: 2}[action]

    def get_search_action(self, horizon: int = 12) -> int:
        """Query the Chess-like Lookahead Search Planner for the best action."""
        from AI_arena.player.search_planner import PacmanLookaheadSearch

        searcher = PacmanLookaheadSearch(self, horizon=horizon)
        return searcher.get_best_action()

    def get_search_distribution(
        self, horizon: int = 12, temperature: float = 1.0
    ) -> torch.Tensor:
        """Query the Lookahead Search Planner for action distribution (for distillation)."""
        from AI_arena.player.search_planner import PacmanLookaheadSearch

        searcher = PacmanLookaheadSearch(self, horizon=horizon)
        return searcher.get_action_distribution(temperature=temperature)

