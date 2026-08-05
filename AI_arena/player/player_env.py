"""Headless Pac-Man environment for player reinforcement learning against BFS ghosts."""

from __future__ import annotations

import random
from typing import Any

import pygame

import torch

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    CNN_HEIGHT,
    CNN_WIDTH,
    EXTRA_FEATURE_COUNT,
    GHOST_COUNT,
)

from AI_arena.data.formatter import ObservationFormatter

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

from src.graphics.entitys.ghost import Ghost
from src.graphics.entitys.graphic_lib import PacmanMode as pm, SpriteLibrary
from src.graphics.entitys.player import Player
from src.logic.config import CELL_SIZE, EAST, NORTH, SOUTH, WEST
from src.logic.level_manager import LevelManager
from src.logic.movement import MovementSystem

GHOST_SPECS = [
    ("Blinky", (255, 0, 0)),
    ("Pinky", (255, 182, 193)),
    ("Inky", (0, 255, 255)),
    ("Clyde", (255, 165, 0)),
]


# Random maze size range — upper bounds are the CNN tensor dimensions so the
# observation grid never overflows.  Lower bound is the minimum playable size.
# CNN_WIDTH  = 25  (columns / x-axis)
# CNN_HEIGHT = 50  (rows / y-axis, but LevelManager caps height at 23 anyway)
MAZE_WIDTH_MIN = 5
# MAZE_WIDTH_MAX = CNN_WIDTH  # 25 — do not exceed CNN_WIDTH or formatter will fail
MAZE_WIDTH_MAX = 10

MAZE_HEIGHT_MIN = 5
MAZE_HEIGHT_MAX = 10
# MAZE_HEIGHT_MAX = min(CNN_HEIGHT, 23)  # 23 — game engine hard cap

# Physics safety cap: max ticks to advance before giving up on centering
MAX_PHYSICS_TICKS = 300

# Multiplier for episode step limits based on maze size (w * h)
# e.g. 2.5 * (10 * 10) = 250 max steps per episode.
MAZE_STEP_MULTIPLIER: float = 5


class PacmanPlayerEnv:
    """Headless environment in which RL policy controls Pac-Man against 4 BFS ghosts.

    Step semantics: each call to step() advances the physics simulation until
    Pac-Man reaches the *center* of the next grid cell (or hits a wall and
    stops).  Reward is computed exactly once per cell crossing, eliminating
    the 5-7× reward-spam that occurs when observing every pixel tick.
    """

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int | None = None,
        stage: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        import os

        if not pygame.get_init():
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pygame.init()
        if not pygame.display.get_surface():
            pygame.display.set_mode((1, 1))

        SpriteLibrary.instance().load(CELL_SIZE)
        SpriteLibrary.instance().load_ghosts(CELL_SIZE)

        self.stage = stage
        self.user_max_steps = max_steps
        self.max_steps = (
            max_steps if max_steps is not None else 800
        )  # measured in cell crossings
        self.step_count = 0
        self.seed = seed
        self.rng = random.Random(seed)

        self.maze: list[list[int]] | None = None
        self.movement: MovementSystem | None = None
        self.player: Player | None = None
        self.ghosts: list[Ghost] = []
        self.pellets: list[list[int]] | None = None
        self.total_pellets = 0
        self.remaining_pellets = 0

        self.visited_tiles: set[tuple[int, int]] = set()
        self.last_action: int | None = None
        self.device = torch.device(device)

        # Anti-oscillation tracking (last two cell positions)
        self.last_cell: tuple[int, int] | None = None
        self.prev_prev_cell: tuple[int, int] | None = None

        self.use_bfs_shaping = True
        self.bfs_shaping_coef = 3  # start small; this only nudges navigation,
        # it shouldn't compete with the +5/+10 eat rewards
        self.bfs_shaping_gamma = 0.99  # match your PPO gamma

        self._pellet_dist_grid: list[list[int]] | None = None
        self._cached_potential: float = 0.0

        self.episode_event_counts: dict[str, int] = {}
        self.episode_reward_breakdown: dict[str, float] = {}

    def reset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset episode state and return (grid, features, valid_actions).

        A new random maze is generated each episode so the policy learns to
        *navigate* rather than memorise a single fixed layout.
        """
        self.step_count = 0
        self.visited_tiles = set()
        self.last_action = None
        self.last_cell = None
        self.prev_prev_cell = None
        self.episode_event_counts = {
            "pellet": 0,
            "super": 0,
            "osc": 0,
            "died": 0,
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
            "bfs": 0.0,
        }

        # Sample a fresh random maze size for this episode.
        maze_w = self.rng.randint(MAZE_WIDTH_MIN, MAZE_WIDTH_MAX)
        maze_h = self.rng.randint(MAZE_HEIGHT_MIN, MAZE_HEIGHT_MAX)
        # Episode seed: deterministic when outer seed is set, random otherwise.
        current_seed = self.rng.randint(1, 44444)

        maze_gen = LevelManager.build_maze(maze_w, maze_h, seed=current_seed)
        self.maze = maze_gen.maze
        self.movement = MovementSystem(self.maze)

        # Dynamic max steps based on maze size (w * h * MAZE_STEP_MULTIPLIER)
        maze_size = maze_w * maze_h
        if self.user_max_steps is None:
            self.max_steps = int(maze_size * MAZE_STEP_MULTIPLIER)
        else:
            self.max_steps = self.user_max_steps

        # Create Player and Ghosts
        self._create_entities()

        # Create Pellets
        self._create_pellets()
        if self.use_bfs_shaping:
            self._pellet_dist_grid = self._compute_pellet_distance_grid()
            self._cached_potential = self._potential_at(
                self.player.grid_y, self.player.grid_x
            )

        if self.player is not None:
            start_cell = (self.player.grid_y, self.player.grid_x)
            self.visited_tiles.add(start_cell)
            self.last_cell = start_cell

        return self._get_observation()

    def _compute_pellet_distance_grid(self) -> list[list[int]]:
        """Multi-source BFS from every remaining pellet cell at once — gives
        distance-to-nearest-pellet for every walkable cell in a single O(H*W)
        pass. Independent of player position, so it only needs recomputing
        when the pellet set actually changes (i.e. a pellet was just eaten).
        """
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
            # Maze cells contain directional wall bits; checking only whether
            # the neighboring cell equals 15 allows shaping to pass through
            # walls. Use the same legal connectivity as the game instead.
            for ny, nx in self.movement.get_neighbors(y, x):
                if dist[ny][nx] == -1:
                    dist[ny][nx] = d + 1
                    q.append((ny, nx))
        return dist

    def _potential_at(self, y: int, x: int) -> float:
        """Phi(s) = -distance to nearest remaining pellet. 0 pellets left
        (episode about to end) or an unreachable pocket both map to a finite
        value rather than blowing up."""
        if self.remaining_pellets <= 0 or self._pellet_dist_grid is None:
            return 0.0
        d = self._pellet_dist_grid[y][x]
        if d < 0:  # unreachable from here — shouldn't happen on a connected
            sentinel = len(self.maze) + len(self.maze[0])  # maze, but be safe
            return -float(sentinel)
        return -float(d)

    def step(
        self,
        action: int | torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        float,
        bool,
        dict[str, Any],
    ]:
        """Apply one action and advance physics until Pac-Man reaches a cell center.

        Reward fires exactly ONCE per cell crossing (not once per pixel tick).
        This removes the ~11× step-penalty spam that made exploration unprofitable.
        """
        if isinstance(action, torch.Tensor):
            action = int(action.item())

        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"Invalid action index {action}")

        _, _, valid_actions = self._get_observation()
        if not bool(valid_actions[0, action]):
            legal = torch.where(valid_actions[0])[0].tolist()
            if legal:
                action = self.rng.choice(legal)

        assert self.player is not None
        self.player.next_direction = DIRECTIONS[action]

        events = {
            "pellet_eaten": False,
            "super_pellet_eaten": False,
            "ghost_eaten": False,
            "pacman_died": False,
            "level_completed": False,
            "new_tile_visited": False,
            "oscillating": False,
        }

        start_cell = (self.player.grid_y, self.player.grid_x)
        potential_before = self._cached_potential if self.use_bfs_shaping else 0.0
        cell_changed = False

        for _ in range(MAX_PHYSICS_TICKS):
            prev_grid = (self.player.grid_x, self.player.grid_y)
            self._update_entities()

            # Detect grid-cell change mid-transit for event checking
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

        # If the player didn't cross to a new cell (wall hit / no valid move),
        # we still count it as a step so training doesn't freeze.
        current_pos = (self.player.grid_y, self.player.grid_x)

        # Track visited tiles
        if current_pos not in self.visited_tiles:
            events["new_tile_visited"] = True
            self.visited_tiles.add(current_pos)

        # Anti-oscillation: penalise A→B→A bouncing
        if cell_changed and self.prev_prev_cell is not None:
            if current_pos == self.prev_prev_cell:
                events["oscillating"] = True

        # Update oscillation history
        if cell_changed:
            self.prev_prev_cell = self.last_cell
            self.last_cell = current_pos

        bfs_shaping = 0.0
        if self.use_bfs_shaping:
            if events["pellet_eaten"] or events["super_pellet_eaten"]:
                self._pellet_dist_grid = self._compute_pellet_distance_grid()
            potential_after = self._potential_at(*current_pos)
            bfs_shaping = self.bfs_shaping_gamma * potential_after - potential_before
            self._cached_potential = potential_after

        reward, breakdown = self._calculate_reward(events, bfs_shaping)
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
        if events["level_completed"]:
            self.episode_event_counts["completed"] += 1

        self.step_count += 1

        terminated = bool(events["pacman_died"] or events["level_completed"])
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        pellets_eaten = self.total_pellets - self.remaining_pellets
        completion_pct = (
            (pellets_eaten / self.total_pellets * 100.0)
            if self.total_pellets > 0
            else 0.0
        )
        if truncated:
            self.episode_event_counts["truncated"] += 1

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
            "maze": (len(self.maze[0]), len(self.maze)),
        }
        if done:
            info["episode_event_counts"] = (
                dict(self.episode_event_counts) if done else None
            )
            info["episode_reward_breakdown"] = (
                dict(self.episode_reward_breakdown) if done else None
            )
        return self._get_observation(), reward, done, info

    def _create_entities(self) -> None:
        if self.maze is None:
            raise RuntimeError("Maze must be created first.")

        height = len(self.maze)
        width = len(self.maze[0])

        center_y = height // 2
        center_x = width // 2
        self.player = Player(center_y, center_x)
        if not self.player.is_valid_spawn(center_y, center_x, self.maze):
            if not self.player.find_player_spawn(None, self.maze):
                raise RuntimeError("Could not find valid player spawn.")

        self.player.powered_mode = None
        self.player.powered_timer = 0.0

        ghost_cells = [
            (0, 0),
            (0, width - 1),
            (height - 1, 0),
            (height - 1, width - 1),
        ]

        self.ghosts = []
        for (y, x), (name, color) in zip(ghost_cells, GHOST_SPECS):
            ghost = Ghost(y, x, color, name)
            ghost.reset()
            self.ghosts.append(ghost)

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
                if self.maze[y][x] == 15:  # Wall block / prison
                    pellets[y][x] = 0
                elif (x, y) == center or (
                    player_spawn is not None and (x, y) == player_spawn
                ):
                    pellets[y][x] = 0
                elif (x, y) in corners:
                    pellets[y][x] = 2  # Super power pellet
                    total += 1
                else:
                    pellets[y][x] = 1  # Regular pellet
                    total += 1

        self.pellets = pellets
        self.total_pellets = total
        self.remaining_pellets = total

    def _update_entities(self) -> None:
        """Advance player and ghosts by one simulation tick."""
        if self.movement is None or self.player is None or self.maze is None:
            return

        self.movement.update_entity(self.player)

        if self.player.powered_timer > 0:
            self.player.powered_timer -= 0.1
            if self.player.powered_timer <= 0:
                self.player.end_powered_mode()
                for ghost in self.ghosts:
                    ghost.is_edible = False

        if self.stage == 1:
            for ghost in self.ghosts:
                ghost.in_prison = True
                ghost.is_edible = False
            return

        for ghost in self.ghosts:
            if ghost.in_prison:
                self.movement.move_inside_prison(ghost)
            elif ghost.is_edible:
                self.movement.update_runaway_ghost(ghost, self.player)
            else:
                self.movement.update_bfs_ghost(ghost, self.player)

    def _check_events(self) -> dict[str, bool]:
        """Detect pellet consumption, ghost collisions, and win/loss states."""
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
            elif pellet_type == 2:
                self.pellets[py][px] = 0
                self.remaining_pellets -= 1
                events["super_pellet_eaten"] = True
                self.player.start_powered_mode(mode=pm.PUNCH, duration=30.0)
                for ghost in self.ghosts:
                    ghost.is_edible = True

        if self.remaining_pellets <= 0:
            events["level_completed"] = True

        if self.stage == 1:
            return events

        for ghost in self.ghosts:
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
                else:
                    # TODO: make true latter
                    events["pacman_died"] = False
                    break

        return events

    def _calculate_reward(
        self, events: dict[str, bool], bfs_shaping: float = 0.0
    ) -> tuple[float, dict[str, float]]:
        breakdown = {
            "step": -0.2,
            "oscillation": 0.0,
            "pellet": 0.0,
            "super_pellet": 0.0,
            "ghost": 0.0,
            "complete": 0.0,
            "death": 0.0,
            "bfs": 0.0,
        }
        if events.get("oscillating", False) and not (
            events["pellet_eaten"] or events["super_pellet_eaten"]
        ):
            # Reversing out of a pellet-bearing dead end is required for a
            # complete maze clear. Penalize only empty A->B->A cycles, and keep
            # the signal small enough that it does not forbid backtracking.
            breakdown["oscillation"] = -0.1
        if events["pellet_eaten"]:
            breakdown["pellet"] = 3.0
        if events["super_pellet_eaten"]:
            breakdown["super_pellet"] = 5.0
        if events["ghost_eaten"]:
            breakdown["ghost"] = 30.0
        if events["level_completed"]:
            remaining_steps = max(0, self.max_steps - self.step_count)
            breakdown["complete"] = 100 + float(remaining_steps)
        if events["pacman_died"]:
            breakdown["death"] = -30.0
        breakdown["bfs"] = self.bfs_shaping_coef * bfs_shaping

        return sum(breakdown.values()), breakdown

    def _get_observation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build grid, extra_features, and valid_actions tensors for Pac-Man using ObservationFormatter."""
        if (
            self.maze is None
            or self.player is None
            or self.pellets is None
            or self.movement is None
        ):
            raise RuntimeError("Environment has not been initialized.")

        ghost_states = [
            {
                "grid_x": ghost.grid_x,
                "grid_y": ghost.grid_y,
                "is_edible": ghost.is_edible,
                "direction": ghost.direction,
            }
            for ghost in self.ghosts
        ]

        grid, extra_features, valid_player_actions, _ = (
            ObservationFormatter.format_observation(
                maze=self.maze,
                pellets=self.pellets,
                player_pos=(self.player.grid_x, self.player.grid_y),
                player_direction=self.player.direction,
                ghost_states=ghost_states,
                movement=self.movement,
                device=self.device,
            )
        )

        return grid, extra_features, valid_player_actions
