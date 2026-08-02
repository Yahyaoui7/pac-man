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


class PacmanPlayerEnv:
    """Headless environment in which RL policy controls Pac-Man against 4 BFS ghosts."""

    def __init__(
        self,
        seed: int | None = None,
        maze_width: int = 20,
        maze_height: int = 25,
        max_steps: int = 1500,
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
        self.max_steps = max_steps
        self.maze_width = maze_width
        self.maze_height = maze_height
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

    def reset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset episode state and return (grid, features, valid_actions)."""
        self.step_count = 0
        self.visited_tiles = set()
        self.last_action = None

        current_seed = (
            self.seed
            if self.seed is not None
            else self.rng.randint(0, 1_000_000)
        )
        maze_gen = LevelManager.build_maze(
            self.maze_width,
            self.maze_height,
            seed=current_seed,
        )
        self.maze = maze_gen.maze
        self.movement = MovementSystem(self.maze)

        # Create Player and Ghosts
        self._create_entities()

        # Create Pellets
        self._create_pellets()

        if self.player is not None:
            self.visited_tiles.add((self.player.grid_y, self.player.grid_x))

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
        """Apply one action to Pac-Man and advance environment until next step."""
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
        }

        sub_ticks = 2
        for _ in range(sub_ticks):
            self._update_entities()
            tick_events = self._check_events()

            for key, val in tick_events.items():
                if val:
                    events[key] = True

            if events["pacman_died"] or events["level_completed"]:
                break

        pos = (self.player.grid_y, self.player.grid_x)
        if pos not in self.visited_tiles:
            events["new_tile_visited"] = True
            self.visited_tiles.add(pos)

        reward = self._calculate_reward(events)
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
        }

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

        total = 0
        for y in range(height):
            for x in range(width):
                if self.maze[y][x] == 15:  # Wall block / prison
                    pellets[y][x] = 0
                elif (x, y) == center:
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

        if (
            self.player is None
            or self.pellets is None
            or self.maze is None
        ):
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
            if (ghost.grid_y == py and ghost.grid_x == px) or dist_sq <= (CELL_SIZE * 0.6) ** 2:
                if ghost.is_edible:
                    events["ghost_eaten"] = True
                    ghost.is_edible = False
                    ghost.in_prison = True
                    ghost.reset()
                else:
                    events["pacman_died"] = True
                    break

        return events

    def _calculate_reward(self, events: dict[str, bool]) -> float:
        """Calculate step reward focused on pure navigation and pellet collection."""
        reward = -0.1  # Mild step penalty to allow long-distance corridor traversal
        if events.get("new_tile_visited", False):
            reward += 2.0  # Strong exploration bonus to drive Pac-Man into unexplored corridors
        if events["pellet_eaten"]:
            reward += 5.0  # High pellet reward (+6.9 net on new tile)
        if events["super_pellet_eaten"]:
            reward += 10.0
        if events["level_completed"]:
            reward += 100.0
        if events["pacman_died"]:
            reward -= 20.0
        return reward

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
