"""Headless Pac-Man environment for player reinforcement learning against BFS ghosts."""

from __future__ import annotations

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import random
from typing import Any

import pygame
pygame.init()
if not pygame.display.get_surface():
    pygame.display.set_mode((1, 1))

from src.graphics.entitys.graphic_lib import SpriteLibrary
from src.logic.config import CELL_SIZE
SpriteLibrary.instance().load(CELL_SIZE)
SpriteLibrary.instance().load_ghosts(CELL_SIZE)

torch_import_error = None
try:
    import torch
except ImportError as err:
    torch_import_error = err

from AI_arena.cnn_controller import DIRECTIONS
from AI_arena.cnn_dataset import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    CNN_HEIGHT,
    CNN_WIDTH,
    EXTRA_FEATURE_COUNT,
    GHOST_COUNT,
)
from src.graphics.entitys.ghost import Ghost
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
    ) -> None:
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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def reset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset episode state and return (grid, features, valid_actions)."""
        self.step_count = 0

        # Generate maze
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
        """Apply one action to Pac-Man and advance the environment until next decision step."""
        if isinstance(action, torch.Tensor):
            action = int(action.item())

        if not 0 <= action < ACTION_COUNT:
            raise ValueError(f"Invalid action index {action}")

        # Validate action against legal moves mask
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
        }

        # Advance entities in sub-ticks until Pac-Man reaches a new cell center or an event occurs
        sub_ticks = 2
        for _ in range(sub_ticks):
            self._update_entities()
            tick_events = self._check_events()

            for key, val in tick_events.items():
                if val:
                    events[key] = True

            if events["pacman_died"] or events["level_completed"]:
                break

        reward = self._calculate_reward(events)
        self.step_count += 1

        terminated = bool(events["pacman_died"] or events["level_completed"])
        truncated = self.step_count >= self.max_steps
        done = terminated or truncated

        info = {
            "step": self.step_count,
            "terminated": terminated,
            "truncated": truncated,
            "events": events,
            "remaining_pellets": self.remaining_pellets,
        }

        return self._get_observation(), reward, done, info

    def _create_entities(self) -> None:
        if self.maze is None:
            raise RuntimeError("Maze must be created first.")

        height = len(self.maze)
        width = len(self.maze[0])

        # Create Player near center
        center_y = height // 2
        center_x = width // 2
        self.player = Player(center_y, center_x)
        if not self.player.is_valid_spawn(center_y, center_x, self.maze):
            if not self.player.find_player_spawn(None, self.maze):
                raise RuntimeError("Could not find valid player spawn.")

        self.player.powered_mode = None
        self.player.powered_timer = 0.0

        # Create 4 Ghosts at four corners
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

        # Advance Pac-Man
        self.movement.update_entity(self.player)

        # Update power timer if Pac-Man is powered
        if self.player.powered_timer > 0:
            self.player.powered_timer -= 0.1
            if self.player.powered_timer <= 0:
                self.player.end_powered_mode()
                for ghost in self.ghosts:
                    ghost.is_edible = False

        # Advance Ghosts with BFS AI
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
                self.player.start_powered_mode(mode=1, duration=30.0)
                for ghost in self.ghosts:
                    ghost.is_edible = True

        if self.remaining_pellets <= 0:
            events["level_completed"] = True

        # Collision detection with ghosts
        for ghost in self.ghosts:
            if ghost.in_prison:
                continue

            # Grid cell or close pixel proximity
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
        """Calculate step reward for Pac-Man agent."""
        reward = -0.01  # Small step penalty to encourage speed
        if events["pellet_eaten"]:
            reward += 1.0
        if events["super_pellet_eaten"]:
            reward += 5.0
        if events["ghost_eaten"]:
            reward += 10.0
        if events["level_completed"]:
            reward += 20.0
        if events["pacman_died"]:
            reward -= 10.0
        return reward

    def _get_observation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build grid, extra_features, and valid_actions tensors for Pac-Man."""
        if (
            self.maze is None
            or self.player is None
            or self.pellets is None
            or self.movement is None
        ):
            raise RuntimeError("Environment has not been initialized.")

        height = len(self.maze)
        width = len(self.maze[0])

        grid = torch.zeros(
            (1, CNN_CHANNEL_COUNT, CNN_HEIGHT, CNN_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )

        for y, row in enumerate(self.maze):
            for x, cell in enumerate(row):
                grid[0, 0, y, x] = bool(cell & NORTH)
                grid[0, 1, y, x] = bool(cell & SOUTH)
                grid[0, 2, y, x] = bool(cell & WEST)
                grid[0, 3, y, x] = bool(cell & EAST)
                grid[0, 11, y, x] = cell != 15

        pellet_t = torch.zeros(
            (1, CNN_HEIGHT, CNN_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        pellet_t[:, :height, :width] = torch.tensor(
            [self.pellets],
            dtype=torch.float32,
            device=self.device,
        )

        grid[0, 4] = (pellet_t == 1).float()
        grid[0, 5] = (pellet_t == 2).float()
        grid[0, 6, self.player.grid_y, self.player.grid_x] = 1.0

        for ghost_index, ghost in enumerate(self.ghosts):
            gy = max(0, min(height - 1, ghost.grid_y))
            gx = max(0, min(width - 1, ghost.grid_x))
            grid[0, 7 + ghost_index, gy, gx] = 1.0

        # Build extra features vector [1, 37]
        player_direction = [
            float(self.player.direction == direction) for direction in DIRECTIONS
        ]
        player_powered = float(any(ghost.is_edible for ghost in self.ghosts))

        features = [
            *player_direction,
            player_powered,
            *(float(ghost.is_edible) for ghost in self.ghosts),
        ]

        bfs_dist = self.movement.bfs_distances((self.player.grid_y, self.player.grid_x))
        max_dim = max(width, height, 1)

        for ghost in self.ghosts:
            gx, gy = ghost.grid_x, ghost.grid_y
            idx = gy * width + gx
            dist = bfs_dist[idx] if 0 <= idx < len(bfs_dist) else -1
            features.extend(
                [
                    (self.player.grid_x - gx) / max_dim,
                    (self.player.grid_y - gy) / max_dim,
                    (dist + 1) / max_dim,
                ]
            )

        for ghost in self.ghosts:
            features.extend(
                [float(ghost.direction == direction) for direction in DIRECTIONS]
            )

        extra_features = torch.tensor(
            [features],
            dtype=torch.float32,
            device=self.device,
        )

        # Build legal actions mask for Pac-Man [1, 4]
        valid_actions = torch.zeros(
            (1, ACTION_COUNT),
            dtype=torch.bool,
            device=self.device,
        )
        for action_index, direction in enumerate(DIRECTIONS):
            valid_actions[0, action_index] = self.movement.can_move(
                self.player.grid_y,
                self.player.grid_x,
                direction,
            )

        return grid, extra_features, valid_actions


def random_smoke_test() -> None:
    """Run environment smoke test using random valid Pac-Man moves against BFS ghosts."""
    print("Running PacmanPlayerEnv smoke test...")
    env = PacmanPlayerEnv(seed=42)
    obs = env.reset()
    grid, features, valid_actions = obs
    print(f"Reset successfully. Shapes:")
    print(f"  grid: {tuple(grid.shape)}")
    print(f"  features: {tuple(features.shape)}")
    print(f"  valid_actions: {tuple(valid_actions.shape)}")

    total_reward = 0.0
    for step in range(500):
        legal_actions = torch.where(valid_actions[0])[0].tolist()
        action = random.choice(legal_actions) if legal_actions else 0

        (grid, features, valid_actions), reward, done, info = env.step(action)
        total_reward += reward

        if done:
            print(f"Smoke test episode ended at step {step + 1}. Total reward: {total_reward:.2f}, info: {info}")
            return

    print(f"Smoke test reached 500 steps. Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    random_smoke_test()
