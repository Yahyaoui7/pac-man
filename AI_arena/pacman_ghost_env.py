"""Headless Pac-Man environment template for ghost reinforcement learning.

Implement the TODO sections in the order documented in AI_arena/RL_README.md.
This file intentionally contains the environment structure first; it does not
yet duplicate the complete Pygame game implementation.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from AI_arena.cnn_controller import DIRECTIONS
from AI_arena.cnn_dataset import ACTION_COUNT, GHOST_COUNT
from src.logic.movement import MovementSystem
from mazegenerator import MazeGenerator

GHOST_NAMES = ["Blinky", "Pinky", "Inky", "Clyde"]
MIN_MAZE_WIDTH = 10
MAX_MAZE_WIDTH = 25
MIN_MAZE_HEIGHT = 10
MAX_MAZE_HEIGHT = 50


class PacmanGhostEnv:
    """Headless environment in which one policy controls four ghosts."""

    def __init__(
        self,
        seed: int | None = None,
        maze_width: int = 20,
        maze_height: int = 25,
        max_steps: int = 2000,
    ) -> None:
        # Store episode configuration.
        self.max_steps = max_steps
        self.maze_width = (maze_width,)
        self.maze_hiegth = (maze_height,)
        self.step_count = 0
        self.rng = random.Random(seed)

        # These objects are created/reset by reset().
        self.maze: list[list[int]] | None = None
        self.movement: MovementSystem | None = None
        self.player: Any | None = None
        self.ghosts: list[Any] = []
        self.pellets: list[list[int]] | None = None

    def reset(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Start an episode and return ``(grid, features, valid_actions)``."""

        self.step_count = 0

        # TODO: Load or generate a maze without opening a Pygame window.
        self.maze = self._create_maze()
        self.movement = MovementSystem(self.maze)

        # TODO: Create/reset one player and exactly four ghosts.
        self.player, self.ghosts = self._create_entities()

        # TODO: Create/reset pellet positions and any score/timer state.
        self.pellets = self._create_pellets()

        return self._get_observation()

    def step(
        self,
        actions: list[int] | torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        float,
        bool,
        dict[str, Any],
    ]:
        """Apply one direction action to each ghost and advance the game."""

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().tolist()
        if len(actions) != GHOST_COUNT:
            raise ValueError("Expected one action for each of four ghosts")

        # Get the current legal-action mask before applying actions.
        _, _, valid_actions = self._get_observation()
        for ghost_index, action_index in enumerate(actions):
            self._apply_ghost_action(
                ghost_index,
                int(action_index),
                valid_actions,
            )

        # TODO: Move Pac-Man with a scripted/random policy. Keyboard input is
        # not available in a headless training environment.
        self._move_player()

        # TODO: Advance entities using MovementSystem, then check collisions.
        self._update_entities()
        events = self._check_events()
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
        }
        return self._get_observation(), reward, done, info

    def _get_observation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the CNN grid, extra features, and legal-action mask."""

        # TODO: Adapt the observation construction from
        # CNNGhostController.predict(). Return shapes:
        #   grid           = [1, 12, 50, 25]
        #   extra_features = [1, 37]
        #   valid_actions  = [1, 4, 4]
        raise NotImplementedError

    def _apply_ghost_action(
        self,
        ghost_index: int,
        action_index: int,
        valid_actions: torch.Tensor,
    ) -> None:
        """Apply one legal action, using a fallback if an action is blocked."""

        if not 0 <= ghost_index < GHOST_COUNT:
            raise ValueError("Invalid ghost index")
        if not 0 <= action_index < ACTION_COUNT:
            raise ValueError("Invalid action index")

        if not bool(valid_actions[0, ghost_index, action_index]):
            legal = torch.where(
                valid_actions[0, ghost_index],
            )[0].tolist()
            if not legal:
                return
            action_index = self.rng.choice(legal)

        # Action order is UP, DOWN, LEFT, RIGHT.
        self.ghosts[ghost_index].next_direction = DIRECTIONS[action_index]

    def _move_player(self) -> None:
        """Move Pac-Man using a deterministic or random non-keyboard policy."""

        # TODO: Keep the current direction if legal; otherwise choose a legal
        # random direction with self.rng.
        pass

    def _update_entities(self) -> None:
        """Advance the player and ghosts by one simulation step."""

        # TODO: Call MovementSystem update methods. Do not call pygame, draw,
        # sound, or keyboard code here.
        pass

    def _check_events(self) -> dict[str, bool]:
        """Detect deaths, eaten ghosts, respawns, and level completion."""

        # TODO: Adapt collision logic from PlayingState.check_collision().
        return {
            "pacman_died": False,
            "ghost_was_eaten": False,
            "level_completed": False,
        }

    def _calculate_reward(self, events: dict[str, bool]) -> float:
        """Convert game events into a shared reward for all ghosts."""

        reward = -0.001  # Encourage catching Pac-Man quickly.
        if events["pacman_died"]:
            reward += 10.0
        if events["ghost_was_eaten"]:
            reward -= 2.0
        if events["level_completed"]:
            reward -= 10.0
        return reward

    def _create_maze(self) -> list[list[int]]:
        from src.logic.level_manager import LevelManager

        seed = (
            self.seed
            if self.seed is not None
            else self.rng.randint(0, 1_000_000)
        )
        maze_generator = LevelManager.build_maze(
            self.maze_width,
            self.maze_height,
            seed=seed,
        )

        return maze_generator.maze

    def _create_entities(self) -> tuple[Any, list[Any]]:
        """Create Pac-Man near the centre and ghosts near the four corners."""
        if self.maze is None or not self.maze or not self.maze[0]:
            raise RuntimeError("A maze must be created before entities")

        from src.graphics.entitys.ghost import Ghost
        from src.graphics.entitys.player import Player

        height = len(self.maze)
        width = len(self.maze[0])
        walkable = [
            (y, x)
            for y in range(height)
            for x in range(width)
            if self.maze[y][x] != 15
        ]
        if len(walkable) < GHOST_COUNT + 1:
            raise RuntimeError("Maze does not contain enough walkable cells")

        # Start Pac-Man at the centre when possible.  If the centre is a wall,
        # Player.find_player_spawn() searches outward for the nearest legal
        # cell, just as the graphical game does.
        center_y, center_x = height // 2, width // 2
        player = Player(center_y, center_x)
        if not player.is_valid_spawn(center_y, center_x, self.maze):
            if not player.find_player_spawn(None, self.maze):
                raise RuntimeError("Could not find a valid Pac-Man spawn cell")
        # Player() already initializes normal mode.  Do not call
        # reset_location() here: that is a death/respawn reset and would clear
        # any powered-mode state.  Initial creation only needs no powered mode.
        player.powered_mode = None
        player_cell = (player.grid_y, player.grid_x)

        # Pick the nearest available walkable cell to each corner.  Generated
        # mazes commonly have walls in the literal corners, so using the
        # nearest legal cell gives ghosts a stable corner spawn without ever
        # placing one on a wall or on another entity.
        chosen: list[tuple[int, int]] = [player_cell]
        corner_order = [
            (0, 0),
            (0, width - 1),
            (height - 1, 0),
            (height - 1, width - 1),
        ]
        for corner in corner_order:
            candidates = sorted(
                (cell for cell in walkable if cell not in chosen),
                key=lambda cell: abs(cell[0] - corner[0])
                + abs(cell[1] - corner[1]),
            )
            if not candidates:
                raise RuntimeError("Maze does not contain enough distinct spawn cells")
            chosen.append(candidates[0])

        ghost_specs = [
            ("Blinky", (255, 0, 0)),
            ("Pinky", (255, 182, 193)),
            ("Inky", (0, 255, 255)),
            ("Clyde", (255, 165, 0)),
        ]
        ghosts = [
            Ghost(y, x, color, name)
            for (y, x), (name, color) in zip(chosen[1:], ghost_specs)
        ]
        # Use each ghost's normal respawn/reset path for initial state too.
        for ghost in ghosts:
            ghost.reset()
        return player, ghosts

    def _create_pellets(self) -> list[list[int]]:

        height = len(self.maze)
        width = len(self.maze[0])
        pellets = [[0] * width for _ in range(height)]
        center_x = width // 2
        center_y = height // 2
        for x in range(height):
            for y in range(width):
                if self.maze[x][y] == 15:
                    pellets[x][y] = 0
                elif x == center_x and y == center_y:
                    pellets[x][y] = 0
                else:
                    pellets[x][y] = 1
        return pellets

    def random_legal_actions(self) -> list[int]:
        """Return one random legal action per ghost for environment testing."""

        _, _, valid_actions = self._get_observation()
        actions: list[int] = []
        for ghost_index in range(GHOST_COUNT):
            legal = torch.where(
                valid_actions[0, ghost_index],
            )[0].tolist()
            # UP is only a defensive fallback; every real cell should have an
            # available action.
            actions.append(self.rng.choice(legal) if legal else 0)
        return actions


def smoke_test() -> None:
    """Run after implementing TODOs to verify random episodes terminate."""

    env = PacmanGhostEnv(seed=42)
    env.reset()
    for _ in range(env.max_steps):
        _, _, done, info = env.step(env.random_legal_actions())
        if done:
            print("Episode finished:", info)
            return
    print("Episode reached the maximum step limit")


if __name__ == "__main__":
    smoke_test()
