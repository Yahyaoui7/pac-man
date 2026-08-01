"""Inference controller for trained Pac-Man player RL model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from AI_arena.cnn_dataset import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    CNN_HEIGHT,
    CNN_WIDTH,
    EXTRA_FEATURE_COUNT,
)
from AI_arena.player_cnn_model import PlayerActorCritic
from src.logic.config import CELL_SIZE, EAST, NORTH, SOUTH, WEST

DEFAULT_PLAYER_MODEL_PATH = Path(__file__).parent / "models" / "player_rl.pt"
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class CNNPlayerController:
    """Build live observations and predict Pac-Man's best move."""

    def __init__(self, model_path: str | Path = DEFAULT_PLAYER_MODEL_PATH) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PlayerActorCritic().to(self.device)

        path = Path(model_path)
        if path.exists():
            weights = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(weights)
            print(f"Loaded player RL checkpoint from {path}")
        else:
            print(f"Warning: Player RL checkpoint {path} not found. Using untrained weights.")

        self.model.eval()

    @staticmethod
    def _entity_position(entity: Any, width: int, height: int) -> tuple[int, int]:
        x = (
            int(entity.x // CELL_SIZE)
            if hasattr(entity, "x")
            else entity.grid_x
        )
        y = (
            int(entity.y // CELL_SIZE)
            if hasattr(entity, "y")
            else entity.grid_y
        )
        return max(0, min(width - 1, x)), max(0, min(height - 1, y))

    def predict(
        self,
        maze: list[list[int]],
        pellets: list[list[int]],
        player: Any,
        ghosts: list[Any],
        movement: Any,
    ) -> str | None:
        """Return the predicted legal move for Pac-Man."""
        height = len(maze)
        width = len(maze[0])

        grid = torch.zeros(
            (1, CNN_CHANNEL_COUNT, CNN_HEIGHT, CNN_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )

        for y, row in enumerate(maze):
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
            [pellets], dtype=torch.float32, device=self.device
        )
        grid[0, 4] = (pellet_t == 1).float()
        grid[0, 5] = (pellet_t == 2).float()

        player_x, player_y = self._entity_position(player, width, height)
        grid[0, 6, player_y, player_x] = 1.0

        for ghost_idx, ghost in enumerate(ghosts[:4]):
            gx, gy = self._entity_position(ghost, width, height)
            grid[0, 7 + ghost_idx, gy, gx] = 1.0

        player_direction = [
            float(player.direction == direction) for direction in DIRECTIONS
        ]
        player_powered = float(any(getattr(ghost, "is_edible", False) for ghost in ghosts))

        features = [
            *player_direction,
            player_powered,
            *(float(getattr(ghost, "is_edible", False)) for ghost in ghosts[:4]),
        ]

        bfs_dist = movement.bfs_distances((player_y, player_x))
        max_dim = max(width, height, 1)

        for ghost in ghosts[:4]:
            gx, gy = self._entity_position(ghost, width, height)
            idx = gy * width + gx
            dist = bfs_dist[idx] if 0 <= idx < len(bfs_dist) else -1
            features.extend(
                [
                    (player_x - gx) / max_dim,
                    (player_y - gy) / max_dim,
                    (dist + 1) / max_dim,
                ]
            )

        for ghost in ghosts[:4]:
            ghost_dir = getattr(ghost, "direction", None)
            features.extend(
                [float(ghost_dir == direction) for direction in DIRECTIONS]
            )

        extra_features = torch.tensor(
            [features], dtype=torch.float32, device=self.device
        )

        valid_actions = torch.zeros(
            (1, ACTION_COUNT), dtype=torch.bool, device=self.device
        )
        for action_idx, direction in enumerate(DIRECTIONS):
            valid_actions[0, action_idx] = movement.can_move(
                player_y, player_x, direction
            )

        with torch.inference_mode():
            logits, _ = self.model(grid, extra_features)
            masked_logits = logits.masked_fill(~valid_actions, float("-inf"))
            if not valid_actions.any():
                return None
            best_action_idx = masked_logits.argmax(dim=-1).item()
            return DIRECTIONS[best_action_idx]


def main() -> None:
    controller = CNNPlayerController()
    print("CNNPlayerController initialized successfully.")


if __name__ == "__main__":
    main()
