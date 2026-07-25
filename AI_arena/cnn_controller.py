"""Run the trained ghost CNN against the live game state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch

from AI_arena.cnn_dataset import CNN_HEIGHT, CNN_WIDTH
from AI_arena.cnn_model import GhostCNN
from src.logic.config import CELL_SIZE, EAST, NORTH, SOUTH, WEST

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "ghost_ai.pt"
QUANTIZED_MODEL_PATH = Path(__file__).parent / "models" / "ghost_ai_quantized.pt"
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
GHOST_NAMES = ("Blinky", "Pinky", "Inky", "Clyde")


class GhostDiagnostic(TypedDict):
    position: tuple[int, int]
    chosen: str | None
    confidence: float
    legal: list[str]
    probabilities: dict[str, float]


class CNNGhostController:
    """Build live observations and predict one legal move per ghost."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.last_diagnostics: dict[str, GhostDiagnostic] = {}

        # Cached observation state (rebuilt on level start via init_observation)
        self._cached_grid: torch.Tensor | None = None
        self._cached_maze_id: int | None = None

        quantized = Path(QUANTIZED_MODEL_PATH)
        if quantized.exists():
            self.model = torch.jit.load(
                str(quantized), map_location=self.device
            )
            self.model.eval()
            return

        self.model = GhostCNN().to(self.device)
        weights = torch.load(
            Path(model_path),
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(weights)
        self.model.eval()

    @staticmethod
    def supports_maze(maze: list[list[int]]) -> bool:
        height = len(maze)
        width = len(maze[0]) if height else 0
        return 0 < height <= CNN_HEIGHT and 0 < width <= CNN_WIDTH

    @staticmethod
    def _entity_position(
        entity: Any,
        width: int,
        height: int,
    ) -> tuple[int, int]:
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

    def init_observation(self, maze: list[list[int]]) -> None:
        """Pre-compute static grid channels that never change during a level.

        Builds channels 0-3 (wall bits) and channel 11 (non-prison mask)
        once, so predict() only needs to fill dynamic channels each frame.
        """
        height = len(maze)
        width = len(maze[0])
        grid = torch.zeros(
            (1, 12, CNN_HEIGHT, CNN_WIDTH),
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
        self._cached_grid = grid
        self._cached_maze_id = id(maze)

    def predict(
        self,
        maze: list[list[int]],
        pellets: list[list[int]],
        player: Any,
        ghosts: list[Any],
        movement: Any,
    ) -> dict[str, str | None]:
        """Return masked CNN actions in the model's fixed ghost order."""

        if not self.supports_maze(maze):
            return {name: None for name in GHOST_NAMES}

        ghosts_by_name = {ghost.name: ghost for ghost in ghosts}
        if set(ghosts_by_name) != set(GHOST_NAMES):
            return {name: None for name in GHOST_NAMES}

        # Ensure static channels are cached
        maze_id = id(maze)
        if self._cached_grid is None or self._cached_maze_id != maze_id:
            self.init_observation(maze)

        height = len(maze)
        width = len(maze[0])
        grid = self._cached_grid

        # Update dynamic channels in-place (avoids clone overhead).
        # Pellets list may be smaller than CNN_HEIGHT x CNN_WIDTH when the
        # maze doesn't fill the full observation grid, so we build the
        # tensor at grid size and only overwrite the active region.
        pellet_t = torch.zeros(
            (1, CNN_HEIGHT, CNN_WIDTH),
            dtype=torch.float32,
            device=self.device,
        )
        pellet_t[:, :height, :width] = torch.tensor(
            [pellets], dtype=torch.float32, device=self.device,
        )
        grid[0, 4] = (pellet_t == 1).float()
        grid[0, 5] = (pellet_t == 2).float()

        # Zero dynamic entity channels before filling
        grid[0, 6].zero_()
        grid[0, 7:11].zero_()

        player_x, player_y = self._entity_position(player, width, height)
        grid[0, 6, player_y, player_x] = 1

        valid_actions = torch.zeros(
            (1, 4, 4),
            dtype=torch.bool,
            device=self.device,
        )
        for ghost_index, name in enumerate(GHOST_NAMES):
            ghost = ghosts_by_name[name]
            ghost_x, ghost_y = self._entity_position(ghost, width, height)
            grid[0, 7 + ghost_index, ghost_y, ghost_x] = 1
            for action_index, direction in enumerate(DIRECTIONS):
                valid_actions[0, ghost_index, action_index] = (
                    movement.can_move(ghost_y, ghost_x, direction)
                )

        # Single BFS from player — reuse distances for all 4 ghosts
        bfs_dist = movement.bfs_distances((player_y, player_x))

        player_direction = [
            float(player.direction == direction) for direction in DIRECTIONS
        ]
        player_powered = float(any(ghost.is_edible for ghost in ghosts))
        features = [
            *player_direction,
            player_powered,
            *(
                float(ghosts_by_name[name].is_edible)
                for name in GHOST_NAMES
            ),
        ]
        max_dimension = max(width, height, 1)
        for name in GHOST_NAMES:
            ghost_x, ghost_y = self._entity_position(
                ghosts_by_name[name],
                width,
                height,
            )
            dist = bfs_dist[ghost_y * width + ghost_x]
            features.extend(
                [
                    (player_x - ghost_x) / max_dimension,
                    (player_y - ghost_y) / max_dimension,
                    (dist + 1) / max_dimension,
                ]
            )
        for name in GHOST_NAMES:
            ghost = ghosts_by_name[name]
            features.extend(
                [
                    float(ghost.direction == direction)
                    for direction in DIRECTIONS
                ]
            )
        extra_features = torch.tensor(
            [features],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            logits = self.model(grid, extra_features)
            masked_logits = logits.masked_fill(
                ~valid_actions,
                float("-inf"),
            )
            probabilities = torch.softmax(masked_logits, dim=-1)
            action_indices = masked_logits.argmax(dim=-1)[0].tolist()

        predictions: dict[str, str | None] = {}
        for ghost_index, name in enumerate(GHOST_NAMES):
            if valid_actions[0, ghost_index].any().item():
                predictions[name] = DIRECTIONS[action_indices[ghost_index]]
            else:
                predictions[name] = None
        return predictions
