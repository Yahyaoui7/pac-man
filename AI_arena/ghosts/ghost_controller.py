"""Run the trained ghost CNN against the live game state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import torch

# from AI_arena.data.constants import CNN_HEIGHT, CNN_WIDTH
from AI_arena.models.cnn_ghost import GhostCNN
from src.logic.config import CELL_SIZE, EAST, NORTH, SOUTH, WEST

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "ghost_ai.pt"
QUANTIZED_MODEL_PATH = Path(__file__).parent.parent / "models" / "ghost_ai_quantized.pt"
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
        if Path(model_path).exists():
            weights = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(weights)
        self.model.eval()

    def predict(
        self,
        maze: list[list[int]],
        pellets: list[list[int]],
        player: Any,
        ghosts: list[Any],
        movement: Any,
    ) -> dict[str, str | None]:
        """Predict the move for each ghost given current game state."""
        ghost_states = [
            {
                "grid_x": ghost.grid_x,
                "grid_y": ghost.grid_y,
                "is_edible": ghost.is_edible,
                "direction": ghost.direction,
            }
            for ghost in ghosts
        ]

        from AI_arena.data.formatter import ObservationFormatter

        grid, extra_features, _, valid_ghost_actions = (
            ObservationFormatter.format_observation(
                maze=maze,
                pellets=pellets,
                player_pos=(player.grid_x, player.grid_y),
                player_direction=player.direction,
                ghost_states=ghost_states,
                movement=movement,
                device=self.device,
            )
        )

        predictions: dict[str, str | None] = {}
        with torch.no_grad():
            logits = self.model(grid, extra_features)  # (1, 4, 4)
            for idx, ghost in enumerate(ghosts):
                valid_mask = valid_ghost_actions[idx]  # (4,)
                ghost_logits = logits[0, idx].masked_fill(~valid_mask, -1e9)
                chosen_idx = int(torch.argmax(ghost_logits).item())
                if bool(valid_mask[chosen_idx]):
                    predictions[ghost.name] = DIRECTIONS[chosen_idx]
                else:
                    predictions[ghost.name] = None

        return predictions
