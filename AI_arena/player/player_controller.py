"""Inference controller for CNN-based Pac-Man player model."""

from __future__ import annotations

from pathlib import Path

import torch

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    CNN_HEIGHT,
    CNN_WIDTH,
)
from AI_arena.models.cnn_player import PlayerActorCritic
from src.graphics.entitys.ghost import Ghost
from src.graphics.entitys.player import Player
from src.logic.config import CELL_SIZE, EAST, NORTH, SOUTH, WEST

from AI_arena.data.formatter import ObservationFormatter

DEFAULT_STAGE1_PATH = Path(__file__).parent.parent / "models" / "player_rl_stage1.pt"
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class CNNPlayerController:
    """Build live observations and predict Pac-Man's best move."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PlayerActorCritic().to(self.device)

        if model_path is None:
            if DEFAULT_STAGE1_PATH.exists():
                path = DEFAULT_STAGE1_PATH

        else:
            path = Path(model_path)

        if path.exists():
            weights = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(weights)
            print(f"Loaded player RL checkpoint from {path}")
        else:
            print(
                f"Warning: Player RL checkpoint {path} not found. Using untrained weights."
            )

        self.model.eval()
        self.last_diagnostics: dict[str, Any] = {}

    def get_action(
        self,
        maze: list[list[int]],
        pellets: list[list[int]],
        player: Player,
        ghosts: list[Ghost],
        movement_system: Any,
        sample: bool = True,
    ) -> str:
        """Construct state tensors and select action (sampling from distribution or greedy)."""
        grid, extra_features, valid_actions = self._build_observation(
            maze, pellets, player, ghosts, movement_system
        )

        with torch.no_grad():
            logits, value = self.model(grid, extra_features)
            masked_logits = logits.masked_fill(~valid_actions, -1e9)
            probs = torch.softmax(masked_logits, dim=-1)[0]
            if sample:
                action_index = int(torch.multinomial(probs, 1).item())
            else:
                action_index = int(torch.argmax(masked_logits, dim=-1).item())

        chosen_action = DIRECTIONS[action_index]
        self.last_diagnostics = {
            "chosen_action": chosen_action,
            "estimated_value": round(float(value.item()), 4),
            "probabilities": {
                d: round(float(probs[i].item()), 4) for i, d in enumerate(DIRECTIONS)
            },
            "logits": {
                d: round(float(logits[0, i].item()), 4)
                for i, d in enumerate(DIRECTIONS)
            },
            "valid_actions": {
                d: bool(valid_actions[0, i].item()) for i, d in enumerate(DIRECTIONS)
            },
        }

        return chosen_action

    def _build_observation(
        self,
        maze: list[list[int]],
        pellets: list[list[int]],
        player: Player,
        ghosts: list[Ghost],
        movement_system: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ghost_states = [
            {
                "grid_x": ghost.grid_x,
                "grid_y": ghost.grid_y,
                "is_edible": ghost.is_edible,
                "direction": ghost.direction,
            }
            for ghost in ghosts
        ]

        grid, extra_features, valid_player_actions, _ = (
            ObservationFormatter.format_observation(
                maze=maze,
                pellets=pellets,
                player_pos=(player.grid_x, player.grid_y),
                player_direction=player.direction,
                ghost_states=ghost_states,
                movement=movement_system,
                device=self.device,
            )
        )
        return grid, extra_features, valid_player_actions
