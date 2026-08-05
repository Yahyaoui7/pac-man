"""Inference controller for CNN-based Pac-Man player model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from AI_arena.models.cnn_player import PlayerImitationCNN
from AI_arena.player.observation import (
    PLAYER_EXTRA_FEATURE_COUNT,
    format_player_observation,
)
from src.graphics.entitys.ghost import Ghost
from src.graphics.entitys.player import Player

DEFAULT_PLAYER_PATH = Path(__file__).parent.parent / "models" / "player_sl.pt"
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class CNNPlayerController:
    """Build live observations and predict Pac-Man's best move."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.model = PlayerImitationCNN(PLAYER_EXTRA_FEATURE_COUNT).to(
            self.device
        )
        path = DEFAULT_PLAYER_PATH if model_path is None else Path(model_path)

        if path.exists():
            weights = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(weights)
            print(f"Loaded supervised player checkpoint from {path}")
        else:
            print(
                f"Warning: supervised player checkpoint {path} not found. "
                "Using untrained weights."
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
        initial_pellet_count: int | None = None,
        sample: bool = False,
    ) -> str:
        """Construct state tensors and select a sampled or greedy action."""
        grid, extra_features, valid_actions = self._build_observation(
            maze,
            pellets,
            player,
            ghosts,
            movement_system,
            initial_pellet_count,
        )

        with torch.no_grad():
            logits = self.model(grid, extra_features)
            masked_logits = logits.masked_fill(~valid_actions, -1e9)
            probs = torch.softmax(masked_logits, dim=-1)[0]
            if sample:
                action_index = int(torch.multinomial(probs, 1).item())
            else:
                action_index = int(torch.argmax(masked_logits, dim=-1).item())

        chosen_action = DIRECTIONS[action_index]
        self.last_diagnostics = {
            "chosen_action": chosen_action,
            "probabilities": {
                direction: round(float(probs[index].item()), 4)
                for index, direction in enumerate(DIRECTIONS)
            },
            "logits": {
                d: round(float(logits[0, i].item()), 4)
                for i, d in enumerate(DIRECTIONS)
            },
            "valid_actions": {
                direction: bool(valid_actions[0, index].item())
                for index, direction in enumerate(DIRECTIONS)
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
        initial_pellet_count: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid, extra_features, valid_player_actions = format_player_observation(
            maze=maze,
            pellets=pellets,
            player=player,
            ghosts=ghosts,
            movement=movement_system,
            initial_pellet_count=initial_pellet_count,
            device=self.device,
        )
        return grid, extra_features, valid_player_actions
