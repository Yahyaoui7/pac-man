"""Inference controller for CNN-based Pac-Man player model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from AI_arena.models.cnn_player import PlayerActorCritic
from src.graphics.entitys.ghost import Ghost
from src.graphics.entitys.player import Player

from AI_arena.player.data.observation import format_player_observation

DEFAULT_STAGE1_PATH = Path(__file__).parent.parent / "models" / "player_rl_stage2.pt"
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
                path = Path(__file__).parent.parent / "models" / "player_sl_best.pt"
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
        self.last_action_idx: int | None = None
        self._hidden: torch.Tensor | None = None  # ← GRU memory persists across steps

    def reset_state(self) -> None:
        """Reset internal state history (e.g. between games)."""
        self.last_action_idx = None
        self._hidden = None  # ← wipe GRU memory on new game

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
            # Pass hidden state into model, receive updated hidden state back
            logits, value, self._hidden = self.model(grid, extra_features, self._hidden)
            masked_logits = logits.masked_fill(~valid_actions, -1e9)
            probs = torch.softmax(masked_logits, dim=-1)[0]
            if sample:
                action_index = int(torch.multinomial(probs, 1).item())
            else:
                action_index = int(torch.argmax(masked_logits, dim=-1).item())

        self.last_action_idx = action_index
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
        return format_player_observation(
            maze=maze,
            pellets=pellets,
            player=player,
            ghosts=ghosts,
            movement=movement_system,
            device=self.device,
        )
