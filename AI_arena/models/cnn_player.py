"""Actor-Critic PyTorch model for Pac-Man player reinforcement learning."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    EXTRA_FEATURE_COUNT,
)
from AI_arena.models.cnn_backbone import PacmanCNNBackbone


class PlayerActorCritic(nn.Module):
    """Actor-Critic model taking spatial grid and state features to output action logits and state value."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = PacmanCNNBackbone(dropout_prob=0.1)
        self.actor = nn.Linear(128, ACTION_COUNT)
        self.critic = nn.Linear(128, 1)

    def forward(
        self,
        grid: Tensor,
        extra_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (action_logits [batch, 4], state_value [batch, 1])."""
        latent = self.backbone.extract_features(grid, extra_features)
        logits = self.actor(latent)
        value = self.critic(latent)
        return logits, value


def main() -> None:
    """Smoke test to verify forward pass of PlayerActorCritic."""
    model = PlayerActorCritic()
    dummy_grid = torch.zeros((2, CNN_CHANNEL_COUNT, 50, 25), dtype=torch.float32)
    dummy_features = torch.zeros((2, EXTRA_FEATURE_COUNT), dtype=torch.float32)

    logits, value = model(dummy_grid, dummy_features)
    print("PlayerActorCritic model test:")
    print(f"  Input grid shape: {tuple(dummy_grid.shape)}")
    print(f"  Input features shape: {tuple(dummy_features.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)}")
    print(f"  Output value shape: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
