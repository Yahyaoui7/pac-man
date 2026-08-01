"""Actor-Critic PyTorch model for Pac-Man player reinforcement learning."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AI_arena.cnn_dataset import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    EXTRA_FEATURE_COUNT,
)


class PlayerActorCritic(nn.Module):
    """Actor-Critic neural network taking spatial grid and extra state features to output action logits and state value."""

    def __init__(self) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(CNN_CHANNEL_COUNT, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 50x25 spatial input undergoes two 2x2 max-pooling operations -> 12x6 grid
        flattened_cnn_dim = 128 * 12 * 6
        total_feature_dim = flattened_cnn_dim + EXTRA_FEATURE_COUNT

        self.trunk = nn.Sequential(
            nn.Linear(total_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.actor = nn.Linear(128, ACTION_COUNT)
        self.critic = nn.Linear(128, 1)

    def forward(
        self,
        grid: Tensor,
        extra_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (action_logits [batch, 4], state_value [batch, 1])."""
        spatial = torch.flatten(self.cnn(grid), start_dim=1)
        combined = torch.cat((spatial, extra_features), dim=1)
        hidden = self.trunk(combined)

        logits = self.actor(hidden)
        value = self.critic(hidden)

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
