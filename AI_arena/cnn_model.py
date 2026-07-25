"""CNN model used to predict one action for each Pac-Man ghost."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AI_arena.cnn_dataset import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    EXTRA_FEATURE_COUNT,
    GHOST_COUNT,
)


class GhostCNN(nn.Module):
    """Combine spatial maze features with non-spatial game-state features."""

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

        # Two 2x2 pools turn the fixed 50x25 observation into 12x6. Keeping
        # that full map preserves exact spatial regions; adaptive 4x4 pooling
        # previously discarded location detail needed for maze navigation.
        flattened_feature_count = 128 * 12 * 6
        self.head = nn.Sequential(
            nn.Linear(flattened_feature_count + EXTRA_FEATURE_COUNT, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, GHOST_COUNT * ACTION_COUNT),
        )

    def forward(self, grid: Tensor, extra_features: Tensor) -> Tensor:
        """Return logits shaped as [batch, ghost, action]."""

        spatial_features = torch.flatten(self.cnn(grid), start_dim=1)
        combined_features = torch.cat(
            (spatial_features, extra_features),
            dim=1,
        )
        logits = self.head(combined_features)
        return logits.view(-1, GHOST_COUNT, ACTION_COUNT)
