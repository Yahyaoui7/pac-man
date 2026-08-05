"""Shared PyTorch CNN backbone for spatial grid processing and feature fusion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AI_arena.data.constants import (
    CNN_CHANNEL_COUNT,
    EXTRA_FEATURE_COUNT,
)


class PacmanCNNBackbone(nn.Module):
    """Shared spatial CNN encoder and dense feature-fusion trunk."""

    def __init__(
        self,
        dropout_prob: float = 0.1,
        extra_feature_count: int = EXTRA_FEATURE_COUNT,
    ) -> None:
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

        # 50x25 observation reduced by two 2x2 pools -> 12x6 grid
        flattened_spatial_dim = 128 * 12 * 6
        total_input_dim = flattened_spatial_dim + extra_feature_count

        self.trunk = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def extract_features(self, grid: Tensor, extra_features: Tensor) -> Tensor:
        """Combine the spatial grid and state features into a latent vector."""
        spatial_features = torch.flatten(self.cnn(grid), start_dim=1)
        combined = torch.cat((spatial_features, extra_features), dim=1)
        return self.trunk(combined)
