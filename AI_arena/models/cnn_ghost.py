"""CNN model used to predict actions for all four Pac-Man ghosts."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    EXTRA_FEATURE_COUNT,
    GHOST_COUNT,
)
from AI_arena.models.cnn_backbone import PacmanCNNBackbone


class GhostCNN(nn.Module):
    """Predict action logits for all four ghosts using the shared PacmanCNNBackbone."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = PacmanCNNBackbone(dropout_prob=0.3)
        self.head = nn.Linear(128, GHOST_COUNT * ACTION_COUNT)

    def forward(self, grid: Tensor, extra_features: Tensor) -> Tensor:
        """Return logits shaped as [batch, GHOST_COUNT, ACTION_COUNT]."""
        latent = self.backbone.extract_features(grid, extra_features)
        logits = self.head(latent)
        return logits.view(-1, GHOST_COUNT, ACTION_COUNT)


def main() -> None:
    """Smoke test to verify forward pass of GhostCNN."""
    model = GhostCNN()
    dummy_grid = torch.zeros((2, CNN_CHANNEL_COUNT, 25, 50), dtype=torch.float32)
    dummy_features = torch.zeros((2, EXTRA_FEATURE_COUNT), dtype=torch.float32)

    logits = model(dummy_grid, dummy_features)
    print("GhostCNN model test:")
    print(f"  Input grid shape: {tuple(dummy_grid.shape)}")
    print(f"  Input features shape: {tuple(dummy_features.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
