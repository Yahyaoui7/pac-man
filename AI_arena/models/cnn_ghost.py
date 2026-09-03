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
        # backbone.out produces 256-dim vectors (Linear(gru_hidden) → ReLU → Dropout → 256)
        self.backbone = PacmanCNNBackbone(dropout_prob=0.3, use_gru=False)
        self.heads = nn.ModuleList(
            [nn.Linear(256, ACTION_COUNT) for _ in range(GHOST_COUNT)]
        )

    def forward(self, grid: Tensor, extra_features: Tensor) -> Tensor:
        """Return logits shaped as [batch, GHOST_COUNT, ACTION_COUNT]."""
        # backbone.forward() returns (out, hidden); we only need the output tensor
        latent, _ = self.backbone(grid, extra_features)

        # Pass latent through each ghost's dedicated head
        ghost_logits = [head(latent) for head in self.heads]

        # Stack the results into [batch, GHOST_COUNT, ACTION_COUNT]
        return torch.stack(ghost_logits, dim=1)


def main() -> None:
    """Smoke test to verify forward pass of GhostCNN."""
    model = GhostCNN()
    dummy_grid = torch.zeros(
        (2, CNN_CHANNEL_COUNT, 25, 50), dtype=torch.float32
    )
    dummy_features = torch.zeros((2, EXTRA_FEATURE_COUNT), dtype=torch.float32)

    logits = model(dummy_grid, dummy_features)
    print("GhostCNN model test:")
    print(f"  Input grid shape: {tuple(dummy_grid.shape)}")
    print(f"  Input features shape: {tuple(dummy_features.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
