"""Deep multi-scale CNN backbone for Pac-Man with SE attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AI_arena.data.constants import CNN_CHANNEL_COUNT, EXTRA_FEATURE_COUNT


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel-wise attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """Residual block with dilation."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=pad, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=pad, dilation=dilation
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


class PacmanCNNBackbone(nn.Module):
    """Multi-scale spatial encoder with SE attention and deep trunk."""

    def __init__(
        self,
        dropout_prob: float = 0.1,
        extra_feature_count: int = EXTRA_FEATURE_COUNT,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(CNN_CHANNEL_COUNT, 64, kernel_size=3, padding=1),
            nn.ReLU(),

        )

        self.local_group = nn.Sequential(
            ResBlock(64, dilation=1),
            SEBlock(64),
            ResBlock(64, dilation=1),
            SEBlock(64),
            ResBlock(64, dilation=2),
            SEBlock(64),
        )

        self.meso_group = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            ResBlock(128, dilation=4),
            SEBlock(128),
            ResBlock(128, dilation=4),
            SEBlock(128),
        )

        self.macro_group = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(),
            ResBlock(128, dilation=8),
            SEBlock(128),
        )

        self.down = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResBlock(128, dilation=1),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
        )

        flat_dim = 64 * 25 * 13
        total_dim = flat_dim + extra_feature_count

        self.trunk = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

    def extract_features(self, grid: Tensor, extra_features: Tensor) -> Tensor:
        x = self.stem(grid)
        x = self.local_group(x)
        x = self.meso_group(x)
        x = self.macro_group(x)
        x = self.down(x)
        x = self.bottleneck(x)
        spatial = torch.flatten(x, start_dim=1)
        combined = torch.cat((spatial, extra_features), dim=1)
        return self.trunk(combined)
