"""CNN-GRU backbone for Pac-Man with temporal memory."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AI_arena.data.constants import CNN_CHANNEL_COUNT, EXTRA_FEATURE_COUNT


class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + x)


class PacmanCNNBackbone(nn.Module):
    """Spatial CNN encoder + GRU temporal memory."""

    def __init__(
        self,
        dropout_prob: float = 0.1,
        extra_feature_count: int = EXTRA_FEATURE_COUNT,
    ) -> None:
        super().__init__()

        # ── CNN: per-frame spatial encoder (identical to your current one) ──
        self.cnn = nn.Sequential(
            nn.Conv2d(CNN_CHANNEL_COUNT, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResBlock(32, dilation=1),
            ResBlock(32, dilation=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            ResBlock(64, dilation=4),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResBlock(64, dilation=1),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
        )

        flat_dim = 32 * 25 * 13  # 10,400
        total_dim = flat_dim + extra_feature_count  # 10,445

        # ── Projection to GRU input size ──
        self.proj = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
        )

        # ── GRU: the memory cell ──
        # input_size = 128 (from proj)
        # hidden_size = 128 (memory capacity)
        # batch_first = True because we feed (batch, seq=1, features)
        self.gru = nn.GRU(128, 128, batch_first=True)

        # ── Output head ──
        self.out = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

    def forward(
        self,
        grid: Tensor,
        extra_features: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            grid: (batch, 7, 25, 50) or (batch, seq_len, 7, 25, 50)
            extra_features: (batch, extra_feature_count) or (batch, seq_len, extra_feature_count)
            hidden: (1, batch, 128) or None
            dones: (batch, seq_len) or (batch, seq_len, 1) or None — 1.0 if step is after reset

        Returns:
            latent: (batch, 128) or (batch, seq_len, 128)
            hidden: (1, batch, 128)
        """
        if grid.ndim == 5:
            # Sequence chunk mode: (batch, seq_len, channels, height, width)
            b, l, c, h, w = grid.shape
            grid_flat = grid.reshape(b * l, c, h, w)
            extra_flat = extra_features.reshape(b * l, -1)

            x = self.cnn(grid_flat)
            x = torch.flatten(x, start_dim=1)
            x = torch.cat((x, extra_flat), dim=1)
            x = self.proj(x)  # (b * l, 128)

            x = x.view(b, l, 128)  # (batch, seq_len, 128)

            if dones is not None:
                dones_tensor = dones.view(b, l, 1).to(device=grid.device, dtype=x.dtype)
                h = (
                    hidden
                    if hidden is not None
                    else torch.zeros(1, b, 128, device=grid.device, dtype=x.dtype)
                )
                outs: list[Tensor] = []
                for t in range(l):
                    mask = dones_tensor[:, t : t + 1].permute(1, 0, 2)  # (1, b, 1)
                    h = h * (1.0 - mask)
                    out_t, h = self.gru(x[:, t : t + 1], h)
                    outs.append(out_t)
                out = torch.cat(outs, dim=1)
                hidden = h
            else:
                out, hidden = self.gru(x, hidden)  # out: (batch, seq_len, 128)

            return self.out(out), hidden

        # Single step mode: (batch, channels, height, width)
        x = self.cnn(grid)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, extra_features), dim=1)
        x = self.proj(x)  # (batch, 128)

        x = x.unsqueeze(1)  # (batch, 1, 128)
        out, hidden = self.gru(x, hidden)
        out = out.squeeze(1)  # (batch, 128)

        return self.out(out), hidden
