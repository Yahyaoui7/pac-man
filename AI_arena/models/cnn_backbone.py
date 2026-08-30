"""CNN-GRU backbone v3 — Dual-Tower, numerically stable, with GRU LayerNorm.

Changes from v2:
  - Added LayerNorm after GRU (prevents hidden state drift over 32-step BPTT)
  - Added dropout=0.1 to GRU (prevents 2-layer overfitting on small batches)
  - Kept all v2 improvements: dual-tower fusion, SE blocks, two-stage
    compression, LayerNorm on scalars, LeakyReLU for sign preservation
"""

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


class SEBlock(nn.Module):
    """Channel attention — lets the CNN learn 'ghost channels matter now'."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class PacmanCNNBackbone(nn.Module):
    """Spatial CNN + Scalar MLP, fused equally. No gating (kept it simple
    and stable). Scalars are 50% of the fusion input — 75× louder than
    the old 0.66%.

    Architecture:
      - Spatial tower: CNN with ResBlocks + SEBlocks → 128-dim
      - Scalar tower: LayerNorm → MLP → 128-dim
      - Fusion: concat (256) → MLP → gru_hidden_size
      - GRU: 2-layer, with dropout between layers
      - LayerNorm: stabilizes GRU output over long sequences
      - Output: Linear → ReLU → Dropout → 256-dim
    """

    def __init__(
        self,
        dropout_prob: float = 0.1,
        extra_feature_count: int = EXTRA_FEATURE_COUNT,
        gru_hidden_size: int = 384,
        gru_num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        # ── Spatial tower (dense map information) ──
        self.cnn = nn.Sequential(
            nn.Conv2d(CNN_CHANNEL_COUNT, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            ResBlock(64, dilation=1),
            ResBlock(64, dilation=2),
            SEBlock(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            ResBlock(64, dilation=4),
            SEBlock(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResBlock(32, dilation=1),
        )

        # Two-stage compression: less brutal than 10,400 → 256 in one shot
        self.spatial_compress = nn.Sequential(
            nn.Linear(32 * 25 * 13, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # ── Scalar tower (exact coordinates, danger flags, deltas) ──
        # LeakyReLU preserves sign (negative = left / moving away) but is
        # piecewise-linear, so it cannot overflow in float16 like GELU.
        self.scalar_encoder = nn.Sequential(
            nn.LayerNorm(extra_feature_count),
            nn.Linear(extra_feature_count, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
        )

        # ── Fusion: 128 (spatial) + 128 (scalar) = 256 ──
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, gru_hidden_size),
            nn.ReLU(),
        )

        # ── GRU memory (2-layer with dropout between layers) ──
        self.gru = nn.GRU(
            gru_hidden_size,
            gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=0.1 if gru_num_layers > 1 else 0.0,  # only between layers
        )

        # ── LayerNorm after GRU — stabilizes output over long sequences ──
        # Critical for 32-step BPTT: prevents hidden state from drifting
        # or exploding across the sequence chunk.
        self.gru_ln = nn.LayerNorm(gru_hidden_size)

        self.out = nn.Sequential(
            nn.Linear(gru_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier init is safer than Kaiming when activations are mixed
        (ReLU + LeakyReLU)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        grid: Tensor,
        extra_features: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # ═══════════════════════════════════════════════════════════════
        #  Sequence chunk mode: (batch, seq_len, C, H, W)
        # ═══════════════════════════════════════════════════════════════
        if grid.ndim == 5:
            b, l, c, h, w = grid.shape
            grid_flat = grid.reshape(b * l, c, h, w)
            extra_flat = extra_features.reshape(b * l, -1)

            # Spatial path
            x_s = self.cnn(grid_flat)
            x_s = torch.flatten(x_s, start_dim=1)
            x_s = self.spatial_compress(x_s)  # [B*L, 128]

            # Scalar path
            x_sc = self.scalar_encoder(extra_flat)  # [B*L, 128]

            # Equal-weight fusion: scalars are 50% of the input
            x_f = torch.cat([x_s, x_sc], dim=-1)  # [B*L, 256]
            x_f = self.fusion(x_f)  # [B*L, gru_hidden_size]
            x_f = x_f.view(b, l, self.gru_hidden_size)

            # GRU with reset-safe masking
            if dones is not None:
                dones_t = dones.view(b, l, 1).to(device=grid.device, dtype=x_f.dtype)
                h = (
                    hidden
                    if hidden is not None
                    else torch.zeros(
                        self.gru_num_layers,
                        b,
                        self.gru_hidden_size,
                        device=grid.device,
                        dtype=x_f.dtype,
                    )
                )
                outs: list[Tensor] = []
                for t in range(l):
                    mask = dones_t[:, t : t + 1].permute(1, 0, 2)  # (1, b, 1)
                    h = h * (1.0 - mask)
                    out_t, h = self.gru(x_f[:, t : t + 1], h)
                    out_t = self.gru_ln(out_t)  # ← normalize each step
                    outs.append(out_t)
                out = torch.cat(outs, dim=1)
                hidden = h
            else:
                out, hidden = self.gru(x_f, hidden)
                out = self.gru_ln(out)  # ← normalize full sequence

            return self.out(out), hidden

        # ═══════════════════════════════════════════════════════════════
        #  Single step mode: (batch, C, H, W)
        # ═══════════════════════════════════════════════════════════════
        x_s = self.cnn(grid)
        x_s = torch.flatten(x_s, start_dim=1)
        x_s = self.spatial_compress(x_s)

        x_sc = self.scalar_encoder(extra_features)

        x_f = torch.cat([x_s, x_sc], dim=-1)
        x_f = self.fusion(x_f).unsqueeze(1)

        out, hidden = self.gru(x_f, hidden)
        out = self.gru_ln(out)  # ← normalize single step
        out = out.squeeze(1)

        return self.out(out), hidden
