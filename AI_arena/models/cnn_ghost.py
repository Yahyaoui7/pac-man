"""CNN model used to predict actions for all four Pac-Man ghosts."""

from __future__ import annotations

import torch
from pathlib import Path
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


class GhostActorCritic(nn.Module):
    """Actor-Critic model for ghosts using GRU, ready for RL training."""

    def __init__(self, extra_feature_count: int = EXTRA_FEATURE_COUNT) -> None:
        super().__init__()
        self.backbone = PacmanCNNBackbone(
            dropout_prob=0.1,
            extra_feature_count=extra_feature_count,
            gru_hidden_size=384,
            use_gru=True,
        )
        # Actor Heads: One for each ghost
        self.actors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(256, 128), nn.LeakyReLU(0.1), nn.Linear(128, ACTION_COUNT)
                )
                for _ in range(GHOST_COUNT)
            ]
        )
        # Critic Head: One shared value for the whole board state
        self.critic = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.1), nn.Linear(128, 1)
        )

    def forward(
        self,
        grid: Tensor,
        extra_features: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        latent, hidden = self.backbone(grid, extra_features, hidden, dones=dones)

        # Get actions for all 4 ghosts
        ghost_logits = [actor(latent) for actor in self.actors]
        logits = torch.stack(ghost_logits, dim=-2)

        # Get the board value
        value = self.critic(latent)

        return logits, value, hidden


def load_sl_ghost_weights_into_rl(
    policy: GhostActorCritic,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> bool:
    """Load SL checkpoint weights into the RL actor-critic model.

    Copies backbone CNN weights exactly.
    Leaves the GRU randomly initialized (SL model didn't have one).
    The heads will also randomly initialize because the RL heads are deeper.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return False

    state_dict = torch.load(path, map_location=device, weights_only=True)
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        state_dict = state_dict["model_state"]

    policy_dict = policy.state_dict()
    matched_dict = {}

    for k, v in state_dict.items():
        if k.startswith("heads."):
            # We skip copying the heads because the RL actors have a different
            # deeper architecture (Sequential with LeakyReLU).
            # Only the spatial backbone needs to be transferred!
            continue

        if k in policy_dict:
            if policy_dict[k].shape == v.shape:
                matched_dict[k] = v
            elif (
                k == "backbone.spatial_compress.0.weight"
                and policy_dict[k].shape[0] == v.shape[0]
            ):
                min_cols = min(policy_dict[k].shape[1], v.shape[1])
                matched = policy_dict[k].clone()
                matched[:, :min_cols] = v[:, :min_cols]
                matched_dict[k] = matched

    policy_dict.update(matched_dict)
    policy.load_state_dict(policy_dict)
    return True


def main() -> None:
    """Smoke test to verify forward pass of GhostCNN."""
    model = GhostCNN()
    dummy_grid = torch.zeros(
        (2, CNN_CHANNEL_COUNT, 25, 50), dtype=torch.float32
    )
    dummy_features = torch.zeros((2, EXTRA_FEATURE_COUNT), dtype=torch.float32)

    logits = model(dummy_grid, dummy_features)
    print("--- SL GhostCNN model test ---")
    print(f"  Input grid shape: {tuple(dummy_grid.shape)}")
    print(f"  Input features shape: {tuple(dummy_features.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)}")

    rl_model = GhostActorCritic()
    rl_logits, rl_value, rl_hidden = rl_model(dummy_grid, dummy_features)
    print("\n--- RL GhostActorCritic test ---")
    print(f"  Output logits shape: {tuple(rl_logits.shape)}")
    print(f"  Output value shape: {tuple(rl_value.shape)}")
    if rl_hidden is not None:
        print(f"  Hidden state shape: {tuple(rl_hidden.shape)}")


if __name__ == "__main__":
    main()
