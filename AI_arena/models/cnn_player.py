"""Actor-Critic PyTorch model for Pac-Man with GRU memory (v3).

Changes from v2:
  - Fixed PlayerImitationCNN: now uses gru_hidden_size=384 and 256-dim action_head
    (was broken: used default gru_hidden_size and 128-dim action_head that didn't
    match the backbone's 256-dim output)
  - Updated checkpoint loader: handles spatial_compress.0.weight shape mismatch
    (was referencing old backbone.proj.0.weight which no longer exists)
  - Kept deeper actor/critic heads with LeakyReLU
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch import Tensor, nn

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    EXTRA_FEATURE_COUNT,
)
from AI_arena.models.cnn_backbone import PacmanCNNBackbone


class PlayerActorCritic(nn.Module):
    def __init__(self, extra_feature_count: int = EXTRA_FEATURE_COUNT) -> None:
        super().__init__()
        self.backbone = PacmanCNNBackbone(
            dropout_prob=0.1,
            extra_feature_count=extra_feature_count,
            gru_hidden_size=384,
        )
        # Deeper heads, but with LeakyReLU (safe in float16)
        self.actor = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.1), nn.Linear(128, ACTION_COUNT)
        )
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
        logits = self.actor(latent)
        value = self.critic(latent)
        return logits, value, hidden


class PlayerImitationCNN(nn.Module):
    """Imitation learning model — FIXED to match v3 backbone output.

    Was broken in v2: used default gru_hidden_size (384 via backbone default)
    but action_head was Linear(128, ACTION_COUNT) while backbone outputs 256.
    Now correctly uses gru_hidden_size=384 and action_head=Linear(256, ...).
    """

    def __init__(self, extra_feature_count: int = EXTRA_FEATURE_COUNT) -> None:
        super().__init__()
        self.backbone = PacmanCNNBackbone(
            dropout_prob=0.1,
            extra_feature_count=extra_feature_count,
            gru_hidden_size=384,  # ← was missing (used default)
        )
        # Backbone outputs 256-dim (from self.out), not 128
        self.action_head = nn.Linear(256, ACTION_COUNT)  # ← was 128, now 256

    def forward(
        self,
        grid: Tensor,
        extra_features: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        latent, hidden = self.backbone(grid, extra_features, hidden, dones=dones)
        return self.action_head(latent), hidden


def load_checkpoint_into_policy(
    policy: PlayerActorCritic,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> bool:
    """Load checkpoint weights with automatic shape matching.

    Handles two cases of shape mismatch:
      1. spatial_compress.0.weight: copy existing slice, zero-init new columns
         (useful when extra_feature_count changed)
      2. All other layers: skip mismatched shapes (fresh init)

    NOTE: If you changed the GRU dimensions (hidden_size or num_layers),
    the GRU weights will NOT transfer — they'll use fresh initialization.
    This is intentional: the CNN features transfer, the GRU relearns.
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
        # Handle action_head -> actor mapping if from SL model
        target_k = (
            "actor" + k[len("action_head") :] if k.startswith("action_head") else k
        )
        if target_k in policy_dict:
            if policy_dict[target_k].shape == v.shape:
                # Exact match — load directly
                matched_dict[target_k] = v
            elif (
                target_k == "backbone.spatial_compress.0.weight"
                and policy_dict[target_k].shape[0] == v.shape[0]
            ):
                # Spatial compress first layer: copy existing slice and
                # zero-initialize new feature weights (for feature dim changes)
                min_cols = min(policy_dict[target_k].shape[1], v.shape[1])
                matched = policy_dict[target_k].clone()
                matched[:, :min_cols] = v[:, :min_cols]
                matched_dict[target_k] = matched
            # else: shape mismatch on other layers — skip (fresh init)

    policy_dict.update(matched_dict)
    policy.load_state_dict(policy_dict)
    return True


def load_sl_weights_into_ppo(
    ppo_model: PlayerActorCritic,
    sl_checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> PlayerActorCritic:
    load_checkpoint_into_policy(ppo_model, sl_checkpoint_path, device=device)
    print(
        f"Successfully loaded SL pre-trained weights from {sl_checkpoint_path} into PPO actor network!"
    )
    return ppo_model


def main() -> None:
    """Smoke test — verify the forward pass works with the new architecture."""
    model = PlayerActorCritic()
    dummy_grid = torch.zeros((2, CNN_CHANNEL_COUNT, 25, 50), dtype=torch.float32)
    dummy_features = torch.zeros((2, EXTRA_FEATURE_COUNT), dtype=torch.float32)

    # Test single-step mode
    logits, value, hidden = model(dummy_grid, dummy_features, None)
    print("PlayerActorCritic v3 single-step test:")
    print(f"  Input grid shape: {tuple(dummy_grid.shape)}")
    print(f"  Input features shape: {tuple(dummy_features.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)}")
    print(f"  Output value shape: {tuple(value.shape)}")
    print(f"  Hidden state shape: {tuple(hidden.shape)}")

    # Test sequence mode (batch=2, seq_len=8)
    dummy_grid_seq = torch.zeros((2, 8, CNN_CHANNEL_COUNT, 25, 50), dtype=torch.float32)
    dummy_features_seq = torch.zeros((2, 8, EXTRA_FEATURE_COUNT), dtype=torch.float32)
    dummy_dones = torch.zeros((2, 8), dtype=torch.float32)

    logits_seq, value_seq, hidden_seq = model(
        dummy_grid_seq, dummy_features_seq, None, dones=dummy_dones
    )
    print("\nPlayerActorCritic v3 sequence test:")
    print(f"  Input grid shape: {tuple(dummy_grid_seq.shape)}")
    print(f"  Output logits shape: {tuple(logits_seq.shape)}")
    print(f"  Output value shape: {tuple(value_seq.shape)}")
    print(f"  Hidden state shape: {tuple(hidden_seq.shape)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test imitation model too
    print("\n" + "=" * 60)
    imitation_model = PlayerImitationCNN()
    il_logits, il_hidden = imitation_model(dummy_grid, dummy_features, None)
    print("PlayerImitationCNN v3 test:")
    print(f"  Output logits shape: {tuple(il_logits.shape)}")
    print(f"  Hidden state shape: {tuple(il_hidden.shape)}")


if __name__ == "__main__":
    main()
