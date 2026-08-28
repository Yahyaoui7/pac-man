"""Actor-Critic PyTorch model for Pac-Man with GRU memory."""

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
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, ACTION_COUNT)
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
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
    def __init__(self, extra_feature_count: int = EXTRA_FEATURE_COUNT) -> None:
        super().__init__()
        self.backbone = PacmanCNNBackbone(
            dropout_prob=0.1,
            extra_feature_count=extra_feature_count,
        )
        self.action_head = nn.Linear(128, ACTION_COUNT)

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
    """Load checkpoint weights with automatic shape matching for feature dimension changes."""
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
        target_k = "actor" + k[len("action_head"):] if k.startswith("action_head") else k
        if target_k in policy_dict:
            if policy_dict[target_k].shape == v.shape:
                matched_dict[target_k] = v
            elif (
                target_k == "backbone.proj.0.weight"
                and policy_dict[target_k].shape[0] == v.shape[0]
            ):
                # Copy existing slice and zero-initialize new feature weights
                min_cols = min(policy_dict[target_k].shape[1], v.shape[1])
                matched = policy_dict[target_k].clone()
                matched[:, :min_cols] = v[:, :min_cols]
                matched_dict[target_k] = matched

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
    model = PlayerActorCritic()
    dummy_grid = torch.zeros((2, CNN_CHANNEL_COUNT, 25, 50), dtype=torch.float32)
    dummy_features = torch.zeros((2, EXTRA_FEATURE_COUNT), dtype=torch.float32)

    # Test with hidden state
    logits, value, hidden = model(dummy_grid, dummy_features, None)
    print("PlayerActorCritic with GRU test:")
    print(f"  Input grid shape: {tuple(dummy_grid.shape)}")
    print(f"  Input features shape: {tuple(dummy_features.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)}")
    print(f"  Output value shape: {tuple(value.shape)}")
    print(f"  Hidden state shape: {tuple(hidden.shape)}")


if __name__ == "__main__":
    main()
