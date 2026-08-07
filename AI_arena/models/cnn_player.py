"""Actor-Critic PyTorch model for Pac-Man player reinforcement learning."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    EXTRA_FEATURE_COUNT,
)
from AI_arena.models.cnn_backbone import PacmanCNNBackbone


class PlayerActorCritic(nn.Module):
    """Legacy actor-critic model retained for old RL checkpoints."""

    def __init__(self) -> None:
        super().__init__()
        # PPO compares action log-probabilities collected during the rollout
        # with probabilities produced during optimization. Dropout would add
        # unrelated randomness to that ratio, so the RL policy must be
        # deterministic for a fixed observation and set of weights.
        self.backbone = PacmanCNNBackbone(dropout_prob=0.0)
        self.actor = nn.Linear(128, ACTION_COUNT)
        self.critic = nn.Linear(128, 1)

    def forward(
        self,
        grid: Tensor,
        extra_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return (action_logits [batch, 4], state_value [batch, 1])."""
        latent = self.backbone.extract_features(grid, extra_features)
        logits = self.actor(latent)
        value = self.critic(latent.detach())
        return logits, value


class PlayerImitationCNN(nn.Module):
    """Pac-Man action classifier trained from expert demonstrations."""

    def __init__(self, extra_feature_count: int = EXTRA_FEATURE_COUNT) -> None:
        super().__init__()
        self.backbone = PacmanCNNBackbone(
            dropout_prob=0.1,
            extra_feature_count=extra_feature_count,
        )
        self.action_head = nn.Linear(128, ACTION_COUNT)

    def forward(self, grid: Tensor, extra_features: Tensor) -> Tensor:
        return self.action_head(
            self.backbone.extract_features(grid, extra_features)
        )


def load_sl_weights_into_ppo(
    ppo_model: PlayerActorCritic,
    sl_checkpoint_path: str,
    device: str | torch.device = "cpu",
) -> PlayerActorCritic:
    """Initialize a PlayerActorCritic PPO model with pre-trained SL weights."""
    sl_dict = torch.load(sl_checkpoint_path, map_location=device)
    if isinstance(sl_dict, dict) and "model_state" in sl_dict:
        sl_dict = sl_dict["model_state"]

    ppo_dict = ppo_model.state_dict()
    mapped_dict = {}

    for k, v in sl_dict.items():
        if k.startswith("action_head"):
            new_k = k.replace("action_head", "actor")
            if new_k in ppo_dict and ppo_dict[new_k].shape == v.shape:
                mapped_dict[new_k] = v
        elif k in ppo_dict and ppo_dict[k].shape == v.shape:
            mapped_dict[k] = v

    ppo_dict.update(mapped_dict)
    ppo_model.load_state_dict(ppo_dict)
    print(f"Successfully loaded SL pre-trained weights from {sl_checkpoint_path} into PPO actor network!")
    return ppo_model


def main() -> None:
    """Smoke test to verify forward pass of PlayerActorCritic."""
    model = PlayerActorCritic()
    dummy_grid = torch.zeros((2, CNN_CHANNEL_COUNT, 50, 25), dtype=torch.float32)
    dummy_features = torch.zeros((2, EXTRA_FEATURE_COUNT), dtype=torch.float32)

    logits, value = model(dummy_grid, dummy_features)
    print("PlayerActorCritic model test:")
    print(f"  Input grid shape: {tuple(dummy_grid.shape)}")
    print(f"  Input features shape: {tuple(dummy_features.shape)}")
    print(f"  Output logits shape: {tuple(logits.shape)}")
    print(f"  Output value shape: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
