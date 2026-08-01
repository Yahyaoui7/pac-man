"""PPO training pipeline for Pac-Man player model against BFS ghosts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from AI_arena.pacman_player_env import PacmanPlayerEnv
from AI_arena.player_cnn_model import PlayerActorCritic

DEFAULT_MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODEL_DIR / "player_rl.pt"


class RolloutSample(NamedTuple):
    grid: torch.Tensor
    features: torch.Tensor
    valid_actions: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    value: torch.Tensor


def train_player_ppo(
    num_updates: int = 100,
    rollout_steps: int = 512,
    ppo_epochs: int = 4,
    minibatch_size: int = 64,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    seed: int = 42,
) -> None:
    """Train Pac-Man player model using Proximal Policy Optimization (PPO)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting PPO training for Pac-Man on device: {device}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    env = PacmanPlayerEnv(seed=seed)
    policy = PlayerActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    obs = env.reset()

    for update in range(1, num_updates + 1):
        rollout_grids = []
        rollout_features = []
        rollout_valid_actions = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        episode_rewards = []
        current_ep_reward = 0.0
        completed_episodes = 0

        # 1. Rollout phase
        for _ in range(rollout_steps):
            grid, features, valid_actions = obs
            grid = grid.to(device)
            features = features.to(device)
            valid_actions = valid_actions.to(device)

            with torch.no_grad():
                logits, value = policy(grid, features)
                masked_logits = logits.masked_fill(~valid_actions, -1e9)
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action.item())
            current_ep_reward += reward

            rollout_grids.append(grid)
            rollout_features.append(features)
            rollout_valid_actions.append(valid_actions)
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(torch.tensor([reward], device=device, dtype=torch.float32))
            rollout_dones.append(torch.tensor([done], device=device, dtype=torch.float32))
            rollout_values.append(value.squeeze(-1))

            if done:
                obs = env.reset()
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                completed_episodes += 1
            else:
                obs = next_obs

        # 2. GAE Advantage Calculation
        with torch.no_grad():
            last_grid, last_features, _ = obs
            _, next_value = policy(last_grid.to(device), last_features.to(device))
            next_value = next_value.squeeze(-1)

        b_grids = torch.cat(rollout_grids, dim=0)
        b_features = torch.cat(rollout_features, dim=0)
        b_valid_actions = torch.cat(rollout_valid_actions, dim=0)
        b_actions = torch.cat(rollout_actions, dim=0)
        b_log_probs = torch.cat(rollout_log_probs, dim=0)
        b_rewards = torch.cat(rollout_rewards, dim=0)
        b_dones = torch.cat(rollout_dones, dim=0)
        b_values = torch.cat(rollout_values, dim=0)

        advantages = torch.zeros_like(b_rewards, device=device)
        last_gae_lam = 0.0
        for t in reversed(range(rollout_steps)):
            if t == rollout_steps - 1:
                next_non_terminal = 1.0 - b_dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - b_dones[t]
                next_val = b_values[t + 1]

            delta = b_rewards[t] + gamma * next_val * next_non_terminal - b_values[t]
            advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + b_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. PPO Optimization Epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        dataset_size = rollout_steps

        for _ in range(ppo_epochs):
            permutation = torch.randperm(dataset_size, device=device)
            for start_idx in range(0, dataset_size, minibatch_size):
                mb_idx = permutation[start_idx : start_idx + minibatch_size]

                mb_grid = b_grids[mb_idx]
                mb_features = b_features[mb_idx]
                mb_valid_actions = b_valid_actions[mb_idx]
                mb_actions = b_actions[mb_idx]
                mb_old_log_probs = b_log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, values = policy(mb_grid, mb_features)
                masked_logits = logits.masked_fill(~mb_valid_actions, -1e9)
                dist = Categorical(logits=masked_logits)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()

        avg_reward = (
            sum(episode_rewards) / len(episode_rewards)
            if episode_rewards
            else current_ep_reward
        )
        if update % 5 == 0 or update == 1 or update == num_updates:
            print(
                f"Update {update:03d}/{num_updates:03d} | "
                f"Completed Ep: {completed_episodes:02d} | "
                f"Avg Ep Reward: {avg_reward:7.2f} | "
                f"Policy Loss: {total_policy_loss / (ppo_epochs * dataset_size):.4f} | "
                f"Value Loss: {total_value_loss / (ppo_epochs * dataset_size):.4f}"
            )

        # Save checkpoint periodically
        if update % 20 == 0 or update == num_updates:
            torch.save(policy.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

    print("Training finished successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Trainer for Pac-Man Player Model")
    parser.add_argument("--num-updates", type=int, default=50, help="Number of PPO update iterations")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Steps per rollout")
    parser.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT_PATH), help="Save path")
    args = parser.parse_args()

    train_player_ppo(
        num_updates=args.num_updates,
        rollout_steps=args.rollout_steps,
        checkpoint_path=Path(args.checkpoint_path),
    )


if __name__ == "__main__":
    main()
