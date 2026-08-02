"""High-performance PPO training pipeline for Pac-Man player model with continuous checkpoint resuming."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from AI_arena.models.cnn_player import PlayerActorCritic
from AI_arena.player.player_env import PacmanPlayerEnv

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"


def train_player_ppo(
    stage: int = 1,
    num_updates: int = 100,
    rollout_steps: int = 512,
    ppo_epochs: int = 4,
    minibatch_size: int = 64,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    model_dir: Path = DEFAULT_MODEL_DIR,
    save_interval: int = 10,
    seed: int = 42,
    resume: bool = True,
    resume_path: Path | None = None,
) -> None:
    """Train Pac-Man player model using PPO with continuous checkpoint resuming."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"Starting Stage {stage} PPO Training for Pac-Man")
    print(f"Device: {device} | Total Updates: {num_updates} | Rollout Steps: {rollout_steps}")
    print(f"============================================================")

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"player_rl_stage{stage}.pt"

    env = PacmanPlayerEnv(seed=seed, stage=stage, device="cpu")
    policy = PlayerActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    if resume:
        target_load = resume_path or checkpoint_path
        if target_load.exists():
            weights = torch.load(target_load, map_location=device, weights_only=True)
            policy.load_state_dict(weights)
            print(f"SUCCESS: Resumed training from checkpoint: {target_load.name}")
        else:
            print("No existing checkpoint found. Starting with fresh initial weights.")
    else:
        print("Starting training with fresh initial weights (--fresh specified).")

    start_time = time.time()
    obs = env.reset()

    from collections import deque
    recent_episodes: deque[dict[str, float]] = deque(maxlen=20)
    current_ep_reward = 0.0
    current_ep_steps = 0
    total_completed_episodes = 0

    for update in range(1, num_updates + 1):
        update_start_time = time.time()
        rollout_grids = []
        rollout_features = []
        rollout_valid_actions = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_dones = []
        rollout_values = []

        completed_episodes_in_update = 0

        # Fast Rollout Collection Phase (CPU -> GPU)
        for _ in range(rollout_steps):
            grid, features, valid_actions = obs

            with torch.no_grad():
                logits, value = policy(grid.to(device), features.to(device))
                masked_logits = logits.masked_fill(~valid_actions.to(device), -1e9)
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action.item())
            current_ep_reward += reward
            current_ep_steps += 1

            rollout_grids.append(grid)
            rollout_features.append(features)
            rollout_valid_actions.append(valid_actions)
            rollout_actions.append(action.cpu())
            rollout_log_probs.append(log_prob.cpu())
            rollout_rewards.append(torch.tensor([reward], dtype=torch.float32))
            rollout_dones.append(torch.tensor([done], dtype=torch.float32))
            rollout_values.append(value.squeeze(-1).cpu())

            if done:
                obs = env.reset()
                recent_episodes.append(
                    {
                        "reward": current_ep_reward,
                        "pellets": float(info["pellets_eaten"]),
                        "pct": float(info["completion_pct"]),
                        "steps": float(current_ep_steps),
                    }
                )
                current_ep_reward = 0.0
                current_ep_steps = 0
                completed_episodes_in_update += 1
                total_completed_episodes += 1
            else:
                obs = next_obs

        # Bulk Transfer Rollout Data to GPU & GAE Calculation
        with torch.no_grad():
            last_grid, last_features, _ = obs
            _, next_value = policy(last_grid.to(device), last_features.to(device))
            next_value = next_value.squeeze(-1)

        b_grids = torch.cat(rollout_grids, dim=0).to(device)
        b_features = torch.cat(rollout_features, dim=0).to(device)
        b_valid_actions = torch.cat(rollout_valid_actions, dim=0).to(device)
        b_actions = torch.cat(rollout_actions, dim=0).to(device)
        b_log_probs = torch.cat(rollout_log_probs, dim=0).to(device)
        b_rewards = torch.cat(rollout_rewards, dim=0).to(device)
        b_dones = torch.cat(rollout_dones, dim=0).to(device)
        b_values = torch.cat(rollout_values, dim=0).to(device)

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

        # PPO GPU Optimization Epochs
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

        update_elapsed = time.time() - update_start_time
        total_elapsed = time.time() - start_time

        if recent_episodes:
            avg_reward = sum(ep["reward"] for ep in recent_episodes) / len(recent_episodes)
            avg_pellets = sum(ep["pellets"] for ep in recent_episodes) / len(recent_episodes)
            max_pellets = int(max(ep["pellets"] for ep in recent_episodes))
            avg_pct = sum(ep["pct"] for ep in recent_episodes) / len(recent_episodes)
            max_pct = max(ep["pct"] for ep in recent_episodes)
        else:
            avg_reward = current_ep_reward
            avg_pellets = float(info.get("pellets_eaten", 0))
            max_pellets = int(info.get("pellets_eaten", 0))
            avg_pct = float(info.get("completion_pct", 0.0))
            max_pct = float(info.get("completion_pct", 0.0))

        avg_policy_loss = total_policy_loss / (ppo_epochs * dataset_size)
        avg_value_loss = total_value_loss / (ppo_epochs * dataset_size)

        if update % 5 == 0 or update == 1 or update == num_updates:
            print(
                f"Upd {update:03d}/{num_updates:03d} | "
                f"Tot Ep: {total_completed_episodes:03d} | "
                f"Avg Rwd: {avg_reward:6.1f} | "
                f"Avg Pellets: {avg_pellets:5.1f} ({avg_pct:4.1f}%) | "
                f"Max Pellets: {max_pellets:3d} ({max_pct:4.1f}%) | "
                f"Loss (P/V): {avg_policy_loss:.4f}/{avg_value_loss:.4f} | "
                f"Time: {total_elapsed:5.1f}s ({update_elapsed:4.2f}s/upd)"
            )

        if update % save_interval == 0 or update == num_updates:
            torch.save(policy.state_dict(), checkpoint_path)

    print(f"============================================================")
    print(f"Stage {stage} Training Completed in {total_elapsed:.1f}s!")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"============================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Stage 1 Trainer for Pac-Man Player Model")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage (1=Pellets, 2=Static Ghosts, 3=BFS Ghosts)")
    parser.add_argument("--num-updates", type=int, default=100, help="Number of PPO update iterations")
    parser.add_argument("--rollout-steps", type=int, default=512, help="Steps per rollout")
    parser.add_argument("--save-interval", type=int, default=10, help="Snapshot save interval")
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR), help="Model checkpoint directory")
    parser.add_argument("--fresh", action="store_true", help="Start training from fresh initial weights instead of resuming checkpoint")
    parser.add_argument("--resume-path", type=str, default="", help="Custom checkpoint path to resume from")
    args = parser.parse_args()

    train_player_ppo(
        stage=args.stage,
        num_updates=args.num_updates,
        rollout_steps=args.rollout_steps,
        save_interval=args.save_interval,
        model_dir=Path(args.model_dir),
        resume=not args.fresh,
        resume_path=Path(args.resume_path) if args.resume_path else None,
    )


if __name__ == "__main__":
    main()
