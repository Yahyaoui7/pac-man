"""High-performance PPO training pipeline for Pac-Man player model with continuous checkpoint resuming."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from AI_arena.models.cnn_player import PlayerActorCritic, load_sl_weights_into_ppo
from AI_arena.player.player_env import PacmanPlayerEnv

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"

from AI_arena.player.utils import (
    BD_LABELS,
    QuitListener,
    TrainingLogger,
    format_breakdown_line,
)


def train_player_ppo(
    stage: int = 1,
    num_updates: int = 100,
    rollout_steps: int = 4096,
    ppo_epochs: int = 2,
    minibatch_size: int = 64,
    learning_rate: float = 1e-5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.1,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    model_dir: Path = DEFAULT_MODEL_DIR,
    save_interval: int = 10,
    seed: int = 42,
    resume: bool = True,
    resume_path: Path | None = None,
    log_file: Path = Path("training_log.txt"),
    sl_warmstart: bool = True,
    quiet: bool = False,
) -> None:
    """Train Pac-Man player model using PPO with continuous checkpoint resuming."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    logger = TrainingLogger(log_file, quiet)

    logger.log(f"============================================================")
    logger.log(f"Starting Stage {stage} PPO Training for Pac-Man")
    logger.log(
        f"Device: {device} | AMP: {device.type == 'cuda'} | Total Updates: {num_updates} | Rollout Steps: {rollout_steps}"
    )
    logger.log(f"============================================================")

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"player_rl_stage{stage}.pt"

    env = PacmanPlayerEnv(seed=seed, stage=stage, device="cpu")
    policy = PlayerActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    loaded_checkpoint = False
    sl_best_path = model_dir / "player_sl_best.pt"
    loaded_checkpoint = False
    if resume and not sl_warmstart:
        target_load = resume_path or checkpoint_path
        if not target_load.exists() and stage > 1:
            stage1_best = model_dir / f"player_rl_stage{stage-1}_best.pt"
            stage1_last = model_dir / f"player_rl_stage{stage-1}.pt"
            if stage1_best.exists():
                target_load = stage1_best
            elif stage1_last.exists():
                target_load = stage1_last

        if target_load.exists():
            weights = torch.load(target_load, map_location=device, weights_only=True)
            policy.load_state_dict(weights)
            loaded_checkpoint = True
            logger.log(f"SUCCESS: Resumed training from checkpoint: {target_load.name}")

    ref_policy: PlayerActorCritic | None = None
    if not loaded_checkpoint:
        if sl_warmstart and sl_best_path.exists():
            load_sl_weights_into_ppo(policy, str(sl_best_path), device=device)
            logger.log(f"SUCCESS: Warm-started PPO policy with pre-trained SL weights ({sl_best_path.name})")

            # Freeze CNN backbone to preserve expert spatial representation
            for p in policy.backbone.parameters():
                p.requires_grad = False
            logger.log("INFO: Frozen CNN backbone parameters to prevent spatial policy degradation.")

            # Create frozen reference policy for KL anchoring
            ref_policy = PlayerActorCritic().to(device)
            load_sl_weights_into_ppo(ref_policy, str(sl_best_path), device=device)
            ref_policy.eval()
            for p in ref_policy.parameters():
                p.requires_grad = False
        else:
            logger.log("Starting training with fresh random initial weights.")

    quit_listener = QuitListener()
    quit_listener.start()

    start_time = time.time()
    obs = env.reset()

    recent_episodes: deque[dict[str, Any]] = deque(maxlen=100)
    current_ep_reward = 0.0
    current_ep_steps = 0
    total_completed_episodes = 0
    last_update_completed = 0

    if sl_warmstart and sl_best_path.exists() and not loaded_checkpoint:
        learning_rate = 5e-5
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=learning_rate)
        logger.log(f"INFO: Adjusted learning rate to {learning_rate} for PPO head fine-tuning.")

    best_checkpoint_path = model_dir / f"player_rl_stage{stage}_best.pt"
    best_avg_pct: float = 0.0
    best_avg_pellets: float = 0.0

    try:
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
            save_window_episodes: list[dict[str, float]] = []

            for _ in range(rollout_steps):
                grid, features, valid_actions = obs

                with torch.no_grad():
                    logits, value = policy(grid.to(device), features.to(device))
                    masked_logits = logits.masked_fill(~valid_actions.to(device), -1e4)
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
                    ep_steps = max(1.0, float(current_ep_steps))
                    osc_cnt = float(info["episode_event_counts"].get("osc", 0))
                    ep_record = {
                        "reward": current_ep_reward,
                        "pellets": float(info["pellets_eaten"]),
                        "pct": float(info["completion_pct"]),
                        "steps": ep_steps,
                        "osc_count": osc_cnt,
                        "osc_pct": (osc_cnt / ep_steps) * 100.0,
                        "maze": info["maze"],
                        "episode_event_counts": info["episode_event_counts"],
                        "episode_reward_breakdown": info["episode_reward_breakdown"],
                    }
                    current_ep_reward = 0.0
                    current_ep_steps = 0
                    completed_episodes_in_update += 1
                    total_completed_episodes += 1
                    recent_episodes.append(ep_record)
                    save_window_episodes.append(ep_record)
                else:
                    obs = next_obs

            # Bulk Transfer Rollout Data to GPU & GAE Calculation
            with torch.no_grad():
                last_grid, last_features, _ = obs
                _, next_value = policy(last_grid.to(device), last_features.to(device))
                next_value = next_value.squeeze(-1)

            b_grids = torch.cat(rollout_grids, dim=0).to(device, non_blocking=True)
            b_features = torch.cat(rollout_features, dim=0).to(device, non_blocking=True)
            b_valid_actions = torch.cat(rollout_valid_actions, dim=0).to(device, non_blocking=True)
            b_actions = torch.cat(rollout_actions, dim=0).to(device, non_blocking=True)
            b_log_probs = torch.cat(rollout_log_probs, dim=0).to(device, non_blocking=True)
            b_rewards = torch.cat(rollout_rewards, dim=0).to(device, non_blocking=True)
            b_dones = torch.cat(rollout_dones, dim=0).to(device, non_blocking=True)
            b_values = torch.cat(rollout_values, dim=0).to(device, non_blocking=True)

            advantages = torch.zeros_like(b_rewards, device=device)
            last_gae_lam = 0.0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    next_non_terminal = 1.0 - b_dones[t]
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - b_dones[t]
                    next_val = b_values[t + 1]

                delta = (
                    b_rewards[t] + gamma * next_val * next_non_terminal - b_values[t]
                )
                advantages[t] = last_gae_lam = (
                    delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                )

            returns = advantages + b_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO GPU Optimization Epochs
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy_loss = 0.0
            dataset_size = rollout_steps

            kl_coef = 0.20 if ref_policy is not None else 0.0
            eff_entropy_coef = 0.001 if ref_policy is not None else entropy_coef

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

                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        logits, values = policy(mb_grid, mb_features)
                        masked_logits = logits.masked_fill(~mb_valid_actions, -1e4)
                        dist = Categorical(logits=masked_logits)

                        new_log_probs = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()

                        log_ratio = new_log_probs - mb_old_log_probs
                        ratio = torch.exp(log_ratio)

                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                        policy_loss = -torch.min(surr1, surr2).mean()

                        value_loss = F.mse_loss(values.squeeze(-1), mb_returns)

                        kl_loss = torch.tensor(0.0, device=device)
                        if ref_policy is not None:
                            with torch.no_grad():
                                ref_logits, _ = ref_policy(mb_grid, mb_features)
                                ref_masked_logits = ref_logits.masked_fill(~mb_valid_actions, -1e4)
                                ref_probs = F.softmax(ref_masked_logits, dim=-1)
                                ref_log_probs = F.log_softmax(ref_masked_logits, dim=-1)
                            log_probs_dist = F.log_softmax(masked_logits, dim=-1)
                            kl_loss = (ref_probs * (ref_log_probs - log_probs_dist)).sum(dim=-1).mean()

                        loss = (
                            policy_loss + value_coef * value_loss - eff_entropy_coef * entropy + kl_coef * kl_loss
                        )

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy.item()

            update_elapsed = time.time() - update_start_time
            total_elapsed = time.time() - start_time
            last_update_completed = update

            if recent_episodes:
                avg_reward = sum(ep["reward"] for ep in recent_episodes) / len(
                    recent_episodes
                )
                avg_pellets = sum(ep["pellets"] for ep in recent_episodes) / len(
                    recent_episodes
                )

                avg_pct = sum(ep["pct"] for ep in recent_episodes) / len(
                    recent_episodes
                )
            else:
                avg_reward = current_ep_reward
                avg_pellets = float(info.get("pellets_eaten", 0))
                avg_pct = float(info.get("completion_pct", 0.0))

            avg_policy_loss = total_policy_loss / (ppo_epochs * dataset_size)
            avg_value_loss = total_value_loss / (ppo_epochs * dataset_size)

            # Save best checkpoint whenever avg_pct improves
            is_best = bool(recent_episodes and avg_pct > best_avg_pct)
            if is_best:
                best_avg_pct = avg_pct
                best_avg_pellets = avg_pellets
                torch.save(policy.state_dict(), best_checkpoint_path)

            if save_window_episodes:
                window_max_pct = max(ep["pct"] for ep in save_window_episodes)
                max_pellets = int(max(ep["pellets"] for ep in save_window_episodes))
                epoch_avg_reward = sum(
                    ep["reward"] for ep in save_window_episodes
                ) / len(save_window_episodes)
                avg_area = sum(
                    ep["maze"][0] * ep["maze"][1] for ep in save_window_episodes
                ) / len(save_window_episodes)
                avg_w = sum(ep["maze"][0] for ep in save_window_episodes) / len(
                    save_window_episodes
                )
                avg_h = sum(ep["maze"][1] for ep in save_window_episodes) / len(
                    save_window_episodes
                )
                completion_rate = sum(
                    ep["episode_event_counts"].get("completed", 0) > 0
                    for ep in save_window_episodes
                ) / len(save_window_episodes)
                truncation_rate = sum(
                    ep["episode_event_counts"].get("truncated", 0) > 0
                    for ep in save_window_episodes
                ) / len(save_window_episodes)
                avg_osc_pct = sum(
                    ep.get("osc_pct", 0.0) for ep in save_window_episodes
                ) / len(save_window_episodes)
            else:
                window_max_pct = 0.0
                max_pellets = 0
                epoch_avg_reward = 0.0
                avg_area = 0.0
                avg_w = 0.0
                avg_h = 0.0
                completion_rate = 0.0
                truncation_rate = 0.0
                avg_osc_pct = 0.0

            # Build the reward-breakdown chunk (averaged over the 100-ep window)
            breakdown_line = format_breakdown_line(save_window_episodes)

            if update % 1 == 0 or update == 1 or update == num_updates:
                logger.log(
                    f"Upd {update:03d}/{num_updates:03d} | "
                    f"Tot Ep: {total_completed_episodes:03d} | "
                    f"Averge Epoch Rwd: {epoch_avg_reward:6.1f} | "
                    f"Max Epoch Pellets: {max_pellets:3d} ({window_max_pct:4.1f}%) | "
                    f"Avg Pellets: {avg_pellets:5.1f} ({avg_pct:4.1f}%) | "
                    f"Osc%: {avg_osc_pct:4.1f}% | "
                    f"Avg Rwd: {avg_reward:4.1f} | "
                    f"{breakdown_line} | "
                    f"Loss (P/V): {avg_policy_loss:.4f}/{avg_value_loss:.4f} | "
                    f"Time: {total_elapsed:5.1f}s ({update_elapsed:4.2f}s/upd)"
                    f" | Complete: {completion_rate:5.1%}"
                    f" | Truncated: {truncation_rate:5.1%}"
                    f" | Avg Maze Area: {avg_area:.1f} ({avg_w:.1f}x{avg_h:.1f})"
                )

            if update % save_interval == 0 or update == num_updates:
                torch.save(policy.state_dict(), checkpoint_path)
                save_window_episodes = []

            if quit_listener.stop_requested:
                logger.log(
                    f"\n'q' pressed — stopping after update {update}/{num_updates}."
                )
                torch.save(policy.state_dict(), checkpoint_path)
                logger.log(f"Checkpoint saved to: {checkpoint_path}")
                logger.log(
                    f"Best checkpoint (avg {best_avg_pct:.1f}%): {best_checkpoint_path}"
                )
                break
    except KeyboardInterrupt:
        # Also handle Ctrl+C gracefully with a final save.
        logger.log(
            f"\nKeyboardInterrupt — saving checkpoint at update {last_update_completed}."
        )
        torch.save(policy.state_dict(), checkpoint_path)
        logger.log(f"Checkpoint saved to: {checkpoint_path}")
        logger.log(
            f"Best checkpoint (avg {best_avg_pct:.1f}% | {best_avg_pellets:.0f} pellets): {best_checkpoint_path}"
        )
    finally:
        logger.log(f"============================================================")
        logger.log(
            f"Stage {stage} Training Stopped/Completed after {last_update_completed} update(s), "
            f"{time.time() - start_time:.1f}s!"
        )
        logger.log(f"Checkpoint Path: {checkpoint_path}")
        logger.log(
            f"Best Checkpoint: {best_checkpoint_path} (peak avg_pct={best_avg_pct:.1f}%)"
        )
        logger.log(f"============================================================")
        quit_listener.stop()
        logger.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PPO Stage 1 Trainer for Pac-Man Player Model"
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        help="Curriculum stage (1=Pellets, 2=Static Ghosts, 3=BFS Ghosts)",
    )
    parser.add_argument(
        "--num-updates", type=int, default=100, help="Number of PPO update iterations"
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=2048, help="Steps per rollout"
    )
    parser.add_argument(
        "--save-interval", type=int, default=2, help="Snapshot save interval"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Model checkpoint directory",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start training from fresh initial weights instead of resuming checkpoint",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default="",
        help="Custom checkpoint path to resume from",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="training_log.txt",
        help="Path to training log file (append mode)",
    )
    parser.add_argument(
        "--no-sl-warmstart",
        action="store_true",
        help="Disable warm-starting PPO policy from Supervised Learning weights",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout output (log to file only)",
    )
    args = parser.parse_args()

    train_player_ppo(
        stage=args.stage,
        num_updates=args.num_updates,
        rollout_steps=args.rollout_steps,
        save_interval=args.save_interval,
        model_dir=Path(args.model_dir),
        resume=not args.fresh,
        resume_path=Path(args.resume_path) if args.resume_path else None,
        log_file=Path(args.log_file),
        sl_warmstart=not args.no_sl_warmstart,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
