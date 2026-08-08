"""Clean PPO training pipeline for Pac-Man player with GRU memory."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from AI_arena.models.cnn_player import PlayerActorCritic, load_sl_weights_into_ppo
from AI_arena.player.player_env import PacmanPlayerEnv
from AI_arena.player.utils import (
    QuitListener,
    TrainingLogger,
    format_breakdown_line,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit these by hand
# ═══════════════════════════════════════════════════════════════════════════════

STAGE = 1
NUM_UPDATES = 1000
ROLLOUT_STEPS = 5000
SEQ_LEN = 16  # Temporal sequence chunk length for GRU BPTT
NUM_SEQUENCES = ROLLOUT_STEPS // SEQ_LEN  # 128 sequence chunks per rollout
MINIBATCH_SEQS = 4  # 4 sequences per minibatch (64 total frames)
PPO_EPOCHS = 2
MINIBATCH_SIZE = 64

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

SEED = 42
SAVE_INTERVAL = 50
RESUME = True  # set True to resume from checkpoint
SL_WARMSTART = False  # set True to warm-start from supervised-learning weights

MODEL_DIR = Path(__file__).parent.parent / "models"
LOG_FILE = Path("training_log.txt")

# ═══════════════════════════════════════════════════════════════════════════════


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae
        )

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


def train() -> None:
    """Main PPO training loop with GRU memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    logger = TrainingLogger(LOG_FILE, quiet=False)

    logger.log("=" * 60)
    logger.log(f"Stage {STAGE} PPO Training | Device: {device}")
    logger.log(f"Updates: {NUM_UPDATES} | Rollout: {ROLLOUT_STEPS}")
    logger.log("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODEL_DIR / f"player_rl_stage{STAGE}.pt"
    best_checkpoint_path = MODEL_DIR / f"player_rl_stage{STAGE}_best.pt"

    env = PacmanPlayerEnv(seed=SEED, stage=STAGE, device="cpu")
    policy = PlayerActorCritic().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    loaded_checkpoint = False
    sl_best_path = MODEL_DIR / "player_sl_best.pt"

    if RESUME and not SL_WARMSTART:
        target = checkpoint_path
        if not target.exists() and STAGE > 1:
            stage1_best = MODEL_DIR / f"player_rl_stage{STAGE - 1}_best.pt"
            stage1_last = MODEL_DIR / f"player_rl_stage{STAGE - 1}.pt"
            if stage1_best.exists():
                target = stage1_best
            elif stage1_last.exists():
                target = stage1_last

        if target.exists():
            weights = torch.load(target, map_location=device, weights_only=True)
            policy.load_state_dict(weights)
            loaded_checkpoint = True
            logger.log(f"Resumed from {target.name}")

    ref_policy: PlayerActorCritic | None = None
    if not loaded_checkpoint:
        if SL_WARMSTART and sl_best_path.exists():
            load_sl_weights_into_ppo(policy, str(sl_best_path), device=device)
            logger.log(f"Warm-started from {sl_best_path.name}")

            for p in policy.backbone.parameters():
                p.requires_grad = False
            logger.log("Frozen CNN backbone (SL pre-trained).")

            ref_policy = PlayerActorCritic().to(device)
            load_sl_weights_into_ppo(ref_policy, str(sl_best_path), device=device)
            ref_policy.eval()
            for p in ref_policy.parameters():
                p.requires_grad = False
        else:
            logger.log("Starting from fresh random weights.")

    if SL_WARMSTART and sl_best_path.exists() and not loaded_checkpoint:
        optimizer = torch.optim.Adam(
            [p for p in policy.parameters() if p.requires_grad],
            lr=5e-5,
        )
        logger.log("LR set to 5e-5 for head fine-tuning.")

    quit_listener = QuitListener()
    quit_listener.start()

    start_time = time.time()
    obs = env.reset()

    recent_episodes: deque[dict[str, Any]] = deque(maxlen=100)
    current_ep_reward = 0.0
    current_ep_steps = 0
    total_completed_episodes = 0
    last_update_completed = 0

    best_avg_pct = 0.0
    best_avg_pellets = 0.0

    try:
        for update in range(1, NUM_UPDATES + 1):
            update_start = time.time()

            # ── Rollout collection ──
            rollout_grids: list[torch.Tensor] = []
            rollout_features: list[torch.Tensor] = []
            rollout_valid_actions: list[torch.Tensor] = []
            rollout_actions: list[torch.Tensor] = []
            rollout_log_probs: list[torch.Tensor] = []
            rollout_rewards: list[torch.Tensor] = []
            rollout_dones: list[torch.Tensor] = []
            rollout_values: list[torch.Tensor] = []
            rollout_seq_hiddens: list[torch.Tensor] = []

            save_window_episodes: list[dict[str, float]] = []
            policy_hidden: torch.Tensor | None = None

            for step in range(ROLLOUT_STEPS):
                grid, features, valid_actions = obs

                # Record hidden state at start of each sequence chunk
                if step % SEQ_LEN == 0:
                    rollout_seq_hiddens.append(
                        policy_hidden.view(128).cpu()
                        if policy_hidden is not None
                        else torch.zeros(128)
                    )

                with torch.no_grad():
                    logits, value, policy_hidden = policy(
                        grid.to(device),
                        features.to(device),
                        policy_hidden,
                    )
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
                    policy_hidden = None

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
                    total_completed_episodes += 1
                    recent_episodes.append(ep_record)
                    save_window_episodes.append(ep_record)
                else:
                    obs = next_obs

            # ── Bootstrap value for GAE ──
            with torch.no_grad():
                last_grid, last_features, _ = obs
                _, next_value, _ = policy(
                    last_grid.to(device), last_features.to(device), None
                )
                next_value = next_value.squeeze(-1)

            # ── Truncate tensors to match full sequence chunks ──
            num_seq_steps = NUM_SEQUENCES * SEQ_LEN
            b_grids = torch.cat(rollout_grids, dim=0)[:num_seq_steps].to(device)
            b_features = torch.cat(rollout_features, dim=0)[:num_seq_steps].to(device)
            b_valid_actions = torch.cat(rollout_valid_actions, dim=0)[:num_seq_steps].to(device)
            b_actions = torch.cat(rollout_actions, dim=0)[:num_seq_steps].to(device)
            b_log_probs = torch.cat(rollout_log_probs, dim=0)[:num_seq_steps].to(device)
            b_rewards = torch.cat(rollout_rewards, dim=0)[:num_seq_steps].to(device)
            b_dones = torch.cat(rollout_dones, dim=0)[:num_seq_steps].to(device)
            b_values = torch.cat(rollout_values, dim=0)[:num_seq_steps].to(device)
            b_seq_hiddens = torch.stack(rollout_seq_hiddens, dim=0)[:NUM_SEQUENCES].to(device)  # (NUM_SEQUENCES, 128)

            # ── GAE ──
            advantages, returns = compute_gae(
                b_rewards, b_values, b_dones, next_value, GAMMA, GAE_LAMBDA
            )

            # ── Reshape into Sequence Chunks for BPTT ──
            b_grids_seq = b_grids.view(NUM_SEQUENCES, SEQ_LEN, *b_grids.shape[1:])
            b_features_seq = b_features.view(NUM_SEQUENCES, SEQ_LEN, *b_features.shape[1:])
            b_valid_actions_seq = b_valid_actions.view(NUM_SEQUENCES, SEQ_LEN, *b_valid_actions.shape[1:])
            b_actions_seq = b_actions.view(NUM_SEQUENCES, SEQ_LEN)
            b_log_probs_seq = b_log_probs.view(NUM_SEQUENCES, SEQ_LEN)
            advantages_seq = advantages.view(NUM_SEQUENCES, SEQ_LEN)
            returns_seq = returns.view(NUM_SEQUENCES, SEQ_LEN)

            # ── PPO update epochs (BPTT across sequence chunks) ──
            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy_loss = 0.0
            num_minibatches = 0

            kl_coef = 0.20 if ref_policy is not None else 0.0
            eff_entropy_coef = 0.001 if ref_policy is not None else ENTROPY_COEF

            for _ in range(PPO_EPOCHS):
                seq_perm = torch.randperm(NUM_SEQUENCES, device=device)
                for start in range(0, NUM_SEQUENCES, MINIBATCH_SEQS):
                    mb_seq_idx = seq_perm[start : start + MINIBATCH_SEQS]

                    mb_grid = b_grids_seq[mb_seq_idx]
                    mb_features = b_features_seq[mb_seq_idx]
                    mb_valid = b_valid_actions_seq[mb_seq_idx]
                    mb_actions = b_actions_seq[mb_seq_idx]
                    mb_old_log_probs = b_log_probs_seq[mb_seq_idx]
                    mb_adv = advantages_seq[mb_seq_idx]
                    mb_returns = returns_seq[mb_seq_idx]

                    mb_hidden = b_seq_hiddens[mb_seq_idx].unsqueeze(0)  # (1, MINIBATCH_SEQS, 128)

                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        logits, values, _ = policy(mb_grid, mb_features, mb_hidden)
                        masked_logits = logits.masked_fill(~mb_valid, -1e4)
                        dist = Categorical(logits=masked_logits)

                        new_log_probs = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()

                        ratio = torch.exp(new_log_probs - mb_old_log_probs)
                        surr1 = ratio * mb_adv
                        surr2 = (
                            torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
                        )
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.mse_loss(values.squeeze(-1), mb_returns)

                        kl_loss = torch.tensor(0.0, device=device)
                        if ref_policy is not None:
                            with torch.no_grad():
                                ref_logits, _, _ = ref_policy(
                                    mb_grid, mb_features, mb_hidden
                                )
                                ref_masked = ref_logits.masked_fill(~mb_valid, -1e4)
                                ref_probs = F.softmax(ref_masked, dim=-1)
                                ref_log_p = F.log_softmax(ref_masked, dim=-1)
                            log_p = F.log_softmax(masked_logits, dim=-1)
                            kl_loss = (
                                (ref_probs * (ref_log_p - log_p)).sum(dim=-1).mean()
                            )

                        loss = (
                            policy_loss
                            + VALUE_COEF * value_loss
                            - eff_entropy_coef * entropy
                            + kl_coef * kl_loss
                        )

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()

                    total_policy_loss += policy_loss.item()
                    total_value_loss += value_loss.item()
                    total_entropy_loss += entropy.item()
                    num_minibatches += 1

            # ── Logging ──
            update_elapsed = time.time() - update_start
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
                avg_pellets = 0.0
                avg_pct = 0.0

            avg_policy_loss = total_policy_loss / max(1, num_minibatches)
            avg_value_loss = total_value_loss / max(1, num_minibatches)

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

            breakdown_line = format_breakdown_line(save_window_episodes)

            logger.log(
                f"Upd {update:03d}/{NUM_UPDATES:03d} | "
                f"Tot Ep: {total_completed_episodes:03d} | "
                f"Avg Epoch Rwd: {epoch_avg_reward:6.1f} | "
                f"Max Pellets: {max_pellets:3d} ({window_max_pct:4.1f}%) | "
                f"Avg Pellets: {avg_pellets:5.1f} ({avg_pct:4.1f}%) | "
                f"Osc%: {avg_osc_pct:4.1f}% | "
                f"Avg Rwd: {avg_reward:4.1f} | "
                f"{breakdown_line} | "
                f"Loss (P/V): {avg_policy_loss:.4f}/{avg_value_loss:.4f} | "
                f"Time: {total_elapsed:5.1f}s ({update_elapsed:4.2f}s/upd)"
                f" | Complete: {completion_rate:5.1%}"
                f" | Truncated: {truncation_rate:5.1%}"
                f" | Avg Maze: {avg_area:.1f} ({avg_w:.1f}x{avg_h:.1f})"
            )

            death_rate = (
                sum(
                    ep["episode_event_counts"].get("died", 0) > 0
                    for ep in save_window_episodes
                )
                / max(1, len(save_window_episodes))
                * 100
                if save_window_episodes
                else 0.0
            )

            if update % SAVE_INTERVAL == 0 or update == NUM_UPDATES:
                torch.save(policy.state_dict(), checkpoint_path)
                save_window_episodes = []

            if quit_listener.stop_requested:
                logger.log(f"\n'q' pressed — stopping at update {update}.")
                torch.save(policy.state_dict(), checkpoint_path)
                logger.log(f"Checkpoint: {checkpoint_path}")
                logger.log(
                    f"Best ({best_avg_pct:.1f}%): {best_checkpoint_path}"
                    f" | Death rate: {death_rate:.1f}%"
                )
                break

    except KeyboardInterrupt:
        logger.log(f"\nKeyboardInterrupt — saving at update {last_update_completed}.")
        torch.save(policy.state_dict(), checkpoint_path)
        logger.log(f"Checkpoint: {checkpoint_path}")
        logger.log(
            f"Best ({best_avg_pct:.1f}% | {best_avg_pellets:.0f} pellets):"
            f" {best_checkpoint_path}"
        )

    finally:
        logger.log("=" * 60)
        logger.log(
            f"Stage {STAGE} stopped after {last_update_completed} updates, "
            f"{time.time() - start_time:.1f}s"
        )
        logger.log(f"Checkpoint: {checkpoint_path}")
        logger.log(f"Best: {best_checkpoint_path} (peak {best_avg_pct:.1f}%)")
        logger.log("=" * 60)
        quit_listener.stop()
        logger.close()


if __name__ == "__main__":
    train()
