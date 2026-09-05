"""Clean PPO training pipeline for Pac-Man player with GRU memory.

Refactored from a 735-line `train()` function into a class-based structure:
  - TrainingConfig  : dataclass holding all hyperparameters + argparse loader
  - RolloutBuffer   : bundles the 11 rollout tensor lists into one object
  - PacmanTrainer   : orchestrates setup, rollout, PPO update, eval, checkpointing

Each method is ≤50 lines and has one responsibility, so changing any single
piece (reward logging, eval cadence, optimizer setup, …) no longer requires
reading the whole training loop.

Usage examples
--------------
    # defaults — resume from latest checkpoint, 1000 updates
    python player_training.py

    # fresh start, shorter rollouts for faster iteration
    python player_training.py --fresh --rollout-steps 1500 --updates 500

    # tune entropy / lr without editing the file
    python player_training.py --lr 5e-5 --entropy 0.05

    # graduate curriculum and disable BFS shaping
    python player_training.py --start-pellets 8,12,16 --no-bfs-shaping

    # run on CPU only
    python player_training.py --device cpu

    # tag the run (separates log files)
    python player_training.py --name entropy_bump
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from AI_arena.models.cnn_player import (
    PlayerActorCritic,
    load_checkpoint_into_policy,
    load_sl_weights_into_ppo,
)
from AI_arena.player.player_env import PacmanPlayerEnv
from AI_arena.player.utils import (
    QuitListener,
    TrainingLogger,
    compute_survival_stats,
    format_breakdown_line,
    format_survival_line,
)
from AI_arena.player.utils.evaluate import append_history, run_evaluation

# ═══════════════════════════════════════════════════════════════════════════════
# GAE — kept at module level (pure function, no state)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    terminated: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation with proper truncation bootstrapping."""
    n_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_non_terminal = 1.0 - terminated[t]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - terminated[t]
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        next_non_done = 1.0 - dones[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_done * last_gae

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_pellets(s: str) -> tuple[int, ...] | None:
    """Parse --start-pellets: 'none' / 'full' → None, '3,5,8' → (3,5,8)."""
    s = s.strip().lower()
    if s in ("none", "full", ""):
        return None
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


@dataclass
class TrainingConfig:
    """All training hyperparameters. Edit defaults here, override via CLI."""

    # ── Core loop ──
    stage: int = 2
    num_updates: int = 1500
    rollout_steps: int = 3000
    seq_len: int = 32
    minibatch_seqs: int = 4
    ppo_epochs: int = 4

    # ── Optimization ──
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.005
    max_grad_norm: float = 0.5
    rollout_epsilon: float = 0.0

    # ── Execution ──
    seed: int = 42
    save_interval: int = 50
    resume: bool = True
    sl_warmstart: bool = False
    device: str | None = None  # None → auto (cuda if available)
    run_name: str = ""  # tag for log file naming

    # ── Search Guidance & Distillation ──
    search_guided: bool = False
    search_horizon: int = 12
    search_alpha: float = 0.85
    distill_coef: float = 0.5

    # ── Curriculum ──
    start_pellets: tuple[int, ...] | None = (3, 5, 8)
    use_bfs_shaping: bool = True
    ghost_speed_ratio: float = 0.35
    ghost_confusion_prob: float = 0.0

    # ── Evaluation ──
    eval_episodes: int = 20
    eval_seed_base: int = 10000
    eval_device: str = "cpu"
    eval_stall_patience: int = 5
    eval_min_improvement: float = 2.0

    # ── Auto-curriculum ──
    auto_curriculum: bool = True
    stage1_grad_threshold: float = 0.80  # eval completion rate to graduate stage 1 → 2
    stage1_grad_evals: int = 2  # consecutive evals above threshold to graduate

    # ── Paths ──
    model_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "models"
    )
    log_file: Path = field(default_factory=lambda: Path("training_log.txt"))

    # ── Derived (not set by user) ──
    @property
    def num_sequences(self) -> int:
        return self.rollout_steps // self.seq_len

    @property
    def minibatch_size(self) -> int:
        """Total frames per minibatch = seqs × seq_len."""
        return self.minibatch_seqs * self.seq_len

    @property
    def eval_interval(self) -> int:
        """Eval cadence — defaults to save_interval."""
        return self.save_interval

    # ── Argparse ──
    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> "TrainingConfig":
        """Build config from command-line args, falling back to dataclass defaults.

        Any CLI flag that isn't passed stays None, and we let the dataclass
        default take over (Path(__file__)-relative for model_dir, etc.).
        """
        parser = cls._build_parser()
        args = parser.parse_args(argv)

        # Start from dataclass defaults, then override only what was specified.
        # This avoids Path(None) crashes when optional path flags are skipped.
        kwargs: dict[str, Any] = dict(
            stage=args.stage,
            num_updates=args.updates,
            rollout_steps=args.rollout_steps,
            seq_len=args.seq_len,
            minibatch_seqs=args.minibatch_seqs,
            ppo_epochs=args.ppo_epochs,
            learning_rate=args.lr,
            entropy_coef=args.entropy,
            value_coef=args.value_coef,
            rollout_epsilon=args.epsilon,
            seed=args.seed,
            save_interval=args.save_interval,
            resume=not args.fresh,
            sl_warmstart=args.sl_warmstart,
            device=args.device,
            run_name=args.name,
            start_pellets=_parse_pellets(args.start_pellets),
            use_bfs_shaping=not args.no_bfs_shaping,
            ghost_speed_ratio=args.ghost_speed_ratio,
            ghost_confusion_prob=args.ghost_confusion,
            eval_episodes=args.eval_episodes,
            auto_curriculum=args.auto_curriculum,
            stage1_grad_threshold=args.grad_threshold,
            stage1_grad_evals=args.grad_evals,
            search_guided=args.search_guided,
            search_horizon=args.search_horizon,
            search_alpha=args.search_alpha,
            distill_coef=args.distill_coef,
        )

        # Paths: only override if the user explicitly passed them.
        if args.model_dir is not None:
            kwargs["model_dir"] = Path(args.model_dir)

        # Log file: explicit > name-derived > dataclass default.
        if args.log_file is not None:
            kwargs["log_file"] = Path(args.log_file)
        elif args.name:
            kwargs["log_file"] = Path(f"training_log_{args.name}.txt")

        return cls(**kwargs)

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            description="PPO trainer for Pac-Man with GRU memory.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # ── Core loop ──
        p.add_argument(
            "--stage",
            type=int,
            default=2,
            help="training stage (1=ghost-free, 2=full ghosts)",
        )
        p.add_argument(
            "--updates", type=int, default=1000, help="number of PPO updates"
        )
        p.add_argument(
            "--rollout-steps", type=int, default=3000, help="env steps per rollout"
        )
        p.add_argument("--seq-len", type=int, default=32, help="BPTT sequence length")
        p.add_argument(
            "--minibatch-seqs", type=int, default=4, help="sequences per minibatch"
        )
        p.add_argument(
            "--ppo-epochs", type=int, default=4, help="PPO epochs per rollout"
        )

        # ── Optimization ──
        p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
        p.add_argument(
            "--entropy", type=float, default=0.01, help="entropy coefficient"
        )
        p.add_argument(
            "--value-coef", type=float, default=0.05, help="value loss coefficient"
        )
        p.add_argument(
            "--epsilon", type=float, default=0.0, help="rollout ε-exploration"
        )

        # ── Execution ──
        p.add_argument("--seed", type=int, default=42, help="env seed")
        p.add_argument(
            "--save-interval", type=int, default=50, help="save + eval cadence"
        )
        p.add_argument(
            "--fresh",
            action="store_true",
            help="start from random weights (skip resume)",
        )
        p.add_argument(
            "--sl-warmstart", action="store_true", help="warmstart from SL checkpoint"
        )
        p.add_argument(
            "--device",
            choices=["cuda", "cpu"],
            default=None,
            help="override device (default: auto)",
        )
        p.add_argument("--name", type=str, default="", help="tag for log file naming")

        # ── Curriculum ──
        p.add_argument(
            "--start-pellets",
            type=str,
            default="3,5,8",
            help="pellet counts per episode: '3,5,8' or 'none' for full map",
        )
        p.add_argument(
            "--no-bfs-shaping",
            action="store_true",
            help="disable BFS potential shaping",
        )
        p.add_argument(
            "--ghost-speed-ratio",
            type=float,
            default=0.50,
            help="ghost speed ratio relative to player (default: 0.50)",
        )
        p.add_argument(
            "--ghost-confusion",
            type=float,
            default=0.0,
            help="ghost movement confusion probability (default: 0.0)",
        )

        # ── Evaluation ──
        p.add_argument(
            "--eval-episodes", type=int, default=20, help="episodes per eval run"
        )

        # ── Auto-curriculum ──
        p.add_argument(
            "--no-auto-curriculum",
            action="store_false",
            dest="auto_curriculum",
            help="disable automatic stage 1 → 2 graduation",
        )
        p.set_defaults(auto_curriculum=True)
        p.add_argument(
            "--grad-threshold",
            type=float,
            default=0.80,
            help="eval completion rate needed to graduate stage 1 → 2 (default: 0.80)",
        )
        p.add_argument(
            "--grad-evals",
            type=int,
            default=2,
            help="consecutive evals above threshold required to graduate (default: 2)",
        )

        # ── Search Guidance & Distillation ──
        p.add_argument(
            "--search-guided",
            action="store_true",
            help="enable Chess-like lookahead search guidance during rollout collection",
        )
        p.add_argument(
            "--search-horizon",
            type=int,
            default=12,
            help="lookahead search depth in steps (default: 12)",
        )
        p.add_argument(
            "--search-alpha",
            type=float,
            default=0.85,
            help="probability of taking the search action during rollout (default: 0.85)",
        )
        p.add_argument(
            "--distill-coef",
            type=float,
            default=0.5,
            help="auxiliary distillation cross-entropy loss coefficient (default: 0.5)",
        )

        # ── Paths ──
        p.add_argument(
            "--model-dir", type=str, default=None, help="checkpoint directory"
        )
        p.add_argument("--log-file", type=str, default=None, help="log file path")

        return p

    def summary(self) -> str:
        """Human-readable summary for logging at startup."""
        pellets_str = (
            "full map"
            if self.start_pellets is None
            else ",".join(map(str, self.start_pellets))
        )
        lines = [
            f"  Stage            : {self.stage}",
            f"  Updates          : {self.num_updates}",
            f"  Rollout steps    : {self.rollout_steps}  ({self.num_sequences} seqs × {self.seq_len})",
            f"  Minibatch        : {self.minibatch_seqs} seqs  ({self.minibatch_size} frames)",
            f"  PPO epochs       : {self.ppo_epochs}",
            f"  LR / Ent / ε     : {self.learning_rate:.0e} / {self.entropy_coef:.3f} / {self.rollout_epsilon:.3f}",
            f"  γ / λ / clip     : {self.gamma:.3f} / {self.gae_lambda:.3f} / {self.clip_eps:.2f}",
            f"  Resume           : {self.resume}",
            f"  SL warmstart     : {self.sl_warmstart}",
            f"  Start pellets    : {pellets_str}",
            f"  BFS shaping      : {self.use_bfs_shaping}",
            f"  Ghost speed/conf : {self.ghost_speed_ratio:.2f} / {self.ghost_confusion_prob:.2f}",
            f"  Search guided    : {self.search_guided} (depth {self.search_horizon}, α={self.search_alpha:.2f}, distill={self.distill_coef:.2f})",
            f"  Eval             : every {self.eval_interval} upd × {self.eval_episodes} eps",
            f"  Save dir         : {self.model_dir}",
            f"  Log file         : {self.log_file}",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLOUT BUFFER — bundles the 11 rollout lists into one object
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RolloutBuffer:
    """Collected rollout data before PPO update. Lists of (1,) tensors, one per step."""

    grids: list[torch.Tensor]
    features: list[torch.Tensor]
    valid_actions: list[torch.Tensor]
    actions: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    rewards: list[torch.Tensor]
    dones: list[torch.Tensor]
    terminated: list[torch.Tensor]
    resets: list[torch.Tensor]
    values: list[torch.Tensor]
    seq_hiddens: list[torch.Tensor]
    finished_episodes: list[dict[str, Any]]  # ep records for logging/eval
    search_dists: list[torch.Tensor] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════════════════════════


class PacmanTrainer:
    """Orchestrates PPO training: rollout → GAE → BPTT update → eval → checkpoint."""

    def __init__(self, config: TrainingConfig) -> None:
        self.cfg = config
        self.device = self._select_device()
        # Logging
        self.logger = TrainingLogger(config.log_file, quiet=False)

        # Paths
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)
        # Single unified checkpoint — same weights used across all curriculum stages
        self.checkpoint_path = self.cfg.model_dir / "player_rl.pt"
        self.best_checkpoint_path = self.cfg.model_dir / "player_rl_best.pt"

        self.env = self._build_env()
        self.policy = PlayerActorCritic().to(self.device)
        self.hidden_dim = (
            self.policy.backbone.gru_num_layers * self.policy.backbone.gru_hidden_size
        )  # matches PacmanCNNBackbone GRU (2 layers x 384 = 768)
        self.optimizer = self._build_optimizer()
        # Full float32 precision across CPU and GPU for RL stability
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
        self.ref_policy: PlayerActorCritic | None = None  # for SL warmstart KL reg

        # Loaded state
        self.loaded_checkpoint = False

        # Live training state (initialized in train(), not __init__)
        self.obs: tuple = ()
        self.policy_hidden: torch.Tensor | None = None
        self.recent_episodes: deque[dict[str, Any]] = deque(maxlen=100)
        self.current_ep_reward = 0.0
        self.current_ep_steps = 0
        self.total_completed_episodes = 0
        self.last_update_completed = 0
        self.start_time = 0.0

        # Best-tracking (train window)
        self.best_avg_pct = 0.0
        self.best_avg_pellets = 0.0

        # Eval tracking
        self.eval_env: PacmanPlayerEnv | None = None
        self.best_eval_score = float("-inf")
        self.best_eval_update = -1
        self.last_meaningful_improve_upd = -1
        self.eval_best_active = False
        self.last_stall_warn_update = -1
        self.prev_eval: dict[str, Any] | None = None

        # Quit listener (started in train())
        self.quit_listener = QuitListener()
        # Auto-curriculum graduation counter
        self._grad_consec: int = 0

    # ───────────────────────────────────────────────────────────────────────
    # Setup helpers
    # ───────────────────────────────────────────────────────────────────────

    def _select_device(self) -> torch.device:
        if self.cfg.device:
            dev = torch.device(self.cfg.device)
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        return dev

    def _build_env(self) -> PacmanPlayerEnv:
        return PacmanPlayerEnv(
            seed=self.cfg.seed,
            stage=self.cfg.stage,
            device="cpu",
            start_pellets=self.cfg.start_pellets,
            use_bfs_shaping=self.cfg.use_bfs_shaping,
            ghost_speed_ratio=self.cfg.ghost_speed_ratio,
            ghost_confusion_prob=self.cfg.ghost_confusion_prob,
        )

    def _build_optimizer(self) -> torch.optim.Adam:
        # If SL warmstart is used without a loaded checkpoint, only train
        # the heads (backbone frozen). Otherwise train everything.
        if self.cfg.sl_warmstart and not self.loaded_checkpoint:
            params = [p for p in self.policy.parameters() if p.requires_grad]
            return torch.optim.Adam(params, lr=5e-5)
        return torch.optim.Adam(self.policy.parameters(), lr=self.cfg.learning_rate)

    # ───────────────────────────────────────────────────────────────────────
    # Checkpoint loading
    # ───────────────────────────────────────────────────────────────────────

    def _try_load_checkpoint(self) -> bool:
        """Try to resume from any existing checkpoint. Returns True if loaded."""
        if not self.cfg.resume or self.cfg.sl_warmstart:
            return False

        candidates = [
            self.checkpoint_path,
            self.best_checkpoint_path,
            self.cfg.model_dir / "player_rl_stage2_best.pt",
            self.cfg.model_dir / "player_rl_stage2.pt",
            Path(__file__).resolve().parent.parent
            / "data"
            / "player_rl_stage2_best.pt",
            Path(__file__).resolve().parent.parent / "data" / "player_rl_stage2.pt",
        ]
        for cand in candidates:
            if cand.exists():
                if load_checkpoint_into_policy(self.policy, cand, device=self.device):
                    self.logger.log(f"Resumed from {cand.name} ({cand})")
                    return True
        return False

    def _maybe_sl_warmstart(self) -> None:
        """If SL warmstart is requested, load SL weights and freeze backbone."""
        if self.cfg.sl_warmstart and not self.loaded_checkpoint:
            sl_path = self.cfg.model_dir / "player_sl_best.pt"
            if sl_path.exists():
                load_sl_weights_into_ppo(self.policy, str(sl_path), device=self.device)
                self.logger.log(f"Warm-started from {sl_path.name}")

                for p in self.policy.backbone.parameters():
                    p.requires_grad = False
                self.logger.log("Frozen CNN backbone (SL pre-trained).")

                # Reference policy for KL regularization during fine-tuning
                self.ref_policy = PlayerActorCritic().to(self.device)
                load_sl_weights_into_ppo(
                    self.ref_policy, str(sl_path), device=self.device
                )
                self.ref_policy.eval()
                for p in self.ref_policy.parameters():
                    p.requires_grad = False
            else:
                self.logger.log(
                    f"SL warmstart requested but {sl_path} not found; starting fresh."
                )
        elif not self.loaded_checkpoint:
            self.logger.log("Starting from fresh random weights.")

    # ───────────────────────────────────────────────────────────────────────
    # Rollout collection
    # ───────────────────────────────────────────────────────────────────────

    def collect_rollout(self) -> RolloutBuffer:
        """Run env for cfg.rollout_steps and collect transition data."""
        buf = RolloutBuffer(
            grids=[],
            features=[],
            valid_actions=[],
            actions=[],
            log_probs=[],
            rewards=[],
            dones=[],
            terminated=[],
            resets=[],
            values=[],
            seq_hiddens=[],
            finished_episodes=[],
            search_dists=[],
        )
        step_just_reset = True

        for step in range(self.cfg.rollout_steps):
            grid, features, valid_actions = self.obs

            # Record hidden state at start of each sequence chunk
            if step % self.cfg.seq_len == 0:
                buf.seq_hiddens.append(
                    self.policy_hidden.view(self.hidden_dim).cpu()
                    if self.policy_hidden is not None
                    else torch.zeros(self.hidden_dim)
                )

            search_dist = None
            if self.cfg.search_guided:
                search_dist = self.env.get_search_distribution(
                    horizon=self.cfg.search_horizon
                )
                buf.search_dists.append(search_dist.cpu())

            action, log_prob, value, explore_branch = self._sample_action(
                grid, features, valid_actions, search_dist
            )
            next_obs, reward, done, info = self._env_step(action, explore_branch)

            # Record transition
            buf.grids.append(grid)
            buf.features.append(features)
            buf.valid_actions.append(valid_actions)
            buf.actions.append(action.cpu())
            buf.log_probs.append(log_prob.cpu())
            buf.rewards.append(torch.tensor([reward], dtype=torch.float32))
            buf.dones.append(torch.tensor([done], dtype=torch.float32))
            buf.terminated.append(
                torch.tensor([info.get("terminated", False)], dtype=torch.float32)
            )
            buf.resets.append(
                torch.tensor([1.0 if step_just_reset else 0.0], dtype=torch.float32)
            )
            buf.values.append(value.squeeze(-1).cpu())

            step_just_reset = self._handle_step_outcome(done, info, next_obs, buf)

        return buf

    def _sample_action(
        self,
        grid: torch.Tensor,
        features: torch.Tensor,
        valid_actions: torch.Tensor,
        search_dist: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Forward policy, sample action with ε-exploration, return (action, log_prob, value, explore)."""
        with torch.no_grad():
            logits, value, self.policy_hidden = self.policy(
                grid.to(self.device),
                features.to(self.device),
                self.policy_hidden,
            )
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise RuntimeError(
                    "NaN/Inf detected in policy logits during rollout! Policy network corrupted."
                )

            masked_logits = logits.masked_fill(~valid_actions.to(self.device), -1e4)
            masked_logits = torch.clamp(masked_logits, min=-1e4, max=1e4)
            probs = F.softmax(masked_logits, dim=-1)

            # Search-guided sampling
            if self.cfg.search_guided and search_dist is not None:
                s_dist = search_dist.to(self.device).view(1, -1)
                alpha = self.cfg.search_alpha
                if torch.rand(1).item() < alpha:
                    action = torch.argmax(s_dist, dim=-1)
                else:
                    action = Categorical(probs=probs).sample()
                log_prob = torch.log(probs[0, action].clamp(min=1e-8))
                return action, log_prob, value, False

            if self.cfg.rollout_epsilon > 0:
                valid_f = valid_actions.to(self.device).float()
                uniform = valid_f / valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
                mix = (
                    1.0 - self.cfg.rollout_epsilon
                ) * probs + self.cfg.rollout_epsilon * uniform

                # Branch-explicit sampling: decide branch first, then sample purely
                # from that branch. Behavior log-prob is the MIXTURE prob either way,
                # so PPO importance ratios stay correct.
                if torch.rand(1).item() < self.cfg.rollout_epsilon:
                    dist = Categorical(probs=uniform)
                    explore_branch = True
                else:
                    dist = Categorical(probs=probs)
                    explore_branch = False
                action = dist.sample()
                log_prob = torch.log(mix[0, action].clamp(min=1e-8))
            else:
                explore_branch = False
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        return action, log_prob, value, explore_branch

    def _env_step(
        self, action: torch.Tensor, explore_branch: bool
    ) -> tuple[tuple, float, bool, dict]:
        """Step the env, update episode counters."""
        next_obs, reward, done, info, _ = self.env.step(
            action.item(), explore=explore_branch
        )
        self.current_ep_reward += reward
        self.current_ep_steps += 1
        return next_obs, reward, done, info

    def _handle_step_outcome(
        self,
        done: bool,
        info: dict,
        next_obs: tuple,
        buf: RolloutBuffer,
    ) -> bool:
        """Handle hidden-state reset and episode record. Returns step_just_reset for next step."""
        if done or info.get("events", {}).get("pacman_died", False):
            self.policy_hidden = None
            if done:
                self.obs = self.env.reset()
                ep_record = self._build_episode_record(info)  # ← assign to variable
                buf.finished_episodes.append(ep_record)
                self.recent_episodes.append(ep_record)  # ← now it's defined
                self.current_ep_reward = 0.0
                self.current_ep_steps = 0
                self.total_completed_episodes += 1
                return True
            else:
                self.obs = next_obs
                return True
        else:
            self.obs = next_obs
            return False

    def _build_episode_record(self, info: dict) -> dict[str, Any]:
        ep_steps = max(1.0, float(self.current_ep_steps))
        osc_cnt = float(info["episode_event_counts"].get("osc", 0))
        return {
            "reward": self.current_ep_reward,
            "pellets": float(info["pellets_eaten"]),
            "pct": float(info["completion_pct"]),
            "steps": ep_steps,
            "osc_count": osc_cnt,
            "osc_pct": (osc_cnt / ep_steps) * 100.0,
            "maze": info["maze"],
            "max_steps": float(info.get("max_steps", 0)),
            "episode_event_counts": info["episode_event_counts"],
            "episode_reward_breakdown": info["episode_reward_breakdown"],
            "telemetry": info.get("telemetry", {}),
        }

    # ───────────────────────────────────────────────────────────────────────
    # GAE + sequence preparation
    # ───────────────────────────────────────────────────────────────────────

    def _bootstrap_next_value(self) -> torch.Tensor:
        with torch.no_grad():
            last_grid, last_features, _ = self.obs
            _, next_value, _ = self.policy(
                last_grid.to(self.device), last_features.to(self.device), None
            )
            return next_value.squeeze(-1)

    def _prepare_sequence_tensors(
        self, buf: RolloutBuffer, next_value: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Cat, truncate to seq chunks, reshape into (NUM_SEQ, SEQ_LEN, ...), compute GAE."""
        cfg = self.cfg
        n = cfg.num_sequences * cfg.seq_len

        b_grids = torch.cat(buf.grids, dim=0)[:n].to(self.device)
        b_features = torch.cat(buf.features, dim=0)[:n].to(self.device)
        b_valid = torch.cat(buf.valid_actions, dim=0)[:n].to(self.device)
        b_actions = torch.cat(buf.actions, dim=0)[:n].to(self.device)
        b_log_probs = torch.cat(buf.log_probs, dim=0)[:n].to(self.device)
        b_rewards = torch.cat(buf.rewards, dim=0)[:n].to(self.device)
        b_dones = torch.cat(buf.dones, dim=0)[:n].to(self.device)
        b_terminated = torch.cat(buf.terminated, dim=0)[:n].to(self.device)
        b_resets = torch.cat(buf.resets, dim=0)[:n].to(self.device)
        b_values = torch.cat(buf.values, dim=0)[:n].to(self.device)
        b_seq_hiddens = torch.stack(buf.seq_hiddens, dim=0)[: cfg.num_sequences].to(
            self.device
        )

        advantages, returns = compute_gae(
            b_rewards,
            b_values,
            b_dones,
            b_terminated,
            next_value,
            cfg.gamma,
            cfg.gae_lambda,
        )

        seq_dict = {
            "grids_seq": b_grids.view(
                cfg.num_sequences, cfg.seq_len, *b_grids.shape[1:]
            ),
            "features_seq": b_features.view(
                cfg.num_sequences, cfg.seq_len, *b_features.shape[1:]
            ),
            "valid_seq": b_valid.view(
                cfg.num_sequences, cfg.seq_len, *b_valid.shape[1:]
            ),
            "actions_seq": b_actions.view(cfg.num_sequences, cfg.seq_len),
            "log_probs_seq": b_log_probs.view(cfg.num_sequences, cfg.seq_len),
            "resets_seq": b_resets.view(cfg.num_sequences, cfg.seq_len),
            "advantages_seq": advantages.view(cfg.num_sequences, cfg.seq_len),
            "returns_seq": returns.view(cfg.num_sequences, cfg.seq_len),
            "seq_hiddens": b_seq_hiddens,
        }
        if buf.search_dists:
            b_search_dists = torch.stack(buf.search_dists, dim=0)[:n].to(self.device)
            seq_dict["search_dists_seq"] = b_search_dists.view(
                cfg.num_sequences, cfg.seq_len, 4
            )
        return seq_dict

    # ───────────────────────────────────────────────────────────────────────
    # PPO update
    # ───────────────────────────────────────────────────────────────────────

    def ppo_update(
        self, seq_tensors: dict[str, torch.Tensor]
    ) -> tuple[float, float, float]:
        """Run PPO epochs over minibatches of sequence chunks. Returns (policy_loss, value_loss, entropy)."""
        cfg = self.cfg
        kl_coef = 0.20 if self.ref_policy is not None else 0.0
        eff_entropy = 0.001 if self.ref_policy is not None else cfg.entropy_coef

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_minibatches = 0

        for _ in range(cfg.ppo_epochs):
            seq_perm = torch.randperm(cfg.num_sequences, device=self.device)
            for start in range(0, cfg.num_sequences, cfg.minibatch_seqs):
                mb_idx = seq_perm[start : start + cfg.minibatch_seqs]
                p_loss, v_loss, ent = self._ppo_minibatch_step(
                    seq_tensors, mb_idx, kl_coef, eff_entropy
                )
                total_policy_loss += p_loss
                total_value_loss += v_loss
                total_entropy += ent
                num_minibatches += 1

        n = max(1, num_minibatches)
        return total_policy_loss / n, total_value_loss / n, total_entropy / n

    def _ppo_minibatch_step(
        self,
        seq_tensors: dict[str, torch.Tensor],
        mb_idx: torch.Tensor,
        kl_coef: float,
        eff_entropy: float,
    ) -> tuple[float, float, float]:
        """One gradient step on one minibatch of sequences."""
        cfg = self.cfg

        mb_grid = seq_tensors["grids_seq"][mb_idx]
        mb_features = seq_tensors["features_seq"][mb_idx]
        mb_valid = seq_tensors["valid_seq"][mb_idx]
        mb_actions = seq_tensors["actions_seq"][mb_idx]
        mb_old_log_probs = seq_tensors["log_probs_seq"][mb_idx]
        mb_adv = seq_tensors["advantages_seq"][mb_idx]
        mb_returns = seq_tensors["returns_seq"][mb_idx]
        mb_resets = seq_tensors["resets_seq"][mb_idx]
        mb_h_raw = seq_tensors["seq_hiddens"][mb_idx]
        mb_hidden = (
            mb_h_raw.view(
                -1,
                self.policy.backbone.gru_num_layers,
                self.policy.backbone.gru_hidden_size,
            )
            .permute(1, 0, 2)
            .contiguous()
            .detach()
        )

        self.optimizer.zero_grad()
        # FP32 precision across CPU and GPU for numerical stability
        with torch.amp.autocast("cuda", enabled=False):
            logits, values, _ = self.policy(
                mb_grid, mb_features, mb_hidden, dones=mb_resets
            )
            if torch.isnan(logits).any() or torch.isnan(values).any():
                raise RuntimeError(
                    "NaN detected in policy or value forward pass during PPO update!"
                )

            masked_logits = logits.masked_fill(~mb_valid, -1e4)
            masked_logits = torch.clamp(masked_logits, min=-1e4, max=1e4)
            dist = Categorical(logits=masked_logits)

            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            log_ratio = torch.clamp(new_log_probs - mb_old_log_probs, min=-20.0, max=20.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.smooth_l1_loss(values.reshape(-1), mb_returns.reshape(-1))

            kl_loss = self._compute_kl_loss(
                mb_grid, mb_features, mb_hidden, mb_resets, mb_valid, masked_logits
            )

            distill_loss = torch.tensor(0.0, device=self.device)
            if "search_dists_seq" in seq_tensors:
                mb_search_dists = seq_tensors["search_dists_seq"][mb_idx]
                log_probs_all = F.log_softmax(masked_logits, dim=-1)
                distill_loss = -(mb_search_dists * log_probs_all).sum(dim=-1).mean()

            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - eff_entropy * entropy
                + kl_coef * kl_loss
                + (cfg.distill_coef * distill_loss if cfg.search_guided else 0.0)
            )

            if torch.isnan(loss):
                raise RuntimeError("NaN detected in calculated PPO loss!")

        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def _compute_kl_loss(
        self,
        mb_grid: torch.Tensor,
        mb_features: torch.Tensor,
        mb_hidden: torch.Tensor,
        mb_resets: torch.Tensor,
        mb_valid: torch.Tensor,
        masked_logits: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence to reference (SL) policy. Zero if no ref_policy."""
        if self.ref_policy is None:
            return torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            ref_logits, _, _ = self.ref_policy(
                mb_grid, mb_features, mb_hidden, dones=mb_resets
            )
            ref_masked = ref_logits.masked_fill(~mb_valid, -1e4)
            ref_masked = torch.clamp(ref_masked, min=-1e4, max=1e4)
            ref_probs = F.softmax(ref_masked, dim=-1)
            ref_log_p = F.log_softmax(ref_masked, dim=-1)
        log_p = F.log_softmax(masked_logits, dim=-1)
        return (ref_probs * (ref_log_p - log_p)).sum(dim=-1).mean()

    # ───────────────────────────────────────────────────────────────────────
    # Logging
    # ───────────────────────────────────────────────────────────────────────

    def _compute_window_stats(
        self, window_eps: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Aggregate per-episode stats for the save window."""
        if not window_eps:
            return dict(
                window_max_pct=0.0,
                max_pellets=0,
                epoch_avg_reward=0.0,
                avg_area=0.0,
                avg_w=0.0,
                avg_h=0.0,
                completion_rate=0.0,
                truncation_rate=0.0,
                avg_osc_pct=0.0,
            )
        n = len(window_eps)
        return dict(
            window_max_pct=max(ep["pct"] for ep in window_eps),
            max_pellets=int(max(ep["pellets"] for ep in window_eps)),
            epoch_avg_reward=sum(ep["reward"] for ep in window_eps) / n,
            avg_area=sum(ep["maze"][0] * ep["maze"][1] for ep in window_eps) / n,
            avg_w=sum(ep["maze"][0] for ep in window_eps) / n,
            avg_h=sum(ep["maze"][1] for ep in window_eps) / n,
            completion_rate=sum(
                ep["episode_event_counts"].get("completed", 0) > 0 for ep in window_eps
            )
            / n,
            truncation_rate=sum(
                ep["episode_event_counts"].get("truncated", 0) > 0 for ep in window_eps
            )
            / n,
            avg_osc_pct=sum(ep.get("osc_pct", 0.0) for ep in window_eps) / n,
        )

    def _compute_recent_stats(self) -> tuple[float, float, float]:
        if not self.recent_episodes:
            return self.current_ep_reward, 0.0, 0.0
        n = len(self.recent_episodes)
        return (
            sum(ep["reward"] for ep in self.recent_episodes) / n,
            sum(ep["pellets"] for ep in self.recent_episodes) / n,
            sum(ep["pct"] for ep in self.recent_episodes) / n,
        )

    def _log_update(
        self,
        update: int,
        losses: tuple[float, float, float],
        window_eps: list[dict[str, Any]],
        update_elapsed: float,
    ) -> None:
        avg_policy_loss, avg_value_loss, avg_entropy = losses
        avg_reward, avg_pellets, avg_pct = self._compute_recent_stats()
        stats = self._compute_window_stats(window_eps)

        # Train-window best (only until first eval runs)
        if (
            self.recent_episodes
            and not self.eval_best_active
            and avg_pct > self.best_avg_pct
        ):
            self.best_avg_pct = avg_pct
            self.best_avg_pellets = avg_pellets
            torch.save(self.policy.state_dict(), self.best_checkpoint_path)

        breakdown_line = format_breakdown_line(window_eps)
        breakdown_chunk = f"{breakdown_line} | " if breakdown_line else ""

        surv_window = window_eps if window_eps else list(self.recent_episodes)
        survival_line = format_survival_line(compute_survival_stats(surv_window))

        total_elapsed = time.time() - self.start_time
        self.logger.log(
            f"Upd {update:03d}/{self.cfg.num_updates:03d} | "
            f"Tot Ep: {self.total_completed_episodes:03d} | "
            f"Avg Epoch Rwd: {stats['epoch_avg_reward']:6.1f} | "
            f"Max Pellets: {stats['max_pellets']:3d} ({stats['window_max_pct']:4.1f}%) | "
            f"Avg Pellets: {avg_pellets:5.1f} ({avg_pct:4.1f}%) | "
            f"Osc%: {stats['avg_osc_pct']:4.1f}% | "
            f"Avg Rwd: {avg_reward:4.1f} | "
            f"{breakdown_chunk}"
            f"Ent: {avg_entropy:.3f} | "
            f"Loss (P/V): {avg_policy_loss:.4f}/{avg_value_loss:.4f} | "
            f"Time: {total_elapsed:5.1f}s ({update_elapsed:4.2f}s/upd)"
            f" | Complete: {stats['completion_rate']:5.1%}"
            f" | Truncated: {stats['truncation_rate']:5.1%}"
            f" | Avg Maze: {stats['avg_area']:.1f} ({stats['avg_w']:.1f}x{stats['avg_h']:.1f})"
        )
        self.logger.log(f"   SURV | {survival_line}")

    # ───────────────────────────────────────────────────────────────────────
    # Eval + checkpoint
    # ───────────────────────────────────────────────────────────────────────

    def _maybe_save_and_eval(
        self, update: int, window_eps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Save checkpoint, run eval if cadence matches. Returns remaining window_eps."""
        cfg = self.cfg
        if update % cfg.save_interval != 0 and update != cfg.num_updates:
            return window_eps

        torch.save(self.policy.state_dict(), self.checkpoint_path)
        # Clear the window — next save_interval starts fresh
        remaining: list[dict[str, Any]] = []

        if update % cfg.eval_interval == 0 or update == cfg.num_updates:
            self._run_eval(update)

        return remaining

    def _run_eval(self, update: int) -> None:
        """Fixed-seed benchmark eval with paired seeds across all evals."""
        cfg = self.cfg
        if self.eval_env is None:
            self.eval_env = PacmanPlayerEnv(
                seed=cfg.eval_seed_base,
                stage=cfg.stage,
                device="cpu",
                ghost_speed_ratio=cfg.ghost_speed_ratio,
                ghost_confusion_prob=cfg.ghost_confusion_prob,
            )
            self.eval_env.start_pellets = cfg.start_pellets
        else:
            self.eval_env.start_pellets = cfg.start_pellets

        t_eval = time.time()
        eval_result = run_evaluation(
            self.policy,
            device=self.device,
            stage=cfg.stage,
            episodes=cfg.eval_episodes,
            seed_base=cfg.eval_seed_base,
            env=self.eval_env,
        )
        eval_result["update"] = update
        eval_result["checkpoint"] = self.checkpoint_path.name
        append_history(eval_result)

        self._log_eval_result(eval_result, update, t_eval)
        self._update_eval_tracking(eval_result, update)
        self._check_curriculum_graduation(eval_result, update)
        self.prev_eval = eval_result

    def _log_eval_result(self, eval_result: dict, update: int, t_start: float) -> None:
        score = float(eval_result["eval_score"])
        es = eval_result["survival"]
        esc_str = f"{es['escape_rate'] * 100:.0f}%" if es["escape_rate"] >= 0 else "n/a"
        delta_score = (
            f" ({score - self.prev_eval['eval_score']:+.1f})"
            if self.prev_eval is not None
            else ""
        )
        self.logger.log(
            f"  EVAL @{update:03d}: score {score:7.1f}{delta_score} | "
            f"pellet {eval_result['avg_pellet_pct']:5.1f}% | "
            f"comp {eval_result['completion_rate'] * 100:5.1f}% | "
            f"death {eval_result['death_rate'] * 100:5.1f}% | "
            f"life {es['avg_steps_lived']:4.0f}mv/{es['avg_life_pct']:3.0f}% | "
            f"esc {esc_str:>4}[{es['escape_samples']:3d}] | "
            f"corn {es['cornered_steps_per_ep']:5.2f}/ep | "
            f"MinD {es['avg_min_ghost_dist']:5.2f} | "
            f"conf {eval_result['env']['ghost_confusion']:.2f} | "
            f"({time.time() - t_start:.1f}s)"
        )

    def _update_eval_tracking(self, eval_result: dict, update: int) -> None:
        """Track best score, save best checkpoint, fire stall warnings."""
        cfg = self.cfg
        score = float(eval_result["eval_score"])

        prev_best = self.best_eval_score
        if score > self.best_eval_score:
            if self.best_eval_update >= 0:
                self.logger.log(
                    f"  EVAL: new best score {score:.1f} "
                    f"(was {self.best_eval_score:.1f} @upd {self.best_eval_update}) "
                    f"→ saved {self.best_checkpoint_path.name}"
                )
            self.best_eval_score = score
            self.best_eval_update = update
            self.eval_best_active = True
            self.last_stall_warn_update = -1
            torch.save(self.policy.state_dict(), self.best_checkpoint_path)

        # Stall detection: kill dead runs early
        if prev_best < 0 or score > prev_best + cfg.eval_min_improvement:
            self.last_meaningful_improve_upd = update

        stall_due = (
            self.last_meaningful_improve_upd >= 0
            and update - self.last_meaningful_improve_upd
            >= cfg.eval_stall_patience * cfg.eval_interval
        )
        warned_recently = (
            self.last_stall_warn_update >= 0
            and update - self.last_stall_warn_update
            < cfg.eval_stall_patience * cfg.eval_interval
        )
        if stall_due and not warned_recently:
            self._warn_stall(update, score)

    def _warn_stall(self, update: int, score: float) -> None:
        self.logger.log("!" * 60)
        self.logger.log(
            f"STALL WARNING @upd {update}: no meaningful eval "
            f"improvement (≥{self.cfg.eval_min_improvement:.0f} pts) for "
            f"{update - self.last_meaningful_improve_upd} updates."
        )
        self.logger.log(
            f"  best score {self.best_eval_score:.1f} @upd "
            f"{self.best_eval_update}; current {score:.1f}."
        )
        self.logger.log(
            "  If the SURV lines above are also flat (Esc% / Corn / "
            "CDth not trending), this run is probably NOT learning —"
            " consider killing it ('q') and changing reward/curriculum."
        )
        self.logger.log("!" * 60)
        self.last_stall_warn_update = update

    # ──────────────────────────────────────────────────────────────────────
    # Auto-curriculum
    # ──────────────────────────────────────────────────────────────────────

    def _check_curriculum_graduation(self, eval_result: dict, update: int) -> None:
        """After each stage-1 eval, check if completion threshold met N times in a row."""
        if not self.cfg.auto_curriculum or self.cfg.stage != 1:
            return
        comp = float(eval_result.get("completion_rate", 0.0))
        if comp >= self.cfg.stage1_grad_threshold:
            self._grad_consec += 1
            self.logger.log(
                f"  CURRICULUM: stage-1 completion {comp:.1%} ≥ "
                f"{self.cfg.stage1_grad_threshold:.0%} "
                f"({self._grad_consec}/{self.cfg.stage1_grad_evals} needed to graduate)"
            )
        else:
            if self._grad_consec > 0:
                self.logger.log(
                    f"  CURRICULUM: reset streak — completion {comp:.1%} "
                    f"fell below {self.cfg.stage1_grad_threshold:.0%}"
                )
            self._grad_consec = 0

        if self._grad_consec >= self.cfg.stage1_grad_evals:
            self._graduate_to_stage2(update)

    def _graduate_to_stage2(self, update: int) -> None:
        """Promote model from ghost-free stage 1 to full-ghost stage 2. Same weights, same file."""
        cfg = self.cfg
        self.logger.log("=" * 60)
        self.logger.log(f"  CURRICULUM GRADUATE @upd {update}: stage 1 → 2")
        self.logger.log(
            f"  Completion ≥ {cfg.stage1_grad_threshold:.0%} for "
            f"{cfg.stage1_grad_evals} consecutive evals. Adding ghosts now."
        )
        self.logger.log("  Same model weights, same checkpoint file (player_rl.pt).")
        self.logger.log("=" * 60)
        cfg.stage = 2
        self.env = self._build_env()
        self.obs = self.env.reset()
        self.policy_hidden = None
        self.eval_env = None  # rebuilt on next eval
        self._grad_consec = 0
        # Reset eval tracking so stage-2 evals start fresh
        self.best_eval_score = float("-inf")
        self.best_eval_update = -1
        self.prev_eval = None
        self.last_meaningful_improve_upd = -1

    # ───────────────────────────────────────────────────────────────────────
    # Cleanup
    # ───────────────────────────────────────────────────────────────────────

    def _cleanup_rollout_tensors(
        self, buf: RolloutBuffer, seq_tensors: dict[str, torch.Tensor]
    ) -> None:
        """Free rollout tensors to avoid OOM on long runs."""
        for v in seq_tensors.values():
            del v
        for attr in [
            "grids",
            "features",
            "valid_actions",
            "actions",
            "log_probs",
            "rewards",
            "dones",
            "terminated",
            "resets",
            "values",
            "seq_hiddens",
            "search_dists",
        ]:
            getattr(buf, attr).clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ───────────────────────────────────────────────────────────────────────
    # Final summary
    # ───────────────────────────────────────────────────────────────────────

    def _final_summary(self) -> None:
        self.logger.log("=" * 60)
        self.logger.log(
            f"Stage {self.cfg.stage} stopped after {self.last_update_completed} updates, "
            f"{time.time() - self.start_time:.1f}s"
        )
        self.logger.log(f"Checkpoint: {self.checkpoint_path}")
        if self.best_eval_update >= 0:
            self.logger.log(
                f"Best (eval score {self.best_eval_score:.1f} @upd {self.best_eval_update}):"
                f" {self.best_checkpoint_path}"
            )
        else:
            self.logger.log(
                f"Best (train avg {self.best_avg_pct:.1f}% | {self.best_avg_pellets:.0f}"
                f" pellets): {self.best_checkpoint_path}"
            )
        self.logger.log("Eval history: AI_arena/evals/eval_history.json")
        self.logger.log("=" * 60)

    # ───────────────────────────────────────────────────────────────────────
    # Main training loop
    # ───────────────────────────────────────────────────────────────────────

    def train(self) -> None:
        """Main PPO training loop with GRU memory and reset-safe sequence training."""
        # Initialize state
        self.start_time = time.time()
        self.obs = self.env.reset()
        self.policy_hidden = None
        self.quit_listener.start()

        self.logger.log("=" * 60)
        self.logger.log(f"Stage {self.cfg.stage} PPO Training | Device: {self.device}")
        self.logger.log(self.cfg.summary())
        self.logger.log("=" * 60)

        # Load checkpoint or warmstart (must happen after logger is ready)
        self.loaded_checkpoint = self._try_load_checkpoint()
        self._maybe_sl_warmstart()
        # Rebuild optimizer if SL warmstart froze the backbone
        if self.cfg.sl_warmstart and not self.loaded_checkpoint:
            self.optimizer = self._build_optimizer()
            self.logger.log("LR set to 5e-5 for head fine-tuning.")

        try:
            for update in range(1, self.cfg.num_updates + 1):
                update_start = time.time()

                # ── Rollout ──
                buf = self.collect_rollout()

                # ── Bootstrap value for GAE ──
                next_value = self._bootstrap_next_value()

                # ── Prepare sequence tensors + GAE ──
                seq_tensors = self._prepare_sequence_tensors(buf, next_value)

                # ── PPO update ──
                losses = self.ppo_update(seq_tensors)

                # ── Logging ──
                self.last_update_completed = update
                self._log_update(
                    update, losses, buf.finished_episodes, time.time() - update_start
                )

                # ── Save + eval ──
                # _maybe_save_and_eval clears the window after saving
                buf.finished_episodes = self._maybe_save_and_eval(
                    update, buf.finished_episodes
                )

                # ── Quit listener ──
                if self.quit_listener.stop_requested:
                    self.logger.log(f"\n'q' pressed — stopping at update {update}.")
                    torch.save(self.policy.state_dict(), self.checkpoint_path)
                    break

                # ── Cleanup ──
                self._cleanup_rollout_tensors(buf, seq_tensors)

        except KeyboardInterrupt:
            self.logger.log(
                f"\nKeyboardInterrupt — saving at update {self.last_update_completed}."
            )
            torch.save(self.policy.state_dict(), self.checkpoint_path)
            self.logger.log(f"Checkpoint: {self.checkpoint_path}")
            self.logger.log(
                f"Best ({self.best_avg_pct:.1f}% | {self.best_avg_pellets:.0f} pellets):"
                f" {self.best_checkpoint_path}"
            )

        finally:
            self._final_summary()
            self.quit_listener.stop()
            self.logger.close()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """CLI entry point. Build config from args, create trainer, run."""
    config = TrainingConfig.from_argv()
    trainer = PacmanTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
