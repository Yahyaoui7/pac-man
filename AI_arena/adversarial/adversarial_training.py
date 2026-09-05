"""Clean PPO pipeline for Alternating Adversarial Training of Pac-Man vs Ghosts."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from AI_arena.adversarial.adversarial_env import PacmanAdversarialEnv
from AI_arena.models.cnn_player import (
    PlayerActorCritic,
    load_checkpoint_into_policy as load_player_checkpoint,
)
from AI_arena.models.cnn_ghost import (
    GhostActorCritic,
    load_sl_ghost_weights_into_rl,
)
from AI_arena.player.player_training import compute_gae
from AI_arena.player.utils import (
    QuitListener,
    TrainingLogger,
    format_breakdown_line,
)


@dataclass
class AdversarialTrainingConfig:
    """Hyperparameters for adversarial training."""

    # Core loop
    stage: int = 2
    num_updates: int = 2000
    rollout_steps: int = 2048
    seq_len: int = 32
    minibatch_seqs: int = 4
    ppo_epochs: int = 4

    # Asymmetric Training
    ghosts_warmup_updates: int = 50
    ghost_update_ratio: int = 5

    # Optimization
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.005
    value_coef: float = 0.05
    max_grad_norm: float = 0.5

    # Environment
    seed: int = 42
    save_interval: int = 50
    device: str | None = None
    run_name: str = "adv_train"
    start_pellets: tuple[int, ...] | None = None
    ghost_speed_ratio: float = 0.50

    # Paths
    model_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
        / "models"
    )
    log_file: Path = field(
        default_factory=lambda: Path("adv_training_log.txt")
    )

    @property
    def num_sequences(self) -> int:
        return self.rollout_steps // self.seq_len

    @classmethod
    def from_argv(
        cls, argv: list[str] | None = None
    ) -> "AdversarialTrainingConfig":
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--updates", type=int, default=2000)
        parser.add_argument("--rollout-steps", type=int, default=2048)
        parser.add_argument(
            "--warmup-updates",
            type=int,
            default=50,
            help="Updates where Player is frozen",
        )
        parser.add_argument(
            "--update-ratio",
            type=int,
            default=5,
            help="Ghost updates per Player update",
        )
        parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
        args = parser.parse_args(argv)

        return cls(
            num_updates=args.updates,
            rollout_steps=args.rollout_steps,
            ghosts_warmup_updates=args.warmup_updates,
            ghost_update_ratio=args.update_ratio,
            device=args.device,
        )


@dataclass
class AdvRolloutBuffer:
    """Transition data for both agents."""

    grids: list[torch.Tensor]
    features: list[torch.Tensor]
    valid_actions: list[torch.Tensor]

    p_actions: list[torch.Tensor]
    p_log_probs: list[torch.Tensor]
    p_rewards: list[torch.Tensor]
    p_values: list[torch.Tensor]
    p_seq_hiddens: list[torch.Tensor]

    g_actions: list[torch.Tensor]
    g_log_probs: list[torch.Tensor]
    g_rewards: list[torch.Tensor]
    g_values: list[torch.Tensor]
    g_seq_hiddens: list[torch.Tensor]

    dones: list[torch.Tensor]
    terminated: list[torch.Tensor]
    resets: list[torch.Tensor]
    finished_episodes: list[dict[str, Any]]


class AdversarialTrainer:
    def __init__(self, config: AdversarialTrainingConfig):
        self.cfg = config
        self.device = torch.device(
            self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = TrainingLogger(self.cfg.log_file, quiet=False)
        self.cfg.model_dir.mkdir(parents=True, exist_ok=True)

        self.player_ckpt = self.cfg.model_dir / "player_rl_adv.pt"
        self.ghost_ckpt = self.cfg.model_dir / "ghost_rl_adv.pt"

        self.env = PacmanAdversarialEnv(
            seed=self.cfg.seed,
            stage=self.cfg.stage,
            device="cpu",
            start_pellets=self.cfg.start_pellets,
            ghost_speed_ratio=self.cfg.ghost_speed_ratio,
        )

        # We need to initialize env to get feature count
        self.obs = self.env.reset()
        feature_count = self.obs[1].shape[-1]

        self.player = PlayerActorCritic(extra_feature_count=feature_count).to(
            self.device
        )
        self.ghosts = GhostActorCritic(extra_feature_count=feature_count).to(
            self.device
        )

        # GRU dimensions (2 layers x 384 hidden)
        self.p_hidden_dim = (
            self.player.backbone.gru_num_layers
            * self.player.backbone.gru_hidden_size
        )
        self.g_hidden_dim = (
            self.ghosts.backbone.gru_num_layers
            * self.ghosts.backbone.gru_hidden_size
        )

        self.p_opt = torch.optim.Adam(
            self.player.parameters(), lr=self.cfg.learning_rate
        )
        self.g_opt = torch.optim.Adam(
            self.ghosts.parameters(), lr=self.cfg.learning_rate
        )

        self.quit_listener = QuitListener()

        # State
        self.p_hidden: torch.Tensor | None = None
        self.g_hidden: torch.Tensor | None = None
        self.recent_episodes: deque = deque(maxlen=50)
        self.cur_p_reward = 0.0
        self.cur_g_reward = 0.0
        self.cur_steps = 0
        self.total_eps = 0

    def _try_load_models(self) -> None:
        """Load baseline RL player and SL ghosts if adv checkpoints don't exist."""
        if self.player_ckpt.exists():
            load_player_checkpoint(self.player, self.player_ckpt, self.device)
            self.logger.log("Loaded adversarial player checkpoint.")
        else:
            base_p = self.cfg.model_dir / "player_rl_best.pt"
            if base_p.exists():
                load_player_checkpoint(self.player, base_p, self.device)
                self.logger.log("Loaded baseline player_rl_best.pt.")

        if self.ghost_ckpt.exists():
            # load as standard state dict since we saved it fully during adv training
            self.ghosts.load_state_dict(
                torch.load(
                    self.ghost_ckpt,
                    map_location=self.device,
                    weights_only=True,
                )
            )
            self.logger.log("Loaded adversarial ghost checkpoint.")
        else:
            base_g = self.cfg.model_dir / "ghost_ai.pt"
            if base_g.exists():
                load_sl_ghost_weights_into_rl(self.ghosts, base_g, self.device)
                self.logger.log("Warm-started ghosts from SL model.")

    def collect_rollout(self, is_player_turn: bool) -> AdvRolloutBuffer:
        buf = AdvRolloutBuffer(
            grids=[],
            features=[],
            valid_actions=[],
            p_actions=[],
            p_log_probs=[],
            p_rewards=[],
            p_values=[],
            p_seq_hiddens=[],
            g_actions=[],
            g_log_probs=[],
            g_rewards=[],
            g_values=[],
            g_seq_hiddens=[],
            dones=[],
            terminated=[],
            resets=[],
            finished_episodes=[],
        )
        step_reset = True

        for step in range(self.cfg.rollout_steps):
            grid, features, valid = self.obs

            if step % self.cfg.seq_len == 0:
                buf.p_seq_hiddens.append(
                    self.p_hidden.view(self.p_hidden_dim).cpu()
                    if self.p_hidden is not None
                    else torch.zeros(self.p_hidden_dim)
                )
                buf.g_seq_hiddens.append(
                    self.g_hidden.view(self.g_hidden_dim).cpu()
                    if self.g_hidden is not None
                    else torch.zeros(self.g_hidden_dim)
                )

            with torch.no_grad():
                # Player
                p_log, p_val, self.p_hidden = self.player(
                    grid.to(self.device),
                    features.to(self.device),
                    self.p_hidden,
                )
                p_masked = p_log.masked_fill(
                    ~valid.to(self.device), -1e8
                ).clamp(min=-1e8, max=1e4)
                p_dist = Categorical(logits=p_masked)
                p_act = p_dist.sample()
                p_lp = p_dist.log_prob(p_act)

                # Ghosts
                g_log, g_val, self.g_hidden = self.ghosts(
                    grid.to(self.device),
                    features.to(self.device),
                    self.g_hidden,
                )
                # No mask for ghosts in this env setup since physics resolves invalid moves, but they should learn.
                g_dist = Categorical(logits=g_log)
                g_act = g_dist.sample()
                g_lp = g_dist.log_prob(g_act).sum(
                    dim=-1
                )  # Joint logprob for 4 ghosts

            # Step
            next_obs, p_rew, g_rew, done, info = self.env.step_adversarial(
                p_act, g_act
            )

            self.cur_p_reward += p_rew
            self.cur_g_reward += g_rew
            self.cur_steps += 1

            # Log
            buf.grids.append(grid)
            buf.features.append(features)
            buf.valid_actions.append(valid)
            buf.p_actions.append(p_act.cpu())
            buf.p_log_probs.append(p_lp.cpu())
            buf.p_rewards.append(torch.tensor([p_rew], dtype=torch.float32))
            buf.p_values.append(p_val.squeeze(-1).cpu())

            buf.g_actions.append(g_act.cpu())
            buf.g_log_probs.append(g_lp.cpu())
            buf.g_rewards.append(torch.tensor([g_rew], dtype=torch.float32))
            buf.g_values.append(g_val.squeeze(-1).cpu())

            buf.dones.append(torch.tensor([done], dtype=torch.float32))
            buf.terminated.append(
                torch.tensor(
                    [info.get("terminated", False)], dtype=torch.float32
                )
            )
            buf.resets.append(
                torch.tensor([1.0 if step_reset else 0.0], dtype=torch.float32)
            )

            if done or info.get("events", {}).get("pacman_died", False):
                self.p_hidden = None
                self.g_hidden = None
                if done:
                    self.obs = self.env.reset()
                    ep_rec = {
                        "p_rew": self.cur_p_reward,
                        "g_rew": self.cur_g_reward,
                        "steps": self.cur_steps,
                        "pct": info["completion_pct"],
                        "died": info.get("events", {}).get(
                            "pacman_died", False
                        ),
                    }
                    buf.finished_episodes.append(ep_rec)
                    self.recent_episodes.append(ep_rec)
                    self.cur_p_reward = 0.0
                    self.cur_g_reward = 0.0
                    self.cur_steps = 0
                    self.total_eps += 1
                    step_reset = True
                else:
                    self.obs = next_obs
                    step_reset = True
            else:
                self.obs = next_obs
                step_reset = False

        return buf

    def _prepare_seqs(
        self, buf: AdvRolloutBuffer, is_player_turn: bool
    ) -> dict:
        cfg = self.cfg
        n = cfg.num_sequences * cfg.seq_len

        b_grids = torch.cat(buf.grids, dim=0)[:n].to(self.device)
        b_feats = torch.cat(buf.features, dim=0)[:n].to(self.device)
        b_resets = torch.cat(buf.resets, dim=0)[:n].to(self.device)
        b_dones = torch.cat(buf.dones, dim=0)[:n].to(self.device)
        b_terms = torch.cat(buf.terminated, dim=0)[:n].to(self.device)

        with torch.no_grad():
            last_g, last_f, _ = self.obs
            if is_player_turn:
                _, nxt_v, _ = self.player(
                    last_g.to(self.device), last_f.to(self.device), None
                )
                rews = torch.cat(buf.p_rewards, dim=0)[:n].to(self.device)
                vals = torch.cat(buf.p_values, dim=0)[:n].to(self.device)
            else:
                _, nxt_v, _ = self.ghosts(
                    last_g.to(self.device), last_f.to(self.device), None
                )
                rews = torch.cat(buf.g_rewards, dim=0)[:n].to(self.device)
                vals = torch.cat(buf.g_values, dim=0)[:n].to(self.device)
            nxt_v = nxt_v.squeeze(-1)

        adv, ret = compute_gae(
            rews, vals, b_dones, b_terms, nxt_v, cfg.gamma, cfg.gae_lambda
        )

        res = {
            "grids": b_grids.view(
                cfg.num_sequences, cfg.seq_len, *b_grids.shape[1:]
            ),
            "features": b_feats.view(
                cfg.num_sequences, cfg.seq_len, *b_feats.shape[1:]
            ),
            "resets": b_resets.view(cfg.num_sequences, cfg.seq_len),
            "adv": adv.view(cfg.num_sequences, cfg.seq_len),
            "ret": ret.view(cfg.num_sequences, cfg.seq_len),
        }

        if is_player_turn:
            b_valid = torch.cat(buf.valid_actions, dim=0)[:n].to(self.device)
            b_act = torch.cat(buf.p_actions, dim=0)[:n].to(self.device)
            b_lp = torch.cat(buf.p_log_probs, dim=0)[:n].to(self.device)
            b_h = torch.stack(buf.p_seq_hiddens, dim=0)[
                : cfg.num_sequences
            ].to(self.device)
            res.update(
                {
                    "valid": b_valid.view(
                        cfg.num_sequences, cfg.seq_len, *b_valid.shape[1:]
                    ),
                    "act": b_act.view(cfg.num_sequences, cfg.seq_len),
                    "lp": b_lp.view(cfg.num_sequences, cfg.seq_len),
                    "h": b_h,
                }
            )
        else:
            b_act = torch.cat(buf.g_actions, dim=0)[:n].to(self.device)
            b_lp = torch.cat(buf.g_log_probs, dim=0)[:n].to(self.device)
            b_h = torch.stack(buf.g_seq_hiddens, dim=0)[
                : cfg.num_sequences
            ].to(self.device)
            res.update(
                {
                    "act": b_act.view(
                        cfg.num_sequences, cfg.seq_len, 4
                    ),  # 4 ghosts
                    "lp": b_lp.view(cfg.num_sequences, cfg.seq_len),
                    "h": b_h,
                }
            )

        return res

    def ppo_update(
        self, seqs: dict, is_player_turn: bool
    ) -> tuple[float, float, float]:
        cfg = self.cfg
        opt = self.p_opt if is_player_turn else self.g_opt
        model = self.player if is_player_turn else self.ghosts

        tot_p, tot_v, tot_e = 0.0, 0.0, 0.0
        n_mb = 0

        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(cfg.num_sequences, device=self.device)
            for st in range(0, cfg.num_sequences, cfg.minibatch_seqs):
                idx = perm[st : st + cfg.minibatch_seqs]

                mg = seqs["grids"][idx]
                mf = seqs["features"][idx]
                mres = seqs["resets"][idx]
                madv = seqs["adv"][idx]
                mret = seqs["ret"][idx]
                mact = seqs["act"][idx]
                molp = seqs["lp"][idx]
                mh = (
                    seqs["h"][idx]
                    .view(
                        -1,
                        model.backbone.gru_num_layers,
                        model.backbone.gru_hidden_size,
                    )
                    .permute(1, 0, 2)
                    .contiguous()
                    .detach()
                )

                opt.zero_grad()
                log, val, _ = model(mg, mf, mh, dones=mres)

                if is_player_turn:
                    mval = seqs["valid"][idx]
                    masked = log.masked_fill(~mval, -1e8).clamp(
                        min=-1e8, max=1e8
                    )
                    dist = Categorical(logits=masked)
                    nlp = dist.log_prob(mact)
                else:
                    dist = Categorical(logits=log)
                    nlp = dist.log_prob(mact).sum(dim=-1)

                ent = dist.entropy().mean()

                ratio = torch.exp(torch.clamp(nlp - molp, min=-20.0, max=20.0))
                s1 = ratio * madv
                s2 = (
                    torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
                    * madv
                )
                ploss = -torch.min(s1, s2).mean()
                vloss = F.smooth_l1_loss(val.reshape(-1), mret.reshape(-1))

                loss = ploss + cfg.value_coef * vloss - cfg.entropy_coef * ent
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

                tot_p += ploss.item()
                tot_v += vloss.item()
                tot_e += ent.item()
                n_mb += 1

        n = max(1, n_mb)
        return tot_p / n, tot_v / n, tot_e / n

    def train(self):
        self.quit_listener.start()
        self._try_load_models()

        self.logger.log("=" * 60)
        self.logger.log(
            f"Asymmetric Adv PPO | WU: {self.cfg.ghosts_warmup_updates} | Ratio: {self.cfg.ghost_update_ratio}:1"
        )

        start_t = time.time()

        try:
            for upd in range(1, self.cfg.num_updates + 1):
                upd_t = time.time()

                # Asymmetric Schedule
                if upd <= self.cfg.ghosts_warmup_updates:
                    is_player_turn = False
                    phase = "GHOST WARMUP"
                else:
                    # After warmup, alternate.
                    # e.g. Ratio=5 means 1 player turn, 5 ghost turns, repeat.
                    cyc = upd - self.cfg.ghosts_warmup_updates
                    is_player_turn = (
                        cyc % (self.cfg.ghost_update_ratio + 1)
                    ) == 1
                    phase = "PLAYER TURN" if is_player_turn else "GHOST TURN"

                for p in self.player.parameters():
                    p.requires_grad = is_player_turn
                for p in self.ghosts.parameters():
                    p.requires_grad = not is_player_turn

                buf = self.collect_rollout(is_player_turn)
                seqs = self._prepare_seqs(buf, is_player_turn)
                pl, vl, el = self.ppo_update(seqs, is_player_turn)

                # Logging
                if len(self.recent_episodes) > 0:
                    avg_p = sum(
                        e["p_rew"] for e in self.recent_episodes
                    ) / len(self.recent_episodes)
                    avg_g = sum(
                        e["g_rew"] for e in self.recent_episodes
                    ) / len(self.recent_episodes)
                    avg_pct = sum(
                        e["pct"] for e in self.recent_episodes
                    ) / len(self.recent_episodes)
                    avg_d = sum(e["died"] for e in self.recent_episodes) / len(
                        self.recent_episodes
                    )
                else:
                    avg_p, avg_g, avg_pct, avg_d = 0, 0, 0, 0

                self.logger.log(
                    f"U {upd:04d} [{phase}] | "
                    f"Ep {self.total_eps:04d} | "
                    f"P_Rew: {avg_p:6.1f} | G_Rew: {avg_g:6.1f} | "
                    f"Win%: {avg_pct:5.1f}% | Die%: {avg_d*100:5.1f}% | "
                    f"Loss(P/V): {pl:.3f}/{vl:.3f} | Ent: {el:.3f} | "
                    f"Time: {time.time()-start_t:.1f}s"
                )

                if upd % self.cfg.save_interval == 0:
                    torch.save(self.player.state_dict(), self.player_ckpt)
                    torch.save(self.ghosts.state_dict(), self.ghost_ckpt)

                if self.quit_listener.stop_requested:
                    self.logger.log("Quit requested.")
                    break

        except KeyboardInterrupt:
            self.logger.log("Interrupted.")
        finally:
            torch.save(self.player.state_dict(), self.player_ckpt)
            torch.save(self.ghosts.state_dict(), self.ghost_ckpt)
            self.quit_listener.stop()
            self.logger.close()


def main():
    cfg = AdversarialTrainingConfig.from_argv()
    trainer = AdversarialTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
