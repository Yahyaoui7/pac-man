"""Fixed-seed evaluation harness for the Pac-Man player policy.

Runs the (greedy) policy on an identical benchmark set of mazes every time,
so any two checkpoints can be compared head-to-head without the noise of
training-time sampling and random maze mixes.

Usage (from repo root):
    python -m AI_arena.player.utils.evaluate                     # eval latest best
    python -m AI_arena.player.utils.evaluate --compare           # table of past evals
    python -m AI_arena.player.utils.evaluate \
        --checkpoint AI_arena/models/player_rl_stage2.pt --episodes 30

Every run appends a record to AI_arena/evals/eval_history.json so training
runs and manual benchmarks share one comparable timeline.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.distributions import Categorical

from AI_arena.models.cnn_player import PlayerActorCritic, load_checkpoint_into_policy
from AI_arena.player.player_env import PacmanPlayerEnv
from AI_arena.player.utils.metrics import compute_survival_stats

DEFAULT_SEED_BASE = 10000  # identical benchmark mazes for every evaluation
EVAL_SCORE_COMPLETE_W = 40.0  # weight of completion rate (in [0,1])
EVAL_SCORE_DEATH_W = 30.0  # weight of death rate (in [0,1])

HISTORY_PATH = Path(__file__).parents[2] / "evals" / "eval_history.json"


def eval_score(
    avg_pellet_pct: float, completion_rate: float, death_rate: float
) -> float:
    """Composite benchmark score — higher is better.

    score = avg_pellet_pct + 40 * completion_rate − 30 * death_rate
    Defined once here so every eval in the history is directly comparable.
    """
    return (
        avg_pellet_pct
        + EVAL_SCORE_COMPLETE_W * completion_rate
        - EVAL_SCORE_DEATH_W * death_rate
    )


@torch.no_grad()
def run_evaluation(
    policy: PlayerActorCritic,
    device: str | torch.device = "cpu",
    stage: int = 2,
    episodes: int = 20,
    seed_base: int = DEFAULT_SEED_BASE,
    greedy: bool = True,
    env: PacmanPlayerEnv | None = None,
    ghost_speed_ratio: float = 0.35,
    ghost_confusion_prob: float = 0.0,
) -> dict[str, Any]:
    """Run `episodes` deterministic episodes; returns aggregate metrics.

    A fresh `PacmanPlayerEnv` is created unless one is passed in (pass a
    dedicated instance when calling from the training loop — never the
    rollout env, reseeding it would corrupt its RNG stream).
    """
    was_training = policy.training
    policy.eval()
    if env is None:
        env = PacmanPlayerEnv(
            seed=seed_base,
            stage=stage,
            device="cpu",
            ghost_speed_ratio=ghost_speed_ratio,
            ghost_confusion_prob=ghost_confusion_prob,
        )

    device_t = torch.device(device)
    episode_records: list[dict[str, Any]] = []

    for ep_idx in range(episodes):
        seed = seed_base + ep_idx
        env.set_seed(seed)
        obs = env.reset()
        hidden: torch.Tensor | None = None
        total_reward = 0.0
        steps = 0
        info: dict[str, Any] = {}

        done = False
        while not done:
            grid, features, valid_actions = obs
            logits, _, hidden = policy(grid.to(device_t), features.to(device_t), hidden)
            masked_logits = logits.masked_fill(~valid_actions.to(device_t), -1e8)
            masked_logits = torch.nan_to_num(
                masked_logits, nan=-1e8, posinf=10.0, neginf=-1e8
            )
            if greedy:
                action = int(torch.argmax(masked_logits, dim=-1).item())
            else:
                action = int(Categorical(logits=masked_logits).sample())
            obs, reward, done, info, _ = env.step(action)
            if info.get("events", {}).get("pacman_died", False):
                hidden = None
            total_reward += reward
            steps += 1

        events = info.get("episode_event_counts", {})
        episode_records.append(
            {
                "seed": seed,
                "maze": info["maze"],
                "reward": total_reward,
                "steps": steps,
                "max_steps": float(info.get("max_steps", 0)),
                "pellets": float(info["pellets_eaten"]),
                "pct": float(info["completion_pct"]),
                "completed": bool(events.get("completed", 0)),
                "died": bool(events.get("died", 0)),
                "truncated": bool(events.get("truncated", 0)),
                "episode_event_counts": dict(events),
                "telemetry": dict(info.get("telemetry", {})),
            }
        )

    if was_training:
        policy.train()

    n = len(episode_records)
    avg_pellet_pct = sum(r["pct"] for r in episode_records) / n
    completion_rate = sum(r["completed"] for r in episode_records) / n
    death_rate = sum(r["died"] for r in episode_records) / n
    truncation_rate = sum(r["truncated"] for r in episode_records) / n
    avg_reward = sum(r["reward"] for r in episode_records) / n

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env": {
            "stage": stage,
            # Provenance: scores are only comparable within identical game
            # parameters — a difficulty change invalidates the whole history.
            "ghost_confusion": round(
                float(getattr(env, "ghost_confusion_prob", -1.0)), 3
            ),
        },
        "stage": stage,
        "episodes": n,
        "seed_base": seed_base,
        "greedy": greedy,
        "avg_reward": avg_reward,
        "avg_pellet_pct": avg_pellet_pct,
        "completion_rate": completion_rate,
        "death_rate": death_rate,
        "truncation_rate": truncation_rate,
        "eval_score": eval_score(avg_pellet_pct, completion_rate, death_rate),
        "survival": compute_survival_stats(episode_records),
        "episodes_detail": episode_records,
    }


# ─── History persistence ─────────────────────────────────────────────────────


def append_history(record: dict[str, Any], path: Path = HISTORY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = path.with_suffix(".corrupt.json")
            path.rename(backup)
            print(f"WARNING: unreadable history moved to {backup}")
    history.append(record)
    path.write_text(json.dumps(history), encoding="utf-8")
    return path


def load_history(path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def print_comparison(history: list[dict[str, Any]], last: int = 12) -> None:
    rows = history[-last:]
    header = (
        f"{'when':<17} {'ckpt':<28} {'upd':>5} {'score':>7} "
        f"{'pellet%':>8} {'comp%':>6} {'death%':>7} {'esc%':>5} {'corn/ep':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        s = r.get("survival", {})
        esc = (
            f"{s['escape_rate'] * 100:.0f}" if s.get("escape_rate", -1) >= 0 else "n/a"
        )
        print(
            f"{r.get('timestamp', '?'):<17} "
            f"{r.get('checkpoint', '?'):<28} "
            f"{str(r.get('update', '-')):>5} "
            f"{r['eval_score']:>7.1f} "
            f"{r['avg_pellet_pct']:>8.1f} "
            f"{r['completion_rate'] * 100:>6.1f} "
            f"{r['death_rate'] * 100:>7.1f} "
            f"{esc:>5} "
            f"{s.get('cornered_steps_per_ep', 0.0):>8.2f}"
        )


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _resolve_checkpoint(stage: int, requested: str | None) -> Path:
    if requested:
        p = Path(requested)
        if not p.exists():
            raise SystemExit(f"ERROR: checkpoint not found: {p}")
        return p
    models_dir = Path(__file__).parents[2] / "models"
    for name in (f"player_rl_stage{stage}_best.pt", f"player_rl_stage{stage}.pt"):
        p = models_dir / name
        if p.exists():
            return p
    raise SystemExit(f"ERROR: no checkpoint found in {models_dir} for stage {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--sample", action="store_true", help="Sample instead of argmax"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Print history table after evaluating"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not append to eval_history.json"
    )
    args = parser.parse_args()

    ckpt_path = _resolve_checkpoint(args.stage, args.checkpoint)
    policy = PlayerActorCritic().to(args.device)
    if not load_checkpoint_into_policy(
        policy, ckpt_path, device=torch.device(args.device)
    ):
        raise SystemExit(f"ERROR: could not load weights from {ckpt_path}")

    print(
        f"Evaluating {ckpt_path.name} | stage {args.stage} | "
        f"{args.episodes} eps @ seeds {args.seed_base}..{args.seed_base + args.episodes - 1} "
        f"({'sample' if args.sample else 'greedy'})"
    )
    t0 = time.time()
    result = run_evaluation(
        policy,
        device=args.device,
        stage=args.stage,
        episodes=args.episodes,
        seed_base=args.seed_base,
        greedy=not args.sample,
    )
    result["checkpoint"] = ckpt_path.name

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(
        f"  eval_score : {result['eval_score']:.1f}   "
        "(pellet% + 40*comp_rate - 30*death_rate)"
    )
    print(f"  pellet     : {result['avg_pellet_pct']:.1f}%")
    print(f"  completed  : {result['completion_rate'] * 100:.1f}%")
    print(f"  died       : {result['death_rate'] * 100:.1f}%")
    print(f"  truncated  : {result['truncation_rate'] * 100:.1f}%")
    s = result["survival"]
    esc = f"{s['escape_rate'] * 100:.0f}%" if s["escape_rate"] >= 0 else "n/a"
    life = (
        f"{s['avg_steps_lived']:.0f} moves ({s['avg_life_pct']:.1f}% of budget)"
        if s["avg_life_pct"] >= 0
        else f"{s['avg_steps_lived']:.0f} moves"
    )
    print(f"  survived   : {life}")
    print(
        f"  cornered   : {s['cornered_steps_per_ep']:.2f} steps/ep "
        f"(entries {s['entries_per_ep']:.2f}/ep)"
    )
    print(f"  escapes    : {esc} over {s['escape_samples']} attempts")
    print(
        f"  corner deaths: {s['cornered_deaths']:.0f} "
        f"({s['cornered_death_share'] * 100 if s['cornered_death_share'] >= 0 else 0:.0f}% of deaths)"
    )
    print(
        f"  min ghost dist: {s['avg_min_ghost_dist']:.2f} | approach: {s['approach_pct']:.1f}% of steps"
    )

    if not args.no_save:
        out = append_history(result)
        print(f"\nAppended to {out}")
    if args.compare:
        history = load_history()
        if history:
            print(f"\n=== Eval history ({len(history)} records, last 12) ===")
            print_comparison(history)


if __name__ == "__main__":
    main()
