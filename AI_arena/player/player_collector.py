"""Collect risk-aware expert demonstrations for supervised Pac-Man training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from AI_arena.player.expert import PacmanExpert
from AI_arena.player.observation import format_player_observation
from AI_arena.player.player_env import PacmanPlayerEnv

DEFAULT_DATASET_PATH = (
    Path(__file__).parent.parent / "data" / "PACMAN_IMITATION_DATA.jsonl"
)


def _state_key(grid: torch.Tensor, features: torch.Tensor, label: int) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(grid.to(torch.uint8).cpu().numpy().tobytes())
    digest.update(features.cpu().numpy().tobytes())
    digest.update(bytes([label]))
    return digest.digest()


def collect_demonstrations(
    samples: int,
    output: str | Path = DEFAULT_DATASET_PATH,
    *,
    stage: int = 2,
    seed: int = 42,
    horizon: int = 7,
    keep_forced_probability: float = 0.1,
    max_kept_per_episode: int = 100,
) -> Path:
    if samples < 1:
        raise ValueError("samples must be positive")
    rng = random.Random(seed)
    env = PacmanPlayerEnv(seed=seed, stage=stage, device="cpu")
    expert = PacmanExpert(horizon=horizon)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    seen: set[bytes] = set()
    written = 0
    episode_id = 0

    with destination.open("w", encoding="utf-8") as stream:
        while written < samples:
            env.reset()
            episode_step = 0
            kept_this_episode = 0
            done = False
            while (
                not done
                and written < samples
                and kept_this_episode < max_kept_per_episode
            ):
                assert env.maze is not None and env.pellets is not None
                assert env.player is not None and env.movement is not None
                grid, features, valid_actions = format_player_observation(
                    maze=env.maze,
                    pellets=env.pellets,
                    player=env.player,
                    ghosts=env.ghosts,
                    movement=env.movement,
                    initial_pellet_count=env.total_pellets,
                    device="cpu",
                )
                decision = expert.choose_action(env)
                legal_count = int(valid_actions.sum().item())
                keep = legal_count > 1 or rng.random() < keep_forced_probability
                key = _state_key(grid, features, decision.action)
                if keep and key not in seen:
                    seen.add(key)
                    record = {
                        "schema_version": 1,
                        "grid": grid[0].tolist(),
                        "extra_features": features[0].tolist(),
                        "valid_actions": valid_actions[0].tolist(),
                        "label": decision.action,
                        "teacher_scores": list(decision.scores),
                        "episode_id": episode_id,
                        "episode_step": episode_step,
                        "maze_width": len(env.maze[0]),
                        "maze_height": len(env.maze),
                        "stage": stage,
                    }
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    written += 1
                    kept_this_episode += 1
                    if written % 100 == 0 or written == samples:
                        print(f"Collected {written}/{samples} expert samples")
                _, _, done, _ = env.step(decision.action)
                episode_step += 1
            episode_id += 1
    print(
        f"Saved {written} demonstrations from {episode_id} episodes "
        f"to {destination}"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    collect_demonstrations(
        args.samples,
        args.output,
        stage=args.stage,
        seed=args.seed,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()
