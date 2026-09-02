"""Collect BFS-expert ghost demonstrations and write them to CNN_DATA.jsonl.

The collector:
  1. Runs PacmanPlayerEnv episodes (full game: maze + player + 4 ghosts).
  2. Controls the player with PacmanExpert (same teacher used for player SL),
     so the player behaves intelligently and the ghosts have a meaningful target.
  3. At every step, calls ObservationFormatter to build the grid / extra_features
     tensors — exactly the format expected by CNNJSONLDataset and ghost_training.py.
  4. Calls GhostExpert to compute the BFS-optimal label for each ghost.
  5. Writes one JSONL record per step to the output file.

Usage
-----
    uv run python -m AI_arena.ghosts.ghost_collector              # 50 000 samples
    uv run python -m AI_arena.ghosts.ghost_collector --samples 20000
    uv run python -m AI_arena.ghosts.ghost_collector --samples 100000 --stage 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch

from AI_arena.data.formatter import ObservationFormatter
from AI_arena.ghosts.ghost_expert import GhostExpert
from AI_arena.player.data.expert import PacmanExpert
from AI_arena.player.player_env import PacmanPlayerEnv

# ── Output path — same directory as the rest of the training data ──
DEFAULT_DATASET_PATH = (
    Path(__file__).parent.parent / "data" / "CNN_DATA.jsonl"
)

# How many steps of a single episode to keep at most (avoids one very long
# episode dominating the dataset when the player expert nearly solves the maze)
MAX_STEPS_PER_EPISODE = 300


# ────────────────────────────────────────────────────────────────────────────
#  Deduplication helpers
# ────────────────────────────────────────────────────────────────────────────

def _record_key(grid: torch.Tensor, extra: torch.Tensor, labels: list[int]) -> bytes:
    """Blake2b fingerprint of (grid, extra, labels) for fast deduplication."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(grid.to(torch.float16).cpu().numpy().tobytes())
    digest.update(extra.cpu().numpy().tobytes())
    for lbl in labels:
        digest.update(lbl.to_bytes(1, "little"))
    return digest.digest()


# ────────────────────────────────────────────────────────────────────────────
#  Ghost state helper
# ────────────────────────────────────────────────────────────────────────────

def _ghost_states(env: Any) -> list[dict[str, Any]]:
    """Extract the ghost-state dicts ObservationFormatter expects."""
    return [
        {
            "grid_x": g.grid_x,
            "grid_y": g.grid_y,
            "is_edible": g.is_edible,
            "direction": g.direction,
            "in_prison": getattr(g, "in_prison", False),
        }
        for g in env.ghosts
    ]


# ────────────────────────────────────────────────────────────────────────────
#  Main collection function
# ────────────────────────────────────────────────────────────────────────────

def collect_demonstrations(
    samples: int,
    output: str | Path = DEFAULT_DATASET_PATH,
    *,
    stage: int = 2,
    seed: int = 42,
    player_horizon: int = 7,
    keep_prison_steps: bool = False,
) -> Path:
    """Run episodes and write ghost SL records to *output*.

    Parameters
    ----------
    samples:
        Total number of JSONL records to write.
    output:
        Destination file path (created / overwritten).
    stage:
        PacmanPlayerEnv stage.  Use stage=2 so ghosts are active from the
        first step (they are imprisoned in stage=1).
    seed:
        RNG seed for reproducibility.
    player_horizon:
        BFS search depth for PacmanExpert (controls player quality).
    keep_prison_steps:
        If True, also save steps where ALL ghosts are imprisoned. These steps
        carry very little ghost information, so False is recommended.
    """
    if samples < 1:
        raise ValueError("samples must be positive")

    rng = random.Random(seed)
    env = PacmanPlayerEnv(seed=seed, stage=stage, device="cpu")
    player_expert = PacmanExpert(horizon=player_horizon)
    ghost_expert = GhostExpert()

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    seen: set[bytes] = set()
    written = 0
    episode_id = 0

    print(f"Collecting {samples} ghost SL samples → {destination}")

    with destination.open("w", encoding="utf-8") as stream:
        while written < samples:
            # ── Periodically reset to get a new maze layout ──
            if written % 50 == 0:
                env.reset()
                
            assert env.maze is not None
            assert env.pellets is not None
            assert env.player is not None
            assert env.movement is not None
            
            maze = env.maze
            height = len(maze)
            width = len(maze[0])
            
            # Find all walkable cells
            walkable_cells = [
                (y, x) for y in range(height) for x in range(width)
                if maze[y][x] != 15
            ]
            if not walkable_cells:
                env.reset()
                continue
                
            # ── Randomize Player ──
            py, px = rng.choice(walkable_cells)
            env.player.grid_y = py
            env.player.grid_x = px
            env.player.direction = rng.choice([0, 1, 2, 3])
            
            # 20% chance of being powered (frightened mode)
            is_powered = rng.random() < 0.2
            env.player.powered_mode = True if is_powered else None
            env.player.powered_timer = 10.0 if is_powered else 0.0

            # ── Randomize Ghosts ──
            for g in env.ghosts:
                gy, gx = rng.choice(walkable_cells)
                g.grid_y = gy
                g.grid_x = gx
                g.direction = rng.choice([0, 1, 2, 3])
                g.in_prison = False
                if is_powered:
                    g.is_edible = rng.choice([True, False])
                else:
                    g.is_edible = False

            # ── Build observation ──
            ghost_state_dicts = _ghost_states(env)
            grid, extra_features, _valid_player, valid_ghost_actions = (
                ObservationFormatter.format_observation(
                    maze=env.maze,
                    pellets=env.pellets,
                    player_pos=(env.player.grid_x, env.player.grid_y),
                    player_direction=env.player.direction,
                    ghost_states=ghost_state_dicts,
                    movement=env.movement,
                    device="cpu",
                )
            )

            # ── Ghost expert labels ──
            decision = ghost_expert.choose_actions(env)
            labels = list(decision.labels)

            # ── Deduplicate ──
            key = _record_key(grid, extra_features, labels)
            if key not in seen:
                seen.add(key)

                record = {
                    "grid": grid[0].tolist(),
                    "extra_features": extra_features[0].tolist(),
                    "valid_actions": valid_ghost_actions.tolist(),
                    "labels": labels,
                    "episode_id": written // 50,
                    "episode_step": written % 50,
                    "maze_width": width,
                    "maze_height": height,
                }
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1

                if written % 500 == 0 or written == samples:
                    pct = 100 * written / samples
                    print(
                        f"  [{pct:5.1f}%] {written}/{samples} samples"
                    )

    print(
        f"\nDone — wrote {written} randomized records to {destination}"
    )
    return destination


# ────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50_000,
        help="Number of JSONL records to write (default: 50 000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Destination .jsonl file",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=2,
        help="PacmanPlayerEnv stage (default: 2 — ghosts active)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Player BFS expert search horizon (default: 7)",
    )
    parser.add_argument(
        "--keep-prison",
        action="store_true",
        help="Also save steps where all ghosts are in prison",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    collect_demonstrations(
        samples=args.samples,
        output=args.output,
        stage=args.stage,
        seed=args.seed,
        player_horizon=args.horizon,
        keep_prison_steps=args.keep_prison,
    )


if __name__ == "__main__":
    main()


