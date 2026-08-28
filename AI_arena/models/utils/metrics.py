"""Metric computation and formatting utilities for training breakdown logs."""

from __future__ import annotations

from typing import Any, Sequence

BD_LABELS = {
    "step": "Step",
    "oscillation": "Osc",
    "pellet": "Pellet",
    "super_pellet": "Super",
    "ghost": "Ghost",
    "complete": "Complete",
    "death": "Death",
    "bfs": "BFS",
    "ghost_proximity": "GhostProx",
    "region_cleared": "CleanReg",
    "region_dirty": "DirtyReg",
}


def format_breakdown_line(recent_episodes: Sequence[dict[str, Any]]) -> str:
    """Return a compact 'Key: +X.X' string averaged over the last window."""
    if not recent_episodes:
        return " | ".join(f"{label}: +0.0" for label in BD_LABELS.values())

    parts = []
    for key, label in BD_LABELS.items():
        avg = sum(
            ep["episode_reward_breakdown"].get(key, 0.0) for ep in recent_episodes
        ) / len(recent_episodes)
        parts.append(f"{label}: {avg:+.1f}")
    return " | ".join(parts)


def compute_positive_stats(
    recent_episodes: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Compute average positive reward contributions."""
    if not recent_episodes:
        return {
            "new_tile": 0.0,
            "pellet": 0.0,
            "super_pellet": 0.0,
            "ghost": 0.0,
            "complete": 0.0,
        }
    n = len(recent_episodes)
    return {
        "new_tile": sum(
            ep["episode_reward_breakdown"].get("new_tile", 0.0)
            for ep in recent_episodes
        )
        / n,
        "pellet": sum(
            ep["episode_reward_breakdown"].get("pellet", 0.0) for ep in recent_episodes
        )
        / n,
        "super_pellet": sum(
            ep["episode_reward_breakdown"].get("super_pellet", 0.0)
            for ep in recent_episodes
        )
        / n,
        "ghost": sum(
            ep["episode_reward_breakdown"].get("ghost", 0.0) for ep in recent_episodes
        )
        / n,
        "complete": sum(
            ep["episode_reward_breakdown"].get("complete", 0.0)
            for ep in recent_episodes
        )
        / n,
    }


def compute_negative_stats(
    recent_episodes: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Compute average negative penalty contributions."""
    if not recent_episodes:
        return {"step": 0.0, "oscillation": 0.0, "death": 0.0, "bfs": 0.0}
    n = len(recent_episodes)
    return {
        "step": sum(
            ep["episode_reward_breakdown"].get("step", 0.0) for ep in recent_episodes
        )
        / n,
        "oscillation": sum(
            ep["episode_reward_breakdown"].get("oscillation", 0.0)
            for ep in recent_episodes
        )
        / n,
        "death": sum(
            ep["episode_reward_breakdown"].get("death", 0.0) for ep in recent_episodes
        )
        / n,
        "bfs": sum(
            ep["episode_reward_breakdown"].get("bfs", 0.0) for ep in recent_episodes
        )
        / n,
    }
