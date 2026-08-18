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


def format_breakdown_line(episodes):
    """Format reward breakdown with consistent old-style naming."""
    if not episodes:
        return ""

    # Aggregate all breakdown keys from episodes
    all_keys = set()
    for ep in episodes:
        all_keys.update(ep.get("episode_reward_breakdown", {}).keys())

    # Map internal keys to display names (old format)
    KEY_MAP = {
        "step": "Step",
        "oscillation": "Osc",
        "pellet": "Pellet",
        "super_pellet": "Super",
        "ghost": "Ghost",
        "complete": "Complete",
        "death": "Death",
        "milestone": "Milestone",
        "bfs": "BFS",
        "ghost_proximity": "GhostProx",
        "region_cleared": "CleanReg",
        "region_dirty": "DirtyReg",
        "backtrack": "Backtrack",
        "incomplete": "Incomplete",
        "predictive_threat": "PredThreat",
        "evasion_skill": "Evasion",
        "super_bait": "SuperBait",
        "zone_control": "ZoneCtrl",
        "threat_mastery": "ThreatMaster",
        "ghost_lure": "GhostLure",
        "survival_truncation": "Survival",
    }

    # Sum each key across episodes
    totals = {}
    for ep in episodes:
        bd = ep.get("episode_reward_breakdown", {})
        for key, display_name in KEY_MAP.items():
            totals[display_name] = totals.get(display_name, 0.0) + bd.get(key, 0.0)

    # Format with fixed-width (old style)
    parts = []
    for display_name in [
        "Step",
        "Osc",
        "Pellet",
        "Super",
        "Ghost",
        "Complete",
        "Death",
        "Milestone",
        "BFS",
        "GhostProx",
        "CleanReg",
        "DirtyReg",
        "Backtrack",
        "Incomplete",
        "PredThreat",
        "Evasion",
        "SuperBait",
        "ZoneCtrl",
        "ThreatMaster",
        "GhostLure",
        "Survival",
    ]:
        val = totals.get(display_name, 0.0)
        parts.append(f"{display_name}: {val:+.1f}")

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
