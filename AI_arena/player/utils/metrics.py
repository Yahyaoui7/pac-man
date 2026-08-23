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
    """Format reward breakdown with consistent old-style naming.

    Sparse output: components whose window total is (near) zero are omitted,
    so adding/removing reward terms never bloats the log — only influential
    terms appear.
    """
    if not episodes:
        return ""

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

    # Fixed order, but only non-zero terms (|value| > epsilon)
    parts = [
        f"{display_name}: {totals.get(display_name, 0.0):+.1f}"
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
        ]
        if abs(totals.get(display_name, 0.0)) > 1e-9
    ]

    return " | ".join(parts)


def compute_survival_stats(episodes: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Aggregate danger/trap telemetry across finished episodes.

    Leading indicators for trap-avoidance learning — these move thousands of
    episodes before win-rate or pellet% do:
      - cornered exposure should trend DOWN
      - escape_rate should trend UP (exposure is luck, escaping is the skill)
      - cornered_death_share should trend DOWN
      - avg_min_ghost_dist should trend UP, approach_pct DOWN
    """
    zeros = {
        "episodes": 0.0,
        "death_rate": 0.0,
        "avg_steps_lived": 0.0,
        "avg_life_pct": -1.0,
        "cornered_steps_per_ep": 0.0,
        "cornered_exposure_pct": 0.0,
        "entries_per_ep": 0.0,
        "escape_rate": -1.0,
        "escape_samples": 0,
        "cornered_death_share": -1.0,
        "cornered_deaths": 0.0,
        "avg_min_ghost_dist": -1.0,
        "approach_pct": 0.0,
    }
    if not episodes:
        return zeros

    def tsum(key: str) -> float:
        return sum(float(ep.get("telemetry", {}).get(key, 0.0)) for ep in episodes)

    n = len(episodes)
    total_steps = sum(max(1.0, float(ep.get("steps", 1))) for ep in episodes)
    total_deaths = sum(
        1 for ep in episodes if ep.get("episode_event_counts", {}).get("died", 0) > 0
    )
    cornered_steps = tsum("cornered_steps")
    esc_success = tsum("escape_success")
    esc_failure = tsum("escape_failure")
    cornered_deaths = tsum("deaths_cornered")
    md_sum = tsum("min_ghost_dist_sum")
    md_cnt = tsum("min_ghost_dist_cnt")

    return {
        "episodes": float(n),
        "death_rate": total_deaths / n,
        "avg_steps_lived": sum(float(ep.get("steps", 0)) for ep in episodes) / n,
        "avg_life_pct": (
            sum(
                100.0
                * float(ep.get("steps", 0))
                / float(ep.get("max_steps", 0))
                for ep in episodes
                if float(ep.get("max_steps", 0)) > 0
            )
            / max(1, sum(1 for ep in episodes if float(ep.get("max_steps", 0)) > 0))
            if any(float(ep.get("max_steps", 0)) > 0 for ep in episodes)
            else -1.0
        ),
        "cornered_steps_per_ep": cornered_steps / n,
        "cornered_exposure_pct": 100.0 * cornered_steps / total_steps,
        "entries_per_ep": tsum("cornered_entries") / n,
        "escape_rate": (
            esc_success / (esc_success + esc_failure)
            if (esc_success + esc_failure) > 0
            else -1.0
        ),
        "escape_samples": int(esc_success + esc_failure),
        "cornered_death_share": (
            cornered_deaths / total_deaths if total_deaths > 0 else -1.0
        ),
        "cornered_deaths": cornered_deaths,
        "avg_min_ghost_dist": md_sum / md_cnt if md_cnt > 0 else -1.0,
        "approach_pct": 100.0 * tsum("approach_steps") / total_steps,
    }


def format_survival_line(stats: dict[str, float]) -> str:
    """One-line summary of survival stats for the training log."""
    esc = (
        f"{stats['escape_rate'] * 100:3.0f}%"
        if stats["escape_rate"] >= 0
        else " n/a"
    )
    share = (
        f"{stats['cornered_death_share'] * 100:3.0f}%"
        if stats["cornered_death_share"] >= 0
        else "n/a"
    )
    mind = stats["avg_min_ghost_dist"]
    min_d_str = f"{mind:5.2f}" if mind >= 0 else "  n/a"
    if stats["avg_life_pct"] >= 0:
        life_str = (
            f"{stats['avg_steps_lived']:5.0f}mv ({stats['avg_life_pct']:4.1f}%)"
        )
    else:
        life_str = "        n/a"
    return (
        f"Life: {life_str} | "
        f"Death: {stats['death_rate'] * 100:5.1f}% | "
        f"Corn: {stats['cornered_steps_per_ep']:4.1f}/ep "
        f"({stats['cornered_exposure_pct']:4.1f}%) | "
        f"Esc: {esc} [{stats['escape_samples']:3d}] | "
        f"CDth: {share} | "
        f"MinD: {min_d_str}"
    )


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
