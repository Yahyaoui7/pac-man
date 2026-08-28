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

# ═══════════════════════════════════════════════════════════════════════════════
# ABBREVIATIONS — purely cosmetic. Add entries here for prettier log lines.
# Any channel not in this dict falls back to its full key name.
#
# This is the ONLY place you need to edit when adding a new reward channel:
#   1. Add the key to the breakdown dict in RewardCalculator.calculate()
#   2. (Optional) Add a label here for a prettier short name
# That's it — format_breakdown_line picks up new channels automatically.
# ═══════════════════════════════════════════════════════════════════════════════
_ABBR: dict[str, str] = {
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
    "exploration": "Explore",
    "zone_stagnation": "ZoneStag",
    "hunger": "Hunger",
    "approach_pellet": "AppPellet",
}

# Minimum absolute magnitude for a channel to be considered "active" (shown in POS/NEG).
# Below this, the channel is shown in the OFF list as dormant.
_MIN_MAG = 0.5


def _abbr(key: str) -> str:
    """Get the short label for a channel key, falling back to the full key."""
    return _ABBR.get(key, key)


def _discover_channels(episodes: list[dict[str, Any]]) -> list[str]:
    """Discover all reward channel keys present in any episode's breakdown.

    This is what makes the formatter future-proof: when you add a new
    channel to the breakdown dict in RewardCalculator, it will be picked
    up here automatically on the next training run — no formatter changes
    needed.
    """
    seen: dict[str, None] = {}  # dict-as-ordered-set (preserves insertion order)
    for ep in episodes:
        bd = ep.get("episode_reward_breakdown", {})
        if not bd:
            continue
        for k in bd.keys():
            if k not in seen:
                seen[k] = None
    return list(seen.keys())


def format_breakdown_line(episodes):
    """Format reward breakdown with dynamic channel discovery.

    ALL reward channels are discovered automatically from the episode data.
    When you add a new channel to the breakdown dict in RewardCalculator,
    it appears in the log on the next run — no formatter changes needed.

    Output format (single line, three sections):
        POS[BFS262(27%) Eva241(25%) Pel87(9%)]=+590
        NEG[Dth1200(58%) GPx653(31%) Osc76(4%) ZC30(1%)]=-1959
        OFF[Stp Hgr Btrk PThr ThM GLur SBat RClr RDrty Incmp ZStg Cmpl Ghst Supr Expl]
        NET=-1369 R=3.32x

    Read as:
      - POS: positive channels that fired this window, sorted by magnitude,
        each with its share of the positive total.
      - NEG: negative channels that fired, sorted by magnitude.
      - OFF: channels that are wired up but contributed < _MIN_MAG this window.
      - NET: signed sum of all channels.
      - R:   |NEG| / |POS| ratio. >1 = penalty-dominated, <1 = reward-dominated.
    """
    if not episodes:
        return ""

    # ── Discover all channel keys dynamically from episode data ──
    all_channels = _discover_channels(episodes)
    if not all_channels:
        return ""

    # ── Aggregate raw values across all episodes in the window ──
    totals: dict[str, float] = {k: 0.0 for k in all_channels}
    for ep in episodes:
        bd = ep.get("episode_reward_breakdown", {})
        if not bd:
            continue
        for k in all_channels:
            try:
                totals[k] += float(bd.get(k, 0.0))
            except (TypeError, ValueError):
                continue

    # ── Split into active (fired) and inactive (dormant) ──
    pos_active: list[tuple[str, float]] = []
    neg_active: list[tuple[str, float]] = []
    inactive: list[str] = []

    for k in all_channels:
        v = totals[k]
        if v > _MIN_MAG:
            pos_active.append((k, v))
        elif v < -_MIN_MAG:
            neg_active.append((k, v))
        else:
            inactive.append(k)

    # Sort active sides by magnitude (largest first)
    pos_active.sort(key=lambda x: -x[1])
    neg_active.sort(key=lambda x: x[1])

    pos_total = sum(v for _, v in pos_active)
    neg_total = sum(v for _, v in neg_active)  # negative number
    net = pos_total + neg_total

    # ── Format each section ──
    def _fmt_active(items: list[tuple[str, float]], total: float) -> str:
        if not items or abs(total) < 1e-6:
            return "—"
        parts = []
        for k, v in items:
            abbr = _abbr(k)
            pct = (abs(v) / abs(total)) * 100.0
            parts.append(f"{abbr}{abs(v):.0f}({pct:.0f}%)")
        return " ".join(parts)

    def _fmt_inactive(keys: list[str]) -> str:
        if not keys:
            return ""
        return " ".join(_abbr(k) for k in keys)

    pos_str = _fmt_active(pos_active, pos_total)
    neg_str = _fmt_active(neg_active, neg_total)
    inactive_str = _fmt_inactive(inactive)

    # Ratio: |NEG| / |POS|. >1 = penalty-dominated, <1 = reward-dominated.
    ratio = abs(neg_total) / max(pos_total, 1.0)

    parts = [
        f"POS[{pos_str}]={pos_total:+.0f}",
        f"NEG[{neg_str}]={neg_total:+.0f}",
    ]
    if inactive_str:
        parts.append(f"OFF[{inactive_str}]")
    parts.append(f"NET={net:+.0f}")
    parts.append(f"R={ratio:.2f}x")

    return " ".join(parts)


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
                100.0 * float(ep.get("steps", 0)) / float(ep.get("max_steps", 0))
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
    esc = f"{stats['escape_rate'] * 100:3.0f}%" if stats["escape_rate"] >= 0 else " n/a"
    share = (
        f"{stats['cornered_death_share'] * 100:3.0f}%"
        if stats["cornered_death_share"] >= 0
        else "n/a"
    )
    mind = stats["avg_min_ghost_dist"]
    min_d_str = f"{mind:5.2f}" if mind >= 0 else "  n/a"
    if stats["avg_life_pct"] >= 0:
        life_str = f"{stats['avg_steps_lived']:5.0f}mv ({stats['avg_life_pct']:4.1f}%)"
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
