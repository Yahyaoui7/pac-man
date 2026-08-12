"""Reward calculation for Pac-Man RL environment."""

from __future__ import annotations

from AI_arena.player.constants import (
    COMPLETION_REWARD,
    DEATH_REWARD,
    EAT_GHOST_REWARD,
    MILESTONE_REWARDS,
    OSCILLATION_REWARD,
    PELLET_REWARD,
    STEP_REWARD,
    SUPER_PELLET_REWARD,
)


class RewardCalculator:
    """Computes step rewards and maintains milestone state."""

    def __init__(self, stage: int) -> None:
        self.stage = stage
        self.milestones_hit: set[float] = set()

    def reset(self) -> None:
        self.milestones_hit = set()

    def calculate(
        self,
        *,
        events: dict[str, bool],
        bfs_shaping: float,
        total_pellets: int,
        remaining_pellets: int,
        step_count: int,
        max_steps: int,
        player,
        ghosts: list,
        movement,
        maze: list[list[int]] | None,
        threat_dist: float = float("inf"),
        min_ghost_dist_after: int = -1,
        min_ghost_dist_before: int = -1,
    ) -> tuple[float, dict[str, float]]:
        """Return (total_reward, breakdown_dict)."""
        breakdown = {
            "step": STEP_REWARD,
            "oscillation": 0.0,
            "pellet": 0.0,
            "super_pellet": 0.0,
            "ghost": 0.0,
            "complete": 0.0,
            "death": 0.0,
            "milestone": 0.0,
            "bfs": 0.0,
            "ghost_proximity": 0.0,
            "region_cleared": 0.0,
            "region_dirty": 0.0,
            "backtrack": 0.0,
            "incomplete": 0.0,
        }

        eaten = total_pellets - remaining_pellets
        frac = eaten / total_pellets if total_pellets > 0 else 0.0

        # ── Milestones ──
        for threshold, reward in MILESTONE_REWARDS.items():
            if frac >= threshold and threshold not in self.milestones_hit:
                self.milestones_hit.add(threshold)
                breakdown["milestone"] += reward

        # ── Progressive pellet bonus ──
        pellet_bonus = 1.0
        if frac >= 0.9:
            pellet_bonus = 4.0
        elif frac >= 0.75:
            pellet_bonus = 2.0
        elif frac >= 0.6:
            pellet_bonus = 1.5

        # ── Oscillation: punish even near ghosts (halved), never fully disable ──
        if events.get("oscillating", False) and not (
            events["pellet_eaten"] or events["super_pellet_eaten"]
        ):
            if player is None or player.powered_timer <= 0:
                # Halve the penalty when near a ghost (threat_dist ≤ 5) instead of
                # disabling it entirely — prevents infinite flapping when scared
                scale = 0.4 if threat_dist <= 5 else 1.0
                breakdown["oscillation"] = OSCILLATION_REWARD * scale

        # ── Mild anti-backtrack (coverage inefficiency) ──
        if events.get("backtracked", False):
            if threat_dist > 5 and (player is None or player.powered_timer <= 0):
                breakdown["backtrack"] = -0.2

        # ── Region bonus / penalty ──
        if events.get("cleared_region", False):
            breakdown["region_cleared"] = +2.0  # ← small reward for clearing

        # ← CHANGED: tiny penalty for stragglers — only once model knows how to clear
        if events.get("left_dirty_region", False):
            if threat_dist > 6 and (player is None or player.powered_timer <= 0):
                breakdown["region_dirty"] = -1.0

        # ← CHANGED: gentle incomplete penalty. Don't murder the agent for trying.
        if events.get("truncated", False) and remaining_pellets > 0:
            breakdown["incomplete"] = -0.2 * remaining_pellets

        # ── Pellet & Event Rewards ──
        if events["pellet_eaten"]:
            breakdown["pellet"] = PELLET_REWARD + 3.0 * frac * pellet_bonus

        if events["super_pellet_eaten"]:
            breakdown["super_pellet"] = SUPER_PELLET_REWARD

        if events.get("ghost_eaten", False):
            breakdown["ghost"] = EAT_GHOST_REWARD

        if events["level_completed"]:
            remaining = max(0, max_steps - step_count)
            breakdown["complete"] = COMPLETION_REWARD + float(remaining)

        if events["pacman_died"]:
            breakdown["death"] = DEATH_REWARD

        # ── Ghost proximity ──
        if self.stage > 1 and player is not None and min_ghost_dist_after >= 0:
            powered = player.powered_timer > 0
            d = min_ghost_dist_after

            has_edible_nearby = any(not g.in_prison and g.is_edible for g in ghosts)
            has_threat_nearby = any(not g.in_prison and not g.is_edible for g in ghosts)

            if powered and has_edible_nearby:
                if d == 1:
                    breakdown["ghost_proximity"] += 10.0
                elif d == 2:
                    breakdown["ghost_proximity"] += 5.0
                elif d == 3:
                    breakdown["ghost_proximity"] += 2.0
            elif has_threat_nearby and not powered:
                if d == 1:
                    breakdown["ghost_proximity"] -= 4.0
                elif d == 2:
                    breakdown["ghost_proximity"] -= 1.5
                elif d == 3:
                    breakdown["ghost_proximity"] -= 0.3

                # Approaching a very close ghost: proportional penalty (was -10, too large)
                if min_ghost_dist_before > 0 and d < min_ghost_dist_before and d <= 2:
                    breakdown["ghost_proximity"] -= 2.0

                # Evasion success: reward agent for increasing distance from a close ghost
                if min_ghost_dist_before > 0 and d > min_ghost_dist_before and min_ghost_dist_before <= 3:
                    breakdown["ghost_proximity"] += 1.5

        breakdown["bfs"] = 0.5 * bfs_shaping
        return sum(breakdown.values()), breakdown
