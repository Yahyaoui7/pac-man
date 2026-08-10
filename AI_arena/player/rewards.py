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
            "abandon_pellet": 0.0,
            "region_dirty": 0.0,
            "region_cleared": 0.0,
            "circular_loop": 0.0,
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

        # ── Context-aware oscillation penalty (2-cell A->B->A) ──
        if events.get("oscillating", False) and not (
            events["pellet_eaten"] or events["super_pellet_eaten"]
        ):
            if threat_dist > 5 and (player is None or player.powered_timer <= 0):
                breakdown["oscillation"] = OSCILLATION_REWARD

        # ── Zero-Pellet Circular Loop Penalty (2x2 / 3-cell squares) ──
        if events.get("circular_loop", False):
            if threat_dist > 4 and (player is None or player.powered_timer <= 0):
                breakdown["circular_loop"] = -4.0

        # ── Close-Pellet Abandonment Penalty ──
        if events.get("abandoned_close_pellet", False):
            if threat_dist > 3 and (player is None or player.powered_timer <= 0):
                breakdown["abandon_pellet"] = -3.0

        # ── Region-Leaving Penalty & Cleared Bonus ──
        if events.get("left_dirty_region", False):
            if threat_dist > 4 and (player is None or player.powered_timer <= 0):
                breakdown["region_dirty"] = -5.0

        if events.get("cleared_region", False):
            breakdown["region_cleared"] = +5.0

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

        # ── Ghost proximity & Suicidal Avoidance ──
        if (
            self.stage > 1
            and movement is not None
            and player is not None
            and maze is not None
        ):
            powered = player.powered_timer > 0

            for ghost in ghosts:
                if ghost.in_prison:
                    continue
                d = min_ghost_dist_after
                if d < 0:
                    continue

                if powered and ghost.is_edible:
                    # HUNT MODE: reward closing in on edible ghosts
                    if d == 1:
                        breakdown["ghost_proximity"] += 10.0
                    elif d == 2:
                        breakdown["ghost_proximity"] += 5.0
                    elif d == 3:
                        breakdown["ghost_proximity"] += 2.0
                elif not ghost.is_edible:
                    # AVOID MODE: Severe penalties when dangerously close
                    if d == 1:
                        breakdown["ghost_proximity"] -= 12.0
                    elif d == 2:
                        breakdown["ghost_proximity"] -= 4.0
                    elif d == 3:
                        breakdown["ghost_proximity"] -= 1.5

                    # Suicidal move check: stepping closer to a non-edible ghost when d <= 2
                    if min_ghost_dist_before > 0 and d < min_ghost_dist_before and d <= 2:
                        breakdown["ghost_proximity"] -= 15.0

        breakdown["bfs"] = 2 * bfs_shaping
        return sum(breakdown.values()), breakdown
