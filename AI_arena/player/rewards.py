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

        # ── Context-aware oscillation penalty ──
        if events.get("oscillating", False) and not (
            events["pellet_eaten"] or events["super_pellet_eaten"]
        ):
            threat_dist = float("inf")
            if (
                self.stage > 1
                and movement is not None
                and player is not None
                and maze is not None
            ):
                py, px = player.grid_y, player.grid_x
                bfs = movement.bfs_distances((py, px))
                w = len(maze[0]) if maze else 1
                for ghost in ghosts:
                    if ghost.in_prison or ghost.is_edible:
                        continue
                    idx = ghost.grid_y * w + ghost.grid_x
                    if 0 <= idx < len(bfs) and bfs[idx] >= 0:
                        threat_dist = min(threat_dist, bfs[idx])

            if threat_dist > 5 and player.powered_timer <= 0:
                breakdown["oscillation"] = OSCILLATION_REWARD

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

        # ── Ghost proximity: avoid hunters, chase edible ──
        if (
            self.stage > 1
            and movement is not None
            and player is not None
            and maze is not None
        ):
            py, px = player.grid_y, player.grid_x
            bfs = movement.bfs_distances((py, px))
            w = len(maze[0]) if maze else 1
            powered = player.powered_timer > 0

            for ghost in ghosts:
                if ghost.in_prison:
                    continue
                idx = ghost.grid_y * w + ghost.grid_x
                if not (0 <= idx < len(bfs)):
                    continue
                d = bfs[idx]
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
                    # AVOID MODE
                    if d == 1:
                        breakdown["ghost_proximity"] -= 3.0
                    elif d == 2:
                        breakdown["ghost_proximity"] -= 1.5
                    elif d == 3:
                        breakdown["ghost_proximity"] -= 0.5

        breakdown["bfs"] = 2 * bfs_shaping
        return sum(breakdown.values()), breakdown
