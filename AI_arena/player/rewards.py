"""Reward calculation for Pac-Man RL environment — Survival-First Mode."""

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
    """Computes step rewards — modular survival-first mode."""

    def __init__(self, stage: int) -> None:
        self.stage = stage

        # ── Active state ──
        self.current_block: tuple[int, int] | None = None
        self.steps_in_block: int = 0
        self.visited_this_episode: set[tuple[int, int]] = set()

        # ── Legacy / prepared state (for disabled methods) ──
        self.milestones_hit: set[float] = set()
        self.threat_history: list[tuple[int, int]] = []
        self.consecutive_threat_steps: int = 0
        self.in_danger_zone: bool = False
        self.danger_zone_entry_step: int = 0
        self.danger_zone_entry_pos: tuple[int, int] | None = None
        self.last_min_ghost_dist: int = -1
        self.last_action_was_toward_ghost: bool = False
        self.steps_in_corner: int = 0
        self.osc_streak: int = 0

    # ═══════════════════════════════════════════════════════════════════
    #  RESET
    # ═══════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        self.current_block = None
        self.steps_in_block = 0
        self.visited_this_episode = set()

        self.milestones_hit = set()
        self.threat_history = []
        self.consecutive_threat_steps = 0
        self.in_danger_zone = False
        self.danger_zone_entry_step = 0
        self.danger_zone_entry_pos = None
        self.last_min_ghost_dist = -1
        self.last_action_was_toward_ghost = False
        self.steps_in_corner = 0
        self.osc_streak = 0

    # ═══════════════════════════════════════════════════════════════════
    #  ACTIVE REWARD METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _death_penalty(
        self, events: dict[str, bool], breakdown: dict[str, float]
    ) -> None:
        """Penalty for dying."""
        if events.get("pacman_died", False):
            breakdown["death"] = DEATH_REWARD

    def _zone_stagnation_penalty(
        self, px: int, py: int, breakdown: dict[str, float]
    ) -> None:
        """Gentle flat penalty for lingering in the same 3×3 block too long."""
        block = (py // 3, px // 3)
        if block == self.current_block:
            self.steps_in_block += 1
            if self.steps_in_block == 12:
                breakdown["zone_stagnation"] = -5.0
            elif self.steps_in_block > 12:
                breakdown["zone_stagnation"] = -0.5
        else:
            self.current_block = block
            self.steps_in_block = 1

    def _exploration_reward(
        self, px: int, py: int, breakdown: dict[str, float]
    ) -> None:
        """Reward for visiting a brand-new tile this episode."""
        cell = (py, px)
        if cell not in self.visited_this_episode:
            self.visited_this_episode.add(cell)
            breakdown["exploration"] = 0.1

    def _ghost_eat_reward(
        self, events: dict[str, bool], breakdown: dict[str, float]
    ) -> None:
        """Reward for eating a ghost."""
        if events.get("ghost_eaten", False):
            breakdown["ghost"] = EAT_GHOST_REWARD

    def _completion_reward(
        self,
        events: dict[str, bool],
        step_count: int,
        max_steps: int,
        breakdown: dict[str, float],
    ) -> None:
        """Bonus for clearing the level."""
        if events.get("level_completed", False):
            remaining = max(0, max_steps - step_count)
            breakdown["complete"] = COMPLETION_REWARD + min(
                float(remaining) * 0.1, 100.0
            )

    # ═══════════════════════════════════════════════════════════════════
    #  BALANCED REWARD METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _step_reward(
        self, events: dict[str, bool], breakdown: dict[str, float]
    ) -> None:
        """No step tax on pellet/super steps — progress should feel good."""
        if events.get("pellet_eaten") or events.get("super_pellet_eaten"):
            breakdown["step"] = 0.0
        else:
            breakdown["step"] = STEP_REWARD  # keep at -0.1 or raise to -0.2

    def _hunger_penalty(
        self, steps_since_pellet: int, breakdown: dict[str, float]
    ) -> None:
        grace = 40
        if steps_since_pellet > grace:
            breakdown["hunger"] = -0.3

    def _pellet_reward(
        self, events: dict[str, bool], frac: float, breakdown: dict[str, float]
    ) -> None:
        additional = 0
        if events.get("pellet_eaten", False):
            if frac > 0.75:
                additional = 2.0
            breakdown["pellet"] = PELLET_REWARD + 1.0 * frac + additional

    def _super_pellet_reward(
        self,
        events: dict[str, bool],
        powered: bool,
        threatening: int,
        breakdown: dict[str, float],
    ) -> None:
        if events.get("super_pellet_eaten", False):
            base = SUPER_PELLET_REWARD
            threat_bonus = min(threatening * 2.0, 6.0) if threatening > 0 else 0.0
            breakdown["super_pellet"] = base + threat_bonus
            breakdown["super_bait"] = threat_bonus

    def _milestone_reward(self, frac: float, breakdown: dict[str, float]) -> None:
        for threshold, reward in MILESTONE_REWARDS.items():
            if frac >= threshold and threshold not in self.milestones_hit:
                self.milestones_hit.add(threshold)
                breakdown["milestone"] += reward

    def _bfs_shaping(self, bfs_shaping: float, breakdown: dict[str, float]) -> None:
        breakdown["bfs"] = 1.0 * bfs_shaping

    def _oscillation_penalty(self, events, threat_dist, breakdown, explore_step=False):
        if explore_step:
            return
        if events.get("oscillating", False) and not (
            events.get("pellet_eaten", False) or events.get("super_pellet_eaten", False)
        ):
            if threat_dist < 4:
                self.osc_streak = 0
                return
            self.osc_streak += 1
            if self.osc_streak >= 3:
                breakdown["oscillation"] = -2.0  # flat small penalty
        else:
            self.osc_streak = 0

    def _ghost_proximity_penalty(
        self,
        min_ghost_dist_after: int,
        min_ghost_dist_before: int,
        events: dict[str, bool],
        powered: bool,
        breakdown: dict[str, float],
    ) -> None:
        """Repulsive proximity field using BFS distance. No escape bonus (handled by evasion_skill)."""
        if powered or min_ghost_dist_after < 0:
            return
        d = min_ghost_dist_after
        # Static repulsion: the closer, the stronger
        if d == 1:
            breakdown["ghost_proximity"] -= 2.0
        elif d == 2:
            breakdown["ghost_proximity"] -= 0.75
        elif d == 3:
            breakdown["ghost_proximity"] -= 0.5
        elif d == 4:
            breakdown["ghost_proximity"] -= 0.2
        elif d == 5:
            breakdown["ghost_proximity"] -= 0.05

        # Directional approach penalty: moving closer to a ghost is bad
        if (
            min_ghost_dist_before > 0
            and d < min_ghost_dist_before
            and not events.get("super_pellet_eaten", False)
        ):
            approach_strength = max(0, 6 - d)  # stronger when already very close
            breakdown["ghost_proximity"] -= 0.4 * approach_strength

    def _region_cleared_reward(
        self, events: dict[str, bool], breakdown: dict[str, float]
    ) -> None:
        if events.get("cleared_region", False):
            breakdown["region_cleared"] = 5.0

    def _region_dirty_penalty(
        self, events: dict[str, bool], breakdown: dict[str, float]
    ) -> None:
        if events.get("left_dirty_region", False):
            breakdown["region_dirty"] = -3.0

    def _backtrack_penalty(
        self, events: dict[str, bool], breakdown: dict[str, float]
    ) -> None:
        if events.get("backtracked", False):
            breakdown["backtrack"] = -1.0

    def _incomplete_penalty(
        self, events: dict[str, bool], breakdown: dict[str, float]
    ) -> None:
        if events.get("truncated", False):
            breakdown["incomplete"] = -50.0

    def _predictive_threat_reward(
        self,
        px: int,
        py: int,
        ghosts: list,
        maze: list[list[int]] | None,
        powered: bool,
        breakdown: dict[str, float],
    ) -> None:
        threatening, _, min_threat_dist, _ = self._count_threatening_ghosts(
            px, py, ghosts, maze
        )
        if not powered and threatening > 0 and min_threat_dist > 0:
            if min_threat_dist >= 5:
                breakdown["predictive_threat"] += 0.30
            elif min_threat_dist == 4:
                breakdown["predictive_threat"] += 0.15
            elif min_threat_dist == 3:
                breakdown["predictive_threat"] -= 0.5
            elif min_threat_dist == 2:
                breakdown["predictive_threat"] -= 1.5
            elif min_threat_dist == 1:
                breakdown["predictive_threat"] -= 4.0

    def _evasion_skill_reward(
        self,
        min_ghost_dist_after: int,
        breakdown: dict[str, float],
    ) -> None:
        """Credit for escaping a close ghost. Scale is moderate to avoid evasion-farming."""
        if (
            self.last_min_ghost_dist > 0
            and min_ghost_dist_after > self.last_min_ghost_dist
            and self.last_min_ghost_dist <= 4
        ):
            escape_quality = min_ghost_dist_after - self.last_min_ghost_dist
            if self.last_min_ghost_dist == 1:
                breakdown["evasion_skill"] += 1.0 * escape_quality
            elif self.last_min_ghost_dist == 2:
                breakdown["evasion_skill"] += 0.5 * escape_quality
            elif self.last_min_ghost_dist == 3:
                breakdown["evasion_skill"] += 0.2 * escape_quality
            elif self.last_min_ghost_dist == 4:
                breakdown["evasion_skill"] += 0.1 * escape_quality

    def _zone_control_reward(
        self,
        px: int,
        py: int,
        maze: list[list[int]] | None,
        threatening: int,
        powered: bool,
        breakdown: dict[str, float],
    ) -> None:
        is_corner = self._is_cornered(px, py, maze)
        if is_corner and not powered:
            if threatening > 0:
                breakdown["zone_control"] -= 10.0
                self.steps_in_corner += 1
                if self.steps_in_corner > 1:
                    breakdown["zone_control"] -= 2.5 * (self.steps_in_corner - 1)
            else:
                breakdown["zone_control"] -= 0.1
        else:
            self.steps_in_corner = max(0, self.steps_in_corner - 1)

    def _threat_mastery_reward(
        self,
        threatening: int,
        min_threat_dist: int,
        min_ghost_dist_after: int,
        powered: bool,
        breakdown: dict[str, float],
    ) -> None:
        if not powered and threatening > 0 and min_threat_dist in (2, 3):
            self.consecutive_threat_steps += 1
            if self.consecutive_threat_steps <= 5:
                breakdown["threat_mastery"] += 0.3
            elif self.consecutive_threat_steps <= 10:
                breakdown["threat_mastery"] += 0.1
            if min_ghost_dist_after > 3 and self.consecutive_threat_steps >= 3:
                breakdown["threat_mastery"] += 3.0
                self.consecutive_threat_steps = 0
        else:
            self.consecutive_threat_steps = max(0, self.consecutive_threat_steps - 1)

    def _ghost_lure_reward(
        self,
        edible_nearby: int,
        min_edible_dist: int,
        min_ghost_dist_after: int,
        powered: bool,
        breakdown: dict[str, float],
    ) -> None:
        if (
            powered
            and edible_nearby > 0
            and min_edible_dist > 0
            and self.last_min_ghost_dist > 0
            and min_ghost_dist_after > 0
            and min_ghost_dist_after < self.last_min_ghost_dist
            and self.last_min_ghost_dist <= 5
        ):
            approach_quality = self.last_min_ghost_dist - min_ghost_dist_after
            breakdown["ghost_lure"] += 2.0 * approach_quality

    def _survival_truncation_reward(
        self, events: dict[str, bool], frac: float, breakdown: dict[str, float]
    ) -> None:
        if events.get("truncated", False):
            survival_bonus = 200.0
            pellet_bonus = frac * 50.0
            breakdown["survival_truncation"] = survival_bonus + pellet_bonus

    def _dense_survival_reward(
        self,
        threatening: int,
        min_threat_dist: int,
        powered: bool,
        breakdown: dict[str, float],
    ) -> None:
        if not powered and threatening > 0:
            breakdown["survival_truncation"] += (
                0.25
                if min_threat_dist >= 5
                else (0.05 if min_threat_dist >= 3 else 0.1)
            )
        else:
            breakdown["survival_truncation"] += 0.2

    # ═══════════════════════════════════════════════════════════════════
    #  HELPERS
    # ═══════════════════════════════════════════════════════════════════

    def is_cornered(self, px: int, py: int, maze: list[list[int]]) -> bool:
        """Public wrapper so the env can log trap exposure without reward coupling."""
        return self._is_cornered(px, py, maze)

    def count_threatening(self, px: int, py: int, ghosts: list) -> int:
        """Public count of hunting ghosts within Manhattan distance 8."""
        return self._count_threatening_ghosts(px, py, ghosts, None)[0]

    def _is_cornered(self, px: int, py: int, maze: list[list[int]]) -> bool:
        """Check if Pac-Man is in a dead-end or corner (≤1 escape route)."""
        if not maze:
            return False
        h, w = len(maze), len(maze[0])
        exits = 0
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < h and 0 <= nx < w and maze[ny][nx] != 15:
                exits += 1
        return exits <= 1

    def _count_threatening_ghosts(
        self, px: int, py: int, ghosts: list, maze: list[list[int]] | None
    ) -> tuple[int, int, int, int]:
        threatening = 0
        edible_nearby = 0
        min_threat_dist = -1
        min_edible_dist = -1

        for g in ghosts:
            if g.in_prison:
                continue
            dist = abs(g.grid_x - px) + abs(g.grid_y - py)
            if dist > 8:
                continue
            if g.is_edible:
                edible_nearby += 1
                if min_edible_dist == -1 or dist < min_edible_dist:
                    min_edible_dist = dist
            else:
                threatening += 1
                if min_threat_dist == -1 or dist < min_threat_dist:
                    min_threat_dist = dist

        return threatening, edible_nearby, min_threat_dist, min_edible_dist

    # ═══════════════════════════════════════════════════════════════════
    #  MAIN CALCULATION — clean list of calls only
    # ═══════════════════════════════════════════════════════════════════

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
        steps_since_pellet,
        ghosts: list,
        movement,
        maze: list[list[int]] | None,
        threat_dist: float = float("inf"),
        min_ghost_dist_after: int = -1,
        min_ghost_dist_before: int = -1,
        explore_step: bool = False,
    ) -> tuple[float, dict[str, float]]:
        """Return (total_reward, breakdown_dict)."""
        breakdown = {
            "step": 0.0,
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
            "hunger": 0.0,
            "predictive_threat": 0.0,
            "evasion_skill": 0.0,
            "super_bait": 0.0,
            "zone_control": 0.0,
            "threat_mastery": 0.0,
            "ghost_lure": 0.0,
            "survival_truncation": 0.0,
            "exploration": 0.0,
            "zone_stagnation": 0.0,
        }

        px, py = player.grid_x, player.grid_y
        frac = (
            (total_pellets - remaining_pellets) / total_pellets
            if total_pellets > 0
            else 0.0
        )
        powered = bool(player.powered_timer > 0)
        threatening, edible_nearby, min_threat_dist, min_edible_dist = (
            self._count_threatening_ghosts(px, py, ghosts, maze)
        )

        self._death_penalty(events, breakdown)
        self._completion_reward(events, step_count, max_steps, breakdown)
        self._pellet_reward(events, frac, breakdown)
        self._super_pellet_reward(events, powered, threatening, breakdown)
        self._ghost_eat_reward(events, breakdown)
        self._exploration_reward(px, py, breakdown)
        self._oscillation_penalty(events, threat_dist, breakdown, explore_step)

        # Dense navigation guidance — use the BFS work you're already paying for
        if bfs_shaping != 0.0:
            self._bfs_shaping(bfs_shaping, breakdown)
        self._hunger_penalty(steps_since_pellet, breakdown)
        # Gradual ghost-avoidance gradient — THE most important missing signal
        self._ghost_proximity_penalty(
            min_ghost_dist_after,
            min_ghost_dist_before,
            events,
            powered,
            breakdown,
        )

        # Credit for escaping a close ghost
        self._evasion_skill_reward(min_ghost_dist_after, breakdown)

        # Mild corner penalty (anti-trap)
        self._zone_control_reward(
            px,
            py,
            maze,
            threatening,
            powered,
            breakdown,
        )

        # Milestones — break the +1000 completion into reachable chunks
        # self._milestone_reward(frac, breakdown)

        # Survival bonus — give the value function a positive terminal to aim for

        self.last_min_ghost_dist = (
            min_ghost_dist_after if min_ghost_dist_after >= 0 else min_threat_dist
        )
        return sum(breakdown.values()), breakdown
