"""Multi-Agent Reinforcement Learning Environment for Pac-Man vs Ghosts."""

from __future__ import annotations
from typing import Any

import torch
from AI_arena.player.player_env import PacmanPlayerEnv
from AI_arena.player.constants import DIRECTIONS


class PacmanAdversarialEnv(PacmanPlayerEnv):
    """MARL environment overriding ghost BFS with neural network actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_ghost_actions: list[int] | None = None

    def step_adversarial(
        self,
        player_action: int | torch.Tensor,
        ghost_actions: list[int] | torch.Tensor,
        explore: bool = False,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        float,
        float,
        bool,
        dict[str, Any],
    ]:
        """Step the environment using both Player and Ghost NN actions."""
        if isinstance(ghost_actions, torch.Tensor):
            ghost_actions = ghost_actions.view(-1).tolist()

        self.current_ghost_actions = ghost_actions

        # Step using the superclass (this will call our overridden _update_entities)
        obs, p_reward, done, info, _ = super().step(player_action, explore)

        self.current_ghost_actions = None

        # --- Zero-Sum Reward Formulation for Ghosts ---
        # Baseline: whatever is good for Pac-Man is bad for Ghosts
        g_reward = -p_reward * 0.5

        # Massive explicit reward if the ghosts manage to trap/kill Pac-Man
        if info["events"].get("pacman_died"):
            g_reward += 50.0

        # Penalty if a ghost gets eaten (so they learn to run away when edible)
        if info["events"].get("ghost_eaten"):
            g_reward -= 10.0

        return obs, p_reward, g_reward, done, info

    def _update_entities(self) -> None:
        """Override entity updates to inject neural network ghost actions."""
        if self.movement is None or self.player is None or self.maze is None:
            return

        self.movement.update_entity(self.player)

        # Handle power pellet timer
        if self.player.powered_timer > 0:
            self.player.powered_timer -= 0.1
            if self.player.powered_timer <= 0:
                self.player.end_powered_mode()
                for ghost in self.ghosts:
                    ghost.is_edible = False

        if self.current_ghost_actions is not None:
            # ── MARL Neural Network Ghost Control ──
            if self.stage == 1:
                # Stage 1: Ghosts are permanently disabled in prison
                return

            for idx, ghost in enumerate(self.ghosts):
                if idx >= len(self.current_ghost_actions):
                    break

                # Handle prison exit manually (bypassing controller)
                if ghost.in_prison:
                    if self._ghost_respawn_ticks[idx] > 0:
                        self._ghost_respawn_ticks[idx] -= 1
                    else:
                        ghost.in_prison = False
                        ghost.is_edible = False
                        ghost.runaway_target = None
                else:
                    # Physics accumulation (ghosts usually move slightly slower than Pac-Man)
                    frightened_speed = min(0.5, self.ghost_speed_ratio)
                    speed = (
                        frightened_speed
                        if ghost.is_edible
                        else self.ghost_speed_ratio
                    )

                    ghost._tick_accumulator += speed
                    if ghost._tick_accumulator >= 1.0:
                        ghost._tick_accumulator -= 1.0

                        # Apply Neural Network action!
                        action = self.current_ghost_actions[idx]
                        ghost.next_direction = DIRECTIONS[action]
                        self.movement.update_entity(ghost)
        else:
            # ── Fallback to Hardcoded BFS (if run normally) ──
            self._ghost_ctrl.update(
                ghosts=self.ghosts,
                player=self.player,
                stage=self.stage,
                ghost_respawn_ticks=self._ghost_respawn_ticks,
                ghost_confusion_prob=self.ghost_confusion_prob,
            )
