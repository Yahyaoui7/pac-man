"""Inference controller for CNN-based Pac-Man player model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from AI_arena.models.cnn_player import (
    PlayerActorCritic,
    load_checkpoint_into_policy,
)
from src.graphics.entitys.ghost import Ghost
from src.graphics.entitys.player import Player

from AI_arena.player.data.observation import format_player_observation

BEST_STAGE_PATH = Path(__file__).parent.parent / "models" / "player_rl_best.pt"

DEFAULT_STAGE_PATH = Path(__file__).parent.parent / "models" / "player_rl.pt"
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class CNNPlayerController:
    """Build live observations and predict Pac-Man's best move."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_search: bool = True,
        search_horizon: int = 12,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PlayerActorCritic().to(self.device)
        self.use_search = use_search
        self.search_horizon = search_horizon
        self.search_planner: Any = None

        if model_path is None:
            candidates = [DEFAULT_STAGE_PATH, BEST_STAGE_PATH]
            existing = [c for c in candidates if c.exists()]
            if existing:
                path = max(existing, key=lambda p: p.stat().st_mtime)
            else:
                path = DEFAULT_STAGE_PATH
        else:
            path = Path(model_path)

        if path.exists() and load_checkpoint_into_policy(
            self.model, path, device=self.device
        ):
            print(f"Loaded player RL checkpoint from {path} (Search lookahead: {self.use_search})")
        else:
            print(
                f"Warning: Player RL checkpoint {path} not found or failed to load. Using untrained weights."
            )

        self.model.eval()
        self.last_diagnostics: dict[str, Any] = {}
        self.last_action_idx: int | None = None
        self._hidden: torch.Tensor | None = None  # GRU memory persists across steps

        # Observation feature state tracking
        self.visit_counts: list[list[int]] | None = None
        self.initial_pellet_count: int | None = None
        self.prev_nearest_pellet_dist: float = -1.0
        self.prev_nearest_ghost_dist: float = -1.0
        self.prev_nearest_pp_dist: float = -1.0
        self.steps_since_pellet: int = 0
        self.same_action_count: int = 0
        self.last_positions: list[tuple[int, int]] = []

    def reset_state(self) -> None:
        """Reset internal state history (e.g. between games or post-respawn)."""
        self.last_action_idx = None
        self._hidden = None  # wipe GRU memory on new game/life
        self.visit_counts = None
        self.initial_pellet_count = None
        self.prev_nearest_pellet_dist = -1.0
        self.prev_nearest_ghost_dist = -1.0
        self.prev_nearest_pp_dist = -1.0
        self.steps_since_pellet = 0
        self.same_action_count = 0
        self.last_positions = []

    def get_action(
        self,
        maze: list[list[int]],
        pellets: list[list[int]],
        player: Player,
        ghosts: list[Ghost],
        movement_system: Any,
        sample: bool = False,
        use_search: bool | None = None,
    ) -> str:
        """Construct state tensors and select action (sampling from distribution or greedy)."""
        height = len(maze)
        width = len(maze[0]) if height else 0
        px, py = player.grid_x, player.grid_y

        if self.initial_pellet_count is None and pellets:
            self.initial_pellet_count = sum(
                1 for row in pellets for cell in row if cell in (1, 2)
            )

        if (
            self.visit_counts is None
            or len(self.visit_counts) != height
            or (height > 0 and len(self.visit_counts[0]) != width)
        ):
            self.visit_counts = [[0 for _ in range(width)] for _ in range(height)]

        if 0 <= py < height and 0 <= px < width:
            self.visit_counts[py][px] += 1
            self.last_positions.append((py, px))
            if len(self.last_positions) > 20:
                self.last_positions.pop(0)

        grid, extra_features, valid_actions = self._build_observation(
            maze, pellets, player, ghosts, movement_system
        )

        with torch.no_grad():
            # Pass hidden state into model, receive updated hidden state back
            logits, value, self._hidden = self.model(grid, extra_features, self._hidden)
            logits = logits.float()
            value = value.float()
            masked_logits = logits.masked_fill(~valid_actions, -1e8)
            masked_logits = torch.nan_to_num(
                masked_logits, nan=-1e8, posinf=10.0, neginf=-1e8
            )
            probs = torch.softmax(masked_logits, dim=-1)[0]
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            if probs.sum() <= 0:
                probs = valid_actions[0].float()
                if probs.sum() <= 0:
                    probs = torch.ones_like(probs)
            probs = probs / probs.sum()

            nn_action_index = (
                int(torch.multinomial(probs, 1).item())
                if sample
                else int(torch.argmax(masked_logits, dim=-1).item())
            )

        active_search = self.use_search if use_search is None else use_search
        search_scores = None
        if active_search:
            if self.search_planner is None:
                from AI_arena.player.search_planner import PacmanLookaheadSearch

                self.search_planner = PacmanLookaheadSearch(
                    maze=maze,
                    movement=movement_system,
                    horizon=self.search_horizon,
                )
            action_index = self.search_planner.get_best_action(
                player=player,
                ghosts=ghosts,
                pellets=pellets,
            )
            search_scores = self.search_planner.get_action_scores(
                player=player,
                ghosts=ghosts,
                pellets=pellets,
            )
        else:
            action_index = nn_action_index

        if self.last_action_idx is not None and self.last_action_idx == action_index:
            self.same_action_count += 1
        else:
            self.same_action_count = 0

        self.last_action_idx = action_index
        chosen_action = DIRECTIONS[action_index]

        if (
            0 <= py < height
            and 0 <= px < width
            and pellets
            and pellets[py][px] in (1, 2)
        ):
            self.steps_since_pellet = 0
        else:
            self.steps_since_pellet += 1

        self.last_diagnostics = {
            "chosen_action": chosen_action,
            "search_used": active_search,
            "nn_action": DIRECTIONS[nn_action_index],
            "estimated_value": round(float(value.item()), 4),
            "probabilities": {
                d: round(float(probs[i].item()), 4) for i, d in enumerate(DIRECTIONS)
            },
            "logits": {
                d: round(float(logits[0, i].item()), 4)
                for i, d in enumerate(DIRECTIONS)
            },
            "valid_actions": {
                d: bool(valid_actions[0, i].item()) for i, d in enumerate(DIRECTIONS)
            },
        }
        if search_scores is not None:
            self.last_diagnostics["search_scores"] = {
                DIRECTIONS[a]: round(float(s), 1) for a, s in search_scores.items()
            }

        return chosen_action

    def _build_observation(
        self,
        maze: list[list[int]],
        pellets: list[list[int]],
        player: Player,
        ghosts: list[Ghost],
        movement_system: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return format_player_observation(
            maze=maze,
            pellets=pellets,
            player=player,
            ghosts=ghosts,
            movement=movement_system,
            initial_pellet_count=self.initial_pellet_count,
            device=self.device,
            visit_counts=self.visit_counts,
            prev_nearest_pellet_dist=self.prev_nearest_pellet_dist,
            prev_nearest_ghost_dist=self.prev_nearest_ghost_dist,
            prev_nearest_pp_dist=self.prev_nearest_pp_dist,
            steps_since_pellet=self.steps_since_pellet,
            last_positions=self.last_positions,
            just_died=0.0,
            same_action_count=self.same_action_count,
        )
