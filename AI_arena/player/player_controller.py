"""Inference controller for CNN-based Pac-Man player model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import os
import sys
import onnxruntime as ort
import numpy as np

from src.graphics.entitys.ghost import Ghost
from src.graphics.entitys.player import Player

from AI_arena.player.data.observation import format_player_observation

ONNX_MODEL_PATH = Path(__file__).parent.parent / "models" / "player_model.onnx"
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")


class CNNPlayerController:
    """Build live observations and predict Pac-Man's best move."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        use_search: bool = True,
        search_horizon: int = 8,
    ) -> None:
        self.use_search = use_search
        self.search_horizon = search_horizon
        self.search_planner: Any = None

        if model_path:
            path = Path(model_path)
        elif ONNX_MODEL_PATH.exists():
            path = ONNX_MODEL_PATH
        elif Path("AI_arena/models/player_model.onnx").exists():
            path = Path("AI_arena/models/player_model.onnx")
        else:
            exe_dir = (
                Path(sys.executable).resolve().parent
                if getattr(sys, "frozen", False)
                else Path(__file__).resolve().parent
            )
            alt = exe_dir / "AI_arena" / "models" / "player_model.onnx"
            path = alt if alt.exists() else ONNX_MODEL_PATH

        if path.exists():
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            print(
                f"Loaded ONNX model from {path} (Search lookahead: "
                f"{self.use_search}, horizon: {self.search_horizon})"
            )
        else:
            raise FileNotFoundError(f"ONNX model not found at {path}")

        self.last_diagnostics: dict[str, Any] = {}
        self.device = "cpu"
        self.reset_state()

    def reset_state(self) -> None:
        """Reset internal state history (e.g. between games or post-respawn)."""
        self.last_action_idx: int | None = None
        self._hidden: np.ndarray = np.zeros((2, 1, 384), dtype=np.float32)
        self.visit_counts: list[list[int]] | None = None
        self.initial_pellet_count: int | None = None
        self.prev_nearest_pellet_dist: float = -1.0
        self.prev_nearest_ghost_dist: float = -1.0
        self.prev_nearest_pp_dist: float = -1.0
        self.steps_since_pellet: int = 0
        self.same_action_count: int = 0
        self.last_positions: list[tuple[int, int]] = []

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

        if (
            self.visit_counts is not None
            and 0 <= py < height
            and 0 <= px < width
        ):
            self.visit_counts[py][px] += 1
            self.last_positions.append((py, px))
            if len(self.last_positions) > 20:
                self.last_positions.pop(0)

        grid, extra_features, valid_actions = self._build_observation(
            maze, pellets, player, ghosts, movement_system
        )

        # Convert observation tensors / arrays to numpy arrays
        grid_np = (
            grid.cpu().numpy().astype(np.float32)
            if hasattr(grid, "cpu")
            else np.asarray(grid, dtype=np.float32)
        )
        extra_np = (
            extra_features.cpu().numpy().astype(np.float32)
            if hasattr(extra_features, "cpu")
            else np.asarray(extra_features, dtype=np.float32)
        )
        valid_np = (
            valid_actions.cpu().numpy()[0]
            if hasattr(valid_actions, "cpu")
            else np.asarray(valid_actions[0], dtype=bool)
        )

        outputs = self.session.run(
            ["logits", "value", "next_hidden"],
            {
                "grid": grid_np,
                "extra_features": extra_np,
                "hidden": self._hidden,
            },
        )
        logits_np, value_np, self._hidden = outputs
        logits = logits_np[0]  # shape: (4,)

        # Mask invalid actions
        masked_logits = np.where(valid_np, logits, -1e8)

        # Softmax for probabilities
        shift_logits = masked_logits - np.max(masked_logits)
        exp_logits = np.exp(shift_logits)
        probs = exp_logits / np.sum(exp_logits)

        if sample:
            nn_action_index = int(np.random.choice(len(probs), p=probs))
        else:
            nn_action_index = int(np.argmax(masked_logits))

        active_search = self.use_search if use_search is None else use_search
        search_scores = None
        if active_search:
            if self.search_planner is None:
                from AI_arena.player.search_planner import PacmanLookaheadSearch

                self.search_planner = PacmanLookaheadSearch(
                    maze=maze,
                    movement=movement_system,
                    horizon=self.search_horizon,
                    beam_width=20,
                )
            search_scores = self.search_planner.get_action_scores(
                player=player,
                ghosts=ghosts,
                pellets=pellets,
                prev_action=self.last_action_idx,
            )
            action_index = int(max(search_scores, key=lambda a: search_scores[a]))
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
            "estimated_value": round(float(value_np[0, 0]), 4),
            "probabilities": {
                d: round(float(probs[i]), 4) for i, d in enumerate(DIRECTIONS)
            },
            "logits": {
                d: round(float(logits[i]), 4)
                for i, d in enumerate(DIRECTIONS)
            },
            "valid_actions": {
                d: bool(valid_np[i]) for i, d in enumerate(DIRECTIONS)
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
