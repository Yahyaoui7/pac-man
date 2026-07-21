import json
from typing import TextIO

from AI_arena.data_collector.models import TrainingSample

DIRECTION_INDEX = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}


class MLPFormatter:

    @staticmethod
    def format_line(sample: TrainingSample, ghost_idx: int) -> dict:
        w = sample.metadata["maze_width"]
        h = sample.metadata["maze_height"]
        px, py = sample.world.player_position
        ghost = sample.ghosts[ghost_idx]
        gx, gy = ghost.position

        features = [
            px / max(w - 1, 1),
            py / max(h - 1, 1),
            gx / max(w - 1, 1),
            gy / max(h - 1, 1),
            float("UP" in ghost.available_moves),
            float("DOWN" in ghost.available_moves),
            float("LEFT" in ghost.available_moves),
            float("RIGHT" in ghost.available_moves),
            ghost.path_length / max(w + h, 1),
            ghost.manhattan_distance / max(w + h, 1),
            ghost.local_pellet_count / max(w * h, 1),
            ghost.num_exits / 4.0,
            w / 50.0,
            h / 50.0,
            sample.world.remaining_pellets / max(w * h, 1),
            sample.world.remaining_super_pellets / max(w * h, 1),
        ]

        first_dir = ghost.bfs_directions[0] if ghost.bfs_directions else None
        label = DIRECTION_INDEX.get(first_dir, -1)

        return {"features": features, "label": label}


class CNNFormatter:
    """Formats a TrainingSample into grid-based data for CNN training.

    Placeholder — will be expanded with maze/pellet/ghost overlay grids.
    """

    @staticmethod
    def format_line(sample: TrainingSample, ghost_idx: int) -> dict:
        ghost = sample.ghosts[ghost_idx]
        first_dir = ghost.bfs_directions[0] if ghost.bfs_directions else None
        return {
            "maze": sample.world.maze,
            "pellets": sample.world.pellets,
            "player_position": list(sample.world.player_position),
            "ghost_position": list(ghost.position),
            "ghost_name": ghost.name,
            "label": DIRECTION_INDEX.get(first_dir, -1),
        }


class StreamWriter:
    """Appends JSONL lines to a file handle. One write per line, no buffering."""

    def __init__(self, fh: TextIO) -> None:
        self.fh = fh

    def write_line(self, record: dict) -> None:
        self.fh.write(json.dumps(record) + "\n")
