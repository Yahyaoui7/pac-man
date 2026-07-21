import json
from typing import TextIO

from AI_arena.data_collector.models import TrainingSample

DIRECTION_INDEX = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
GHOST_NAMES = ("Blinky", "Pinky", "Inky", "Clyde")
MAX_MAZE_WIDTH = 25
MAX_MAZE_HEIGHT = 50

# MazeGenerator stores the four walls as bits in every maze cell.
NORTH = 1 << 0
EAST = 1 << 1
SOUTH = 1 << 2
WEST = 1 << 3

CNN_CHANNELS = (
    "wall_up",
    "wall_down",
    "wall_left",
    "wall_right",
    "normal_pellet",
    "super_pellet",
    "player",
    "blinky",
    "pinky",
    "inky",
    "clyde",
    "valid_cell",
)


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
        # FIX: Handle None explicitly so the lookup is type-safe.
        label = DIRECTION_INDEX.get(first_dir, -1) if first_dir is not None else -1

        return {"features": features, "label": label}


class CNNFormatter:
    """Create one centralized observation and four action targets per world."""

    @staticmethod
    def format_line(sample: TrainingSample) -> dict:
        width = sample.metadata["maze_width"]
        height = sample.metadata["maze_height"]
        if width > MAX_MAZE_WIDTH or height > MAX_MAZE_HEIGHT:
            raise ValueError(
                f"Maze {width}x{height} exceeds CNN grid "
                f"{MAX_MAZE_WIDTH}x{MAX_MAZE_HEIGHT}"
            )

        # Use one fixed tensor shape so records can be batched directly. The
        # valid-cell channel distinguishes real maze cells from zero padding.
        grid = [
            [
                [0 for _ in range(MAX_MAZE_WIDTH)]
                for _ in range(MAX_MAZE_HEIGHT)
            ]
            for _ in CNN_CHANNELS
        ]

        for y in range(height):
            for x in range(width):
                cell = sample.world.maze[y][x]
                pellet = sample.world.pellets[y][x]
                grid[0][y][x] = int(bool(cell & NORTH))
                grid[1][y][x] = int(bool(cell & SOUTH))
                grid[2][y][x] = int(bool(cell & WEST))
                grid[3][y][x] = int(bool(cell & EAST))
                grid[4][y][x] = int(pellet == 1)
                grid[5][y][x] = int(pellet == 2)
                grid[11][y][x] = 1

        player_x, player_y = sample.world.player_position
        grid[6][player_y][player_x] = 1

        ghosts_by_name = {ghost.name: ghost for ghost in sample.ghosts}
        if set(ghosts_by_name) != set(GHOST_NAMES):
            raise ValueError("CNN samples require exactly the four named ghosts")

        labels = []
        valid_actions = []
        for ghost_idx, name in enumerate(GHOST_NAMES):
            ghost = ghosts_by_name[name]
            ghost_x, ghost_y = ghost.position
            grid[7 + ghost_idx][ghost_y][ghost_x] = 1

            first_dir = ghost.bfs_directions[0] if ghost.bfs_directions else None
            if first_dir not in DIRECTION_INDEX:
                raise ValueError(f"Ghost {name} has no valid supervised label")
            labels.append(DIRECTION_INDEX[first_dir])
            valid_actions.append(
                [
                    int(direction in ghost.available_moves)
                    for direction in DIRECTION_INDEX
                ]
            )

        player_direction = [
            int(sample.world.player_direction == direction)
            for direction in DIRECTION_INDEX
        ]
        maze_area = max(width * height, 1)

        # CENTRAL CNN: global values stay out of spatial channels.
        extra_features = [
            *player_direction,
            float(sample.world.player_powered),
            *(float(ghosts_by_name[name].mode == "FRIGHTENED") for name in GHOST_NAMES),
            sample.world.remaining_pellets / maze_area,
            sample.world.remaining_super_pellets / maze_area,
            width / MAX_MAZE_WIDTH,
            height / MAX_MAZE_HEIGHT,
        ]

        return {
            "grid": grid,
            # Retain source dimensions for masks, diagnostics, and inference.
            "height": height,
            "width": width,
            "extra_features": extra_features,
            "valid_actions": valid_actions,
            "labels": labels,
        }


class StreamWriter:
    """Appends JSONL lines to a file handle. One write per line, no buffering."""

    def __init__(self, fh: TextIO) -> None:
        self.fh = fh

    def write_line(self, record: dict) -> None:
        self.fh.write(json.dumps(record) + "\n")
