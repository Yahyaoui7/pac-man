import json
from typing import TextIO

from AI_arena.data_collector.models import LOCAL_PELLET_RADIUS, TrainingSample

NEAR_PLAYER_FRACTION = 0.3


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
    def format_line(
        sample: TrainingSample,
        ghost_idx: int,
        single_ghost: bool = True,
    ) -> dict:
        w = sample.metadata["maze_width"]
        h = sample.metadata["maze_height"]
        max_dim = max(w, h, 1)
        px, py = sample.world.player_position
        ghost = sample.ghosts[ghost_idx]
        gx, gy = ghost.position

        # --- Player features ---
        player_moves = sample.world.player_available_moves
        player_up = float("UP" in player_moves)
        player_down = float("DOWN" in player_moves)
        player_left = float("LEFT" in player_moves)
        player_right = float("RIGHT" in player_moves)

        # --- Target ghost features ---
        rel_x = (gx - px) / max_dim
        rel_y = (gy - py) / max_dim
        bfs_dist = ghost.path_length / max_dim
        manhattan_dist = ghost.manhattan_distance / max_dim
        pellet_window = min((2 * LOCAL_PELLET_RADIUS + 1) ** 2, w * h)
        local_pellets = ghost.local_pellet_count / max(pellet_window, 1)
        num_exits = ghost.num_exits / 4.0

        ghost_up = float("UP" in ghost.available_moves)
        ghost_down = float("DOWN" in ghost.available_moves)
        ghost_left = float("LEFT" in ghost.available_moves)
        ghost_right = float("RIGHT" in ghost.available_moves)

        mode_chase = float(ghost.mode == "CHASE")
        mode_frightened = float(ghost.mode == "FRIGHTENED")
        fright_timer = ghost.frightened_timer / 10.0

        features = [
            px / max_dim,  # 0  player_x
            py / max_dim,  # 1  player_y
            gx / max_dim,  # 2  ghost_x
            gy / max_dim,  # 3  ghost_y
            rel_x,  # 4  rel_x to player
            rel_y,  # 5  rel_y to player
            bfs_dist,  # 6  BFS distance to player
            manhattan_dist,  # 7  manhattan distance to player
            local_pellets,  # 8  local pellet density
            num_exits,  # 9  exit count (corridor vs open)
            ghost_up,  # 10 ghost can go UP
            ghost_down,  # 11 ghost can go DOWN
            ghost_left,  # 12 ghost can go LEFT
            ghost_right,  # 13 ghost can go RIGHT
            player_up,  # 14 player can go UP
            player_down,  # 15 player can go DOWN
            player_left,  # 16 player can go LEFT
            player_right,  # 17 player can go RIGHT
            mode_chase,  # 18 mode == CHASE
            mode_frightened,  # 19 mode == FRIGHTENED
            fright_timer,  # 20 frightened timer
        ]

        # --- Cross-ghost context ---
        other_ghosts = [
            sample.ghosts[j]
            for j in range(len(sample.ghosts))
            if j != ghost_idx
        ]

        if not single_ghost and other_ghosts:
            # closest other ghost to this ghost
            closest_dist = (
                min(
                    abs(gx - og.position[0]) + abs(gy - og.position[1])
                    for og in other_ghosts
                )
                / max_dim
            )

            # average BFS distance of other ghosts to player
            avg_bfs = sum(og.path_length for og in other_ghosts) / (
                len(other_ghosts) * max_dim
            )

            near_threshold = NEAR_PLAYER_FRACTION * max_dim
            ghosts_near_player = sum(
                1 for og in other_ghosts if og.path_length <= near_threshold
            ) / len(other_ghosts)

            # relative positions of closest other ghost
            closest_og = min(
                other_ghosts,
                key=lambda og: abs(gx - og.position[0])
                + abs(gy - og.position[1]),
            )
            og_rel_x = (closest_og.position[0] - gx) / max_dim
            og_rel_y = (closest_og.position[1] - gy) / max_dim
            has_other_ghosts = 1.0

            features.extend(
                [
                    closest_dist,  # 21 closest other ghost dist
                    avg_bfs,  # 22 avg other ghost BFS to player
                    ghosts_near_player,  # 23 ghosts near player (pressure)
                    og_rel_x,  # 24 closest ghost rel_x
                    og_rel_y,  # 25 closest ghost rel_y
                    has_other_ghosts,  # 26 whether any other ghosts exist
                ]
            )
        elif not single_ghost:
            features.extend([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        first_dir = ghost.bfs_directions[0] if ghost.bfs_directions else None
        # FIX: Handle None explicitly so the lookup is type-safe.
        label = (
            DIRECTION_INDEX.get(first_dir, -1) if first_dir is not None else -1
        )

        return {"features": features, "label": label}

    @staticmethod
    def feature_names(single_ghost: bool = True) -> list[str]:
        names = [
            "player_x",
            "player_y",
            "ghost_x",
            "ghost_y",
            "rel_x",
            "rel_y",
            "bfs_dist",
            "manhattan_dist",
            "local_pellets",
            "num_exits",
            "ghost_up",
            "ghost_down",
            "ghost_left",
            "ghost_right",
            "player_up",
            "player_down",
            "player_left",
            "player_right",
            "mode_chase",
            "mode_frightened",
            "fright_timer",
        ]
        if not single_ghost:
            names.extend(
                [
                    "closest_ghost_dist",
                    "avg_other_bfs",
                    "ghosts_near_player",
                    "closest_ghost_rel_x",
                    "closest_ghost_rel_y",
                    "has_other_ghosts",
                ]
            )
        return names


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
            [[0 for _ in range(width)] for _ in range(height)]
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
                # MazeGenerator uses 15 (all four walls set) for the blocked
                # cells that form its centered "42" pattern.
                grid[11][y][x] = int(cell != 15)

        player_x, player_y = sample.world.player_position
        grid[6][player_y][player_x] = 1

        ghosts_by_name = {ghost.name: ghost for ghost in sample.ghosts}
        if set(ghosts_by_name) != set(GHOST_NAMES):
            raise ValueError(
                "CNN samples require exactly the four named ghosts"
            )

        labels = []
        valid_actions = []
        for ghost_idx, name in enumerate(GHOST_NAMES):
            ghost = ghosts_by_name[name]
            ghost_x, ghost_y = ghost.position
            grid[7 + ghost_idx][ghost_y][ghost_x] = 1

            first_dir = (
                ghost.bfs_directions[0] if ghost.bfs_directions else None
            )
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

        # CENTRAL CNN: global values stay out of spatial channels.
        extra_features = [
            *player_direction,
            float(sample.world.player_powered),
            *(
                float(ghosts_by_name[name].mode == "FRIGHTENED")
                for name in GHOST_NAMES
            ),
        ]

        return {
            "grid": grid,
            # Retain source dimensions for masks, diagnostics, and inference.
            "extra_features": extra_features,
            "valid_actions": valid_actions,
            "labels": labels,
        }


class StreamWriter:
    """Append JSONL records to a file handle, one complete line at a time."""

    def __init__(self, fh: TextIO) -> None:
        self.fh = fh

    def write_line(self, record: dict) -> None:
        self.fh.write(json.dumps(record) + "\n")
