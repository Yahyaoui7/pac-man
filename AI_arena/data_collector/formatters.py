import json
from typing import TextIO

from AI_arena.data_collector.models import TrainingSample

DIRECTION_INDEX = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}


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
        bfs_dist = ghost.path_length / max_dim if ghost.path_length else 1.0
        manhattan_dist = ghost.manhattan_distance / max_dim
        local_pellets = ghost.local_pellet_count / max(w * h, 1)
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
            sample.ghosts[j] for j in range(len(sample.ghosts)) if j != ghost_idx
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

            # how many other ghosts are close to the player (pressure)
            ghosts_near_player = (
                sum(1 for og in other_ghosts if og.path_length <= 10) / 3.0
            )

            # relative positions of closest other ghost
            closest_og = min(
                other_ghosts,
                key=lambda og: abs(gx - og.position[0]) + abs(gy - og.position[1]),
            )
            og_rel_x = (closest_og.position[0] - gx) / max_dim
            og_rel_y = (closest_og.position[1] - gy) / max_dim

            features.extend(
                [
                    closest_dist,  # 21 closest other ghost dist
                    avg_bfs,  # 22 avg other ghost BFS to player
                    ghosts_near_player,  # 23 ghosts near player (pressure)
                    og_rel_x,  # 24 closest ghost rel_x
                    og_rel_y,  # 25 closest ghost rel_y
                ]
            )
        elif not single_ghost:
            # no other ghosts — pad with zeros
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        first_dir = ghost.bfs_directions[0] if ghost.bfs_directions else None
        label = DIRECTION_INDEX.get(first_dir, -1)

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
                ]
            )
        return names


class CNNFormatter:
    """Formats a TrainingSample into grid-based data for CNN training."""

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
