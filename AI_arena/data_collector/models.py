from dataclasses import dataclass, asdict
from typing import Optional, Literal

Direction = Literal["UP", "DOWN", "LEFT", "RIGHT", "NONE"]


LOCAL_PELLET_RADIUS = 3


@dataclass
class GhostState:
    name: str
    position: tuple[int, int]
    mode: Literal["CHASE", "FRIGHTENED"]
    distance_to_player: int
    bfs_path: list[tuple[int, int]]
    bfs_directions: list[Optional[str]]
    path_length: int
    available_moves: list[str]
    manhattan_distance: int
    local_pellet_count: int
    num_exits: int
    frightened_timer: float


@dataclass
class WorldState:
    maze: list[list[int]]
    pellets: list[list[int]]
    player_position: tuple[int, int]
    player_direction: Direction
    player_powered: bool
    remaining_pellets: int
    remaining_super_pellets: int
    player_available_moves: list[str]


@dataclass
class TrainingSample:
    metadata: dict
    world: WorldState
    ghosts: list[GhostState]

    def to_dict(self):
        return asdict(self)
