from dataclasses import dataclass, asdict
from typing import Literal

Direction = Literal["UP", "DOWN", "LEFT", "RIGHT", "NONE"]


@dataclass
class GhostState:
    name: str
    position: tuple[int, int]
    direction: Direction
    mode: Literal[
        "CHASE",
        "FRIGHTENED",
    ]
    distance_to_player: int


@dataclass
class WorldState:

    maze: list[list[int]]
    pellets: list[list[int]]
    player_position: tuple[int, int]
    player_direction: Direction
    player_powered: bool
    remaining_pellets: int
    remaining_super_pellets: int


@dataclass
class Target:

    Blinky: Direction
    Pinky: Direction
    Inky: Direction
    Clyde: Direction


@dataclass
class TrainingSample:
    metadata: dict
    world: WorldState
    ghosts: list[GhostState]
    target: Target

    def to_dict(self):
        return asdict(self)
