from typing import Optional

from src.graphics.entitys.graphic_lib import Facing
from src.logic.helpers import grid_to_pixel


class Entity:
    def __init__(
        self,
        y: int,
        x: int,
        speed: float,
    ) -> None:
        self.spawn_x = x
        self.spawn_y = y
        self.grid_y = y
        self.grid_x = x

        px, py = grid_to_pixel(y, x)
        self.x: float = float(px)
        self.y: float = float(py)

        self.speed = speed

        self.direction: Optional[str] = None
        self.next_direction: Optional[str] = None

        self.row_direction = 0
        self.col_direction = 0
        self.grid_x_direction = 0
        self.grid_y_direction = 0

        self.facing = Facing.RIGHT
