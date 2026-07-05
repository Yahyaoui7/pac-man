from typing import Optional


class Entity:
    def __init__(
        self,
        row: int,
        col: int,
        cell_size: int,
        speed: int,
    ) -> None:
        self.row = row
        self.col = col
        self.cell_size = cell_size

        self.x = col * cell_size + cell_size // 2
        self.y = row * cell_size + cell_size // 2

        self.speed = speed

        self.direction: Optional[str] = None
        self.next_direction: Optional[str] = None

        self.row_direction = 0
        self.col_direction = 0


class Player(Entity):
    def __init__(self, row: int, col: int, cell_size: int) -> None:
        super().__init__(row, col, cell_size, speed=3)

        self.lives = 3
        self.score = 0
        self.is_invincible = False


class Ghost(Entity):
    def __init__(
        self,
        row: int,
        col: int,
        cell_size: int,
        name: str,
    ) -> None:
        super().__init__(row, col, cell_size, speed=1)

        self.name = name

        self.spawn_row = row
        self.spawn_col = col
        self.spawn_x = self.x
        self.spawn_y = self.y

        self.is_edible = False
        self.is_eaten = False
