class Player:
    def __init__(self, row: int, col: int, cell_size: int) -> None:
        self.row = row
        self.col = col
        self.cell_size = cell_size

        self.x = col * cell_size + cell_size // 2
        self.y = row * cell_size + cell_size // 2

        self.lives = 3
        self.score = 0
        self.speed = 5

        self.direction = None
        self.next_direction = None

        self.row_direction = 0
        self.col_direction = 0


class Ghost:
    def __init__(self, row: int, col: int, name: str, cell_size: int) -> None:
        self.row = row
        self.col = col
        self.cell_size = cell_size

        self.x = col * cell_size + cell_size // 2
        self.y = row * cell_size + cell_size // 2

        self.speed = 5
        self.name = name

        self.direction = None
        self.next_direction = None

        self.row_direction = 0
        self.col_direction = 0

        self.is_edible = False
        self.is_eaten = False

        self.spawn_row = row
        self.spawn_col = col
        self.spawn_x = self.x
        self.spawn_y = self.y

    def reset_ghost(self):
        self.row = self.spawn_row
        self.col = self.spawn_col
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.direction = None
        self.next_direction = None
        self.is_eaten = False
