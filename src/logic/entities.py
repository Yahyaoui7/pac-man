class Player:

    def __init__(self, row: int, col: int) -> None:
        self.row = row
        self.col = col
        self.lives = 3
        self.score = 0
        self.speed = 1
        self.direction = None
        self.row_direction = 0
        self.col_dirction = 0


class Ghost:
    def __init__(self, row: int, col: int, name: str) -> None:
        self.row = row
        self.col = col
        self.name = name

        self.direction = None
        self.row_direction = 0
        self.col_direction = 0

        self.is_edible = False
        self.is_eaten = False

        self.spawn_row = row
        self.spawn_col = col
