NORTH = 1
EAST = 2
SOUTH = 4
WEST = 8


class MovementSystem:
    def __init__(self, maze):
        self.maze = maze

    def set_direction(self, entity, direction: str) -> None:
        entity.direction = direction

        if direction == "LEFT":
            entity.row_direction = 0
            entity.col_direction = -1

        elif direction == "RIGHT":
            entity.row_direction = 0
            entity.col_direction = 1

        elif direction == "UP":
            entity.row_direction = -1
            entity.col_direction = 0

        elif direction == "DOWN":
            entity.row_direction = 1
            entity.col_direction = 0

    def can_move(self, row: int, col: int, direction: str) -> bool:
        cell = self.maze[row][col]

        if direction == "LEFT":
            return col > 0 and not (cell & WEST)
        if direction == "RIGHT":
            return col < len(self.maze[0]) - 1 and not (cell & EAST)
        if direction == "UP":
            return row > 0 and not (cell & NORTH)
        if direction == "DOWN":
            return row < len(self.maze) - 1 and not (cell & SOUTH)

        return False

    def update_entity(self, entity) -> None:
        if entity.direction is None:
            return

        if not self.can_move(entity.row, entity.col, entity.direction):
            return
        entity.row += entity.row_direction
        entity.col += entity.col_direction
