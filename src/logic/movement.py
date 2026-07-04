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

    def is_centered(self, entity) -> bool:
        return (
            entity.x % entity.cell_size == entity.cell_size // 2
            and entity.y % entity.cell_size == entity.cell_size // 2
        )

    def update_cell_position(self, entity) -> None:
        entity.row = int(entity.y // entity.cell_size)
        entity.col = int(entity.x // entity.cell_size)

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
        if self.is_centered(entity):
            self.update_cell_position(entity)

            # Try next direction first
            if entity.next_direction is not None:
                if self.can_move(
                    entity.row, entity.col, entity.next_direction
                ):
                    self.set_direction(entity, entity.next_direction)
                    entity.next_direction = None

            # If no direction, stop
            if entity.direction is None:
                return

            # If wall in current direction, stop
            if not self.can_move(entity.row, entity.col, entity.direction):
                return

        # Move by pixels
        entity.x += entity.col_direction * entity.speed
        entity.y += entity.row_direction * entity.speed

    def update_random_ghost(self, ghost) -> None:
        possible_directions = []

        for direction in ["LEFT", "RIGHT", "UP", "DOWN"]:
            if self.can_move(ghost.row, ghost.col, direction):
                possible_directions.append(direction)

        if not possible_directions:
            return

        if ghost.direction is None or not self.can_move(
            ghost.row, ghost.col, ghost.direction
        ):
            import random

            ghost.next_direction = random.choice(possible_directions)

        self.update_entity(ghost)
