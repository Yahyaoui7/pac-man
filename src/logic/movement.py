from collections import deque

# Wall values from maze generator
NORTH = 1
EAST = 2
SOUTH = 4
WEST = 8


class MovementSystem:
    """Controls movement for player and ghosts."""

    def __init__(self, maze):
        """Store the maze so we can check walls."""
        self.maze = maze

    # ----------------------------
    # BASIC ENTITY MOVEMENT
    # Used by player and ghosts
    # ----------------------------

    def set_direction(self, entity, direction: str) -> None:
        """Set entity direction and convert it to row/col movement."""
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
        """Check if entity is exactly in the center of a maze cell."""
        return (
            entity.x % entity.cell_size == entity.cell_size // 2
            and entity.y % entity.cell_size == entity.cell_size // 2
        )

    def update_cell_position(self, entity) -> None:
        """Update row/col using the current pixel position x/y."""
        entity.row = int(entity.y // entity.cell_size)
        entity.col = int(entity.x // entity.cell_size)

    def can_move(self, row: int, col: int, direction: str) -> bool:
        """Check if there is no wall in the wanted direction."""
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
        """Move an entity by pixels, but only change direction at cell center."""

        if self.is_centered(entity):
            self.update_cell_position(entity)

            # Player uses next_direction.
            # This lets Pac-Man continue moving until the player changes direction.
            if entity.next_direction is not None:
                if self.can_move(
                    entity.row, entity.col, entity.next_direction
                ):
                    self.set_direction(entity, entity.next_direction)
                    entity.next_direction = None

            # If entity has no direction, it does not move.
            if entity.direction is None:
                return

            # If there is a wall in front, stop moving.
            if not self.can_move(entity.row, entity.col, entity.direction):
                return

        # Pixel movement.
        entity.x += entity.col_direction * entity.speed
        entity.y += entity.row_direction * entity.speed

    # ----------------------------
    # NOT EDIBLE GHOST MOVEMENT
    # Ghost chases player using BFS
    # ----------------------------

    def get_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """Return all cells the ghost can move to from current cell."""
        neighbors = []

        directions = {
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
            "UP": (-1, 0),
            "DOWN": (1, 0),
        }

        for direction, (d_row, d_col) in directions.items():
            if self.can_move(row, col, direction):
                neighbors.append((row + d_row, col + d_col))

        return neighbors

    def bfs_path(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """Find the shortest path from ghost to player."""
        queue = deque([start])
        visited = {start}
        parent = {}

        while queue:
            current = queue.popleft()

            if current == target:
                path = []
                while current != start:
                    path.append(current)
                    current = parent[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current[0], current[1]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)

        return []

    def direction_to_next_cell(
        self,
        current: tuple[int, int],
        next_cell: tuple[int, int],
    ) -> str | None:
        """Convert next cell into a direction string."""
        row, col = current
        next_row, next_col = next_cell

        if next_row == row and next_col == col - 1:
            return "LEFT"
        if next_row == row and next_col == col + 1:
            return "RIGHT"
        if next_row == row - 1 and next_col == col:
            return "UP"
        if next_row == row + 1 and next_col == col:
            return "DOWN"

        return None

    def update_bfs_ghost(self, ghost, player) -> None:
        """Move ghost toward player when ghost is not edible."""
        if self.is_centered(ghost):
            self.update_cell_position(ghost)

            start = (ghost.row, ghost.col)
            target = (player.row, player.col)

            path = self.bfs_path(start, target)

            if len(path) >= 2:
                direction = self.direction_to_next_cell(path[0], path[1])
                if direction is not None:
                    self.set_direction(ghost, direction)

        self.update_entity(ghost)

    # ----------------------------
    # EDIBLE GHOST MOVEMENT
    # Ghost runs away from player
    # ----------------------------

    def get_next_position(
        self,
        row: int,
        col: int,
        direction: str,
    ) -> tuple[int, int]:
        """Return the next cell if entity moves in this direction."""
        if direction == "LEFT":
            return row, col - 1
        if direction == "RIGHT":
            return row, col + 1
        if direction == "UP":
            return row - 1, col
        if direction == "DOWN":
            return row + 1, col

        return row, col

    def distance(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Calculate distance between two cells."""
        return abs(row1 - row2) + abs(col1 - col2)

    def choose_runaway_direction(self, ghost, player) -> str | None:
        """Choose the direction that makes ghost farthest from player."""
        best_direction = None
        best_distance = -1

        for direction in ["LEFT", "RIGHT", "UP", "DOWN"]:
            if self.can_move(ghost.row, ghost.col, direction):
                next_row, next_col = self.get_next_position(
                    ghost.row,
                    ghost.col,
                    direction,
                )

                dist = self.distance(
                    next_row,
                    next_col,
                    player.row,
                    player.col,
                )

                if dist > best_distance:
                    best_distance = dist
                    best_direction = direction

        return best_direction

    def update_runaway_ghost(self, ghost, player) -> None:
        """Move ghost away from player when ghost is edible."""
        if self.is_centered(ghost):
            self.update_cell_position(ghost)

            direction = self.choose_runaway_direction(ghost, player)

            if direction is not None:
                self.set_direction(ghost, direction)

        self.update_entity(ghost)
