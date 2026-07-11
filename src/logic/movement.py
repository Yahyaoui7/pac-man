from collections import deque

from src.logic.config import CELL_SIZE, EAST, NORTH, SOUTH, WEST
import random


class MovementSystem:
    """Controls movement for player and ghosts."""

    def __init__(self, maze):
        """Store the maze so we can check walls."""
        self.maze = maze
        self.rng = random.Random()

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
            entity.x % CELL_SIZE == CELL_SIZE // 2
            and entity.y % CELL_SIZE == CELL_SIZE // 2
        )

    def update_cell_position(self, entity) -> None:
        """Update row/col using the current pixel position x/y."""
        entity.row = int(entity.y // CELL_SIZE)
        entity.col = int(entity.x // CELL_SIZE)

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
        """Move an entity by pixels, only change direction at cell center."""

        if self.is_centered(entity):
            self.update_cell_position(entity)
            entity.grid_y = entity.row
            entity.grid_x = entity.col

            if entity.next_direction is not None:
                if self.can_move(
                    entity.row,
                    entity.col,
                    entity.next_direction,
                ):
                    self.set_direction(entity, entity.next_direction)
                    entity.next_direction = None

            if entity.direction is None:
                return

            if not self.can_move(entity.row, entity.col, entity.direction):
                return

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

    def _navigate_bfs(self, ghost, target_row: int, target_col: int) -> None:
        """Move ghost toward target using BFS pathfinding."""
        if self.is_centered(ghost):
            self.update_cell_position(ghost)
            path = self.bfs_path((ghost.row, ghost.col), (target_row, target_col))
            if len(path) >= 2:
                direction = self.direction_to_next_cell(path[0], path[1])
                if direction is not None:
                    self.set_direction(ghost, direction)
        self.update_entity(ghost)

    def update_ghost_to_target(self, ghost, target_row, target_col):
        self._navigate_bfs(ghost, target_row, target_col)

    def update_bfs_ghost(self, ghost, player) -> None:
        """Move ghost toward player when ghost is not edible."""
        self._navigate_bfs(ghost, player.row, player.col)

    # ----------------------------
    # EDIBLE GHOST MOVEMENT
    # Ghost runs away from player
    # ----------------------------

    def get_zone(self, row, col):
        middle_row = len(self.maze) // 2
        middle_col = len(self.maze[0]) // 2
        if row < middle_row and col < middle_col:
            return "TOP_LEFT"
        if row < middle_row and col >= middle_col:
            return "TOP_RIGHT"
        if row >= middle_row and col < middle_col:
            return "BOTTOM_LEFT"

        return "BOTTOM_RIGHT"

    def get_zone_bounds(self, zone: str):
        middle_row = len(self.maze) // 2
        middle_col = len(self.maze[0]) // 2
        max_row = len(self.maze) - 1
        max_col = len(self.maze[0]) - 1

        if zone == "TOP_LEFT":
            return (0, middle_row - 1), (0, middle_col - 1)
        if zone == "TOP_RIGHT":
            return (0, middle_row - 1), (middle_col, max_col)
        if zone == "BOTTOM_LEFT":
            return (middle_row, max_row), (0, middle_col - 1)
        return (middle_row, max_row), (middle_col, max_col)

    def is_valid_cell(self, row, col):
        for direction in ["LEFT", "RIGHT", "UP", "DOWN"]:
            if self.can_move(row, col, direction):
                return True

        return False

    def choose_runaway_target_by_zone(self, player):
        player_zone = self.get_zone(player.row, player.col)

        zones = [
            "TOP_LEFT",
            "TOP_RIGHT",
            "BOTTOM_LEFT",
            "BOTTOM_RIGHT",
        ]

        safe_zones = []

        for zone in zones:
            if zone != player_zone:
                safe_zones.append(zone)

        while safe_zones:
            random_zone = self.rng.choice(safe_zones)

            (row_min, row_max), (col_min, col_max) = self.get_zone_bounds(random_zone)

            valid_cells = []

            for row in range(row_min, row_max + 1):
                for col in range(col_min, col_max + 1):
                    if self.is_valid_cell(row, col):
                        valid_cells.append((row, col))

            if valid_cells:
                return self.rng.choice(valid_cells)

            safe_zones.remove(random_zone)

        return None

    # def distance(self, row1: int, col1: int, row2: int, col2: int) -> int:
    #     return abs(row1 - row2) + abs(col1 - col2)

    # def choose_runaway_target(self, player) -> tuple[int, int]:
    #     """Choose the corner farthest from the player."""
    #     max_row = len(self.maze) - 1
    #     max_col = len(self.maze[0]) - 1

    #     corners = [
    #         (0, 0),
    #         (0, max_col),
    #         (max_row, 0),
    #         (max_row, max_col),
    #     ]

    #     best_corner = corners[0]
    #     best_distance = -1

    #     for corner in corners:
    #         dist = self.distance(
    #             corner[0],
    #             corner[1],
    #             player.row,
    #             player.col,
    #         )

    #         if dist > best_distance:
    #             best_distance = dist
    #             best_corner = corner

    #     return best_corner

    def update_runaway_ghost(self, ghost, player) -> None:
        """Move edible ghost away from the player using one fixed random target."""

        if self.is_centered(ghost):
            self.update_cell_position(ghost)

            if ghost.runaway_target is None:
                ghost.runaway_target = self.choose_runaway_target_by_zone(player)

            if ghost.runaway_target == (ghost.row, ghost.col):
                ghost.runaway_target = self.choose_runaway_target_by_zone(player)

            if ghost.runaway_target is not None:
                start = (ghost.row, ghost.col)
                path = self.bfs_path(
                    start,
                    ghost.runaway_target,
                )

                if len(path) >= 2:
                    next_cell = path[1]
                    direction = self.direction_to_next_cell(start, next_cell)

                    if direction is not None:
                        self.set_direction(ghost, direction)
                else:
                    ghost.runaway_target = None

        self.update_entity(ghost)
