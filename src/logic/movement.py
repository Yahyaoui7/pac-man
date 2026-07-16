from collections import deque

from src.logic.config import CELL_SIZE, EAST, NORTH, SOUTH, WEST
import random
import math


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
        """Set entity direction and convert it to grid_y/grid_x movement."""
        entity.direction = direction

        if direction == "LEFT":
            entity.grid_y_direction = 0
            entity.grid_x_direction = -1
        elif direction == "RIGHT":
            entity.grid_y_direction = 0
            entity.grid_x_direction = 1
        elif direction == "UP":
            entity.grid_y_direction = -1
            entity.grid_x_direction = 0
        elif direction == "DOWN":
            entity.grid_y_direction = 1
            entity.grid_x_direction = 0

    def is_centered(self, entity) -> bool:
        tolerance = 5

        cell_grid_x = int(entity.x // CELL_SIZE)
        cell_grid_y = int(entity.y // CELL_SIZE)

        center_x = cell_grid_x * CELL_SIZE + CELL_SIZE // 2
        center_y = cell_grid_y * CELL_SIZE + CELL_SIZE // 2

        return (
            abs(entity.x - center_x) <= tolerance
            and abs(entity.y - center_y) <= tolerance
        )

    def update_cell_position(self, entity) -> None:
        """Update grid_y/grid_x using the current pixel position x/y."""
        entity.grid_y = int(entity.y // CELL_SIZE)
        entity.grid_x = int(entity.x // CELL_SIZE)

    def can_move(self, grid_y: int, grid_x: int, direction: str) -> bool:
        directions = {
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
            "UP": (-1, 0),
            "DOWN": (1, 0),
        }

        if direction not in directions:
            return False

        d_grid_y, d_grid_x = directions[direction]
        next_y = grid_y + d_grid_y
        next_x = grid_x + d_grid_x

        if not (0 <= next_y < len(self.maze)):
            return False

        if not (0 <= next_x < len(self.maze[0])):
            return False

        next_cell = (next_y, next_x)

        if self.pattern_42 and next_cell in self.pattern_42:
            return True

        cell = self.maze[grid_y][grid_x]

        if direction == "LEFT":
            return not (cell & WEST)
        if direction == "RIGHT":
            return not (cell & EAST)
        if direction == "UP":
            return not (cell & NORTH)
        if direction == "DOWN":
            return not (cell & SOUTH)

        return False

    def is_opposite_direction(self, entity) -> bool:
        opposite_direction = {
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
            "UP": "DOWN",
            "DOWN": "UP",
        }

        if entity.direction is None or entity.next_direction is None:
            return False

        return opposite_direction[entity.direction] == entity.next_direction

    def update_entity(self, entity) -> None:
        """Move an entity by pixels, only change direction at cell center."""

        if self.is_opposite_direction(entity):
            self.set_direction(entity, entity.next_direction)
            entity.next_direction = None

        if self.is_centered(entity):
            self.update_cell_position(entity)

            if entity.next_direction is not None:
                if self.can_move(
                    entity.grid_y,
                    entity.grid_x,
                    entity.next_direction,
                ):
                    self.set_direction(entity, entity.next_direction)
                    entity.next_direction = None

            if entity.direction is None:
                return

            if not self.can_move(
                entity.grid_y,
                entity.grid_x,
                entity.direction,
            ):
                return

        entity.x += entity.grid_x_direction * entity.speed
        entity.y += entity.grid_y_direction * entity.speed

    # ----------------------------
    # NOT EDIBLE GHOST MOVEMENT
    # Ghost chases player using BFS
    # ----------------------------

    def get_neighbors(self, grid_y: int, grid_x: int) -> list[tuple[int, int]]:
        """Return all cells the ghost can move to from current cell."""
        neighbors = []

        directions = {
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
            "UP": (-1, 0),
            "DOWN": (1, 0),
        }

        for direction, (d_grid_y, d_grid_x) in directions.items():
            next_cell = (grid_y + d_grid_y, grid_x + d_grid_x)
            if self.can_move(grid_y, grid_x, direction):
                neighbors.append(next_cell)

        return neighbors

    def bfs_path(
        self, start: tuple[int, int], target: tuple[int, int]
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
        grid_y, grid_x = current
        next_grid_y, next_grid_x = next_cell

        if next_grid_y == grid_y and next_grid_x == grid_x - 1:
            return "LEFT"
        if next_grid_y == grid_y and next_grid_x == grid_x + 1:
            return "RIGHT"
        if next_grid_y == grid_y - 1 and next_grid_x == grid_x:
            return "UP"
        if next_grid_y == grid_y + 1 and next_grid_x == grid_x:
            return "DOWN"

        return None

    def move_inside_prison(self, ghost, entity, pattern_42):
        self.pattern_42 = pattern_42

        try:
            if self.is_centered(ghost):
                self.update_cell_position(ghost)

                neighbors = []

                directions = {
                    "LEFT": (0, -1),
                    "RIGHT": (0, 1),
                    "UP": (-1, 0),
                    "DOWN": (1, 0),
                }

                for direction, (d_grid_y, d_grid_x) in directions.items():
                    next_cell = (
                        ghost.grid_y + d_grid_y,
                        ghost.grid_x + d_grid_x,
                    )

                    if next_cell in entity.pattern_42_cells:
                        neighbors.append(direction)

                if neighbors:
                    dirc = random.choice(neighbors)
                    self.set_direction(ghost, dirc)

            self.update_entity(ghost)

        finally:
            self.pattern_42 = None

    def _navigate_bfs(
        self, ghost, target_grid_y: int, target_grid_x: int
    ) -> None:
        """Move ghost toward target using BFS pathfinding."""
        if self.is_centered(ghost):
            self.update_cell_position(ghost)
            start = (ghost.grid_y, ghost.grid_x)
            target = (target_grid_y, target_grid_x)

            path = self.bfs_path(start, target)

            if len(path) >= 2:
                next_cell = path[1]
                direction = self.direction_to_next_cell(start, next_cell)

                if direction is not None:
                    self.set_direction(ghost, direction)

        self.update_entity(ghost)

    # def update_bfs_ghost(self, ghost, player) -> None:
    #     """Move ghost toward player when ghost is not edible."""
    #     if self.is_centered(ghost):
    #         self.update_cell_position(ghost)

    #         start = (ghost.grid_y, ghost.grid_x)
    #         target = (player.grid_y, player.grid_x)

    #         path = self.bfs_path(start, target)

    #         if len(path) >= 2:
    #             direction = self.direction_to_next_cell(path[0], path[1])
    #             if direction is not None:
    #                 self.set_direction(ghost, direction)
    #     self.update_entity(ghost)
    # def update_ghost_to_prison_target(
    #     self, ghost, target_grid_y, target_grid_x
    # ):
    #     target_x = target_grid_x * CELL_SIZE + CELL_SIZE // 2
    #     target_y = target_grid_y * CELL_SIZE + CELL_SIZE // 2

    #     dx = target_x - ghost.x
    #     dy = target_y - ghost.y
    #     distance = math.hypot(dx, dy)

    #     if distance <= ghost.speed:
    #         ghost.x = target_x
    #         ghost.y = target_y
    #         ghost.grid_x = target_grid_x
    #         ghost.grid_y = target_grid_y
    #         return

    #     ghost.x += ghost.speed * dx / distance
    #     ghost.y += ghost.speed * dy / distance

    def update_ghost_to_target(
        self,
        ghost,
        target_y,
        target_x,
        pattern_42=None,
    ):
        self.pattern_42 = pattern_42

        try:
            self._navigate_bfs(ghost, target_y, target_x)
        finally:
            self.pattern_42 = None

    def update_bfs_ghost(self, ghost, player) -> None:
        """Move ghost toward player when ghost is not edible."""
        self.pattern_42 = None
        self._navigate_bfs(ghost, player.grid_y, player.grid_x)

    # ----------------------------
    # EDIBLE GHOST MOVEMENT
    # Ghost runs away from player
    # ----------------------------

    def get_zone(self, grid_y, grid_x):
        middle_grid_y = len(self.maze) // 2
        middle_grid_x = len(self.maze[0]) // 2
        if grid_y < middle_grid_y and grid_x < middle_grid_x:
            return "TOP_LEFT"
        if grid_y < middle_grid_y and grid_x >= middle_grid_x:
            return "TOP_RIGHT"
        if grid_y >= middle_grid_y and grid_x < middle_grid_x:
            return "BOTTOM_LEFT"

        return "BOTTOM_RIGHT"

    def get_zone_bounds(self, zone: str):
        middle_grid_y = len(self.maze) // 2
        middle_grid_x = len(self.maze[0]) // 2
        max_grid_y = len(self.maze) - 1
        max_grid_x = len(self.maze[0]) - 1

        if zone == "TOP_LEFT":
            return (0, middle_grid_y - 1), (0, middle_grid_x - 1)
        if zone == "TOP_RIGHT":
            return (0, middle_grid_y - 1), (middle_grid_x, max_grid_x)
        if zone == "BOTTOM_LEFT":
            return (middle_grid_y, max_grid_y), (0, middle_grid_x - 1)
        return (middle_grid_y, max_grid_y), (middle_grid_x, max_grid_x)

    def is_valid_cell(self, grid_y, grid_x):
        for direction in ["LEFT", "RIGHT", "UP", "DOWN"]:
            if self.can_move(grid_y, grid_x, direction):
                return True

        return False

    def choose_runaway_target_by_zone(self, player):
        player_zone = self.get_zone(player.grid_y, player.grid_x)

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

            (
                grid_y_min,
                grid_y_max,
            ), (
                grid_x_min,
                grid_x_max,
            ) = self.get_zone_bounds(random_zone)

            valid_cells = []

            for grid_y in range(grid_y_min, grid_y_max + 1):
                for grid_x in range(grid_x_min, grid_x_max + 1):
                    if self.is_valid_cell(grid_y, grid_x):
                        valid_cells.append((grid_y, grid_x))

            if valid_cells:
                return self.rng.choice(valid_cells)

            safe_zones.remove(random_zone)

        return None

    # def distance(
    #     self,
    #     grid_y1: int,
    #     grid_x1: int,
    #     grid_y2: int,
    #     grid_x2: int,
    # ) -> int:
    #     return abs(grid_y1 - grid_y2) + abs(grid_x1 - grid_x2)

    # def choose_runaway_target(self, player) -> tuple[int, int]:
    #     """Choose the corner farthest from the player."""
    #     max_grid_y = len(self.maze) - 1
    #     max_grid_x = len(self.maze[0]) - 1

    #     corners = [
    #         (0, 0),
    #         (0, max_grid_x),
    #         (max_grid_y, 0),
    #         (max_grid_y, max_grid_x),
    #     ]

    #     best_corner = corners[0]
    #     best_distance = -1

    #     for corner in corners:
    #         dist = self.distance(
    #             corner[0],
    #             corner[1],
    #             player.grid_y,
    #             player.grid_x,
    #         )

    #         if dist > best_distance:
    #             best_distance = dist
    #             best_corner = corner

    #     return best_corner

    def update_runaway_ghost(self, ghost, player) -> None:
        """Move edible ghost away from the player using fixed random target."""

        if self.is_centered(ghost):
            self.update_cell_position(ghost)

            if ghost.runaway_target is None:
                ghost.runaway_target = self.choose_runaway_target_by_zone(
                    player,
                )

            if ghost.runaway_target == (ghost.grid_y, ghost.grid_x):
                ghost.runaway_target = self.choose_runaway_target_by_zone(
                    player,
                )

            if ghost.runaway_target is not None:
                start = (ghost.grid_y, ghost.grid_x)
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
