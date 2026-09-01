from collections import deque
from typing import List, Optional, Any, Tuple

from src.graphics.entitys.entity import Entity
from src.logic.config import CELL_SIZE, EAST, NORTH, SOUTH, WEST
import random


class MovementSystem:
    """Controls movement for player and ghosts."""

    def __init__(self, maze: List[List[int]]) -> None:
        """Store the maze so we can check walls."""
        self.maze = maze
        self.rng = random.Random()
        self.pattern_42: Optional[List[Tuple[int, int]]] = None
        self._dist_cache: dict[tuple[int, int], list[int]] = {}

    def clear_cache(self) -> None:
        """Clear precomputed distance cache if maze structure changes."""
        self._dist_cache.clear()

    def set_direction(self, entity: Entity, direction: str) -> None:
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

    def is_centered(self, entity: Entity) -> bool:
        tolerance = 5

        cell_grid_x = int(entity.x // CELL_SIZE)
        cell_grid_y = int(entity.y // CELL_SIZE)

        center_x = cell_grid_x * CELL_SIZE + CELL_SIZE // 2
        center_y = cell_grid_y * CELL_SIZE + CELL_SIZE // 2

        if (
            abs(entity.x - center_x) <= tolerance
            and abs(entity.y - center_y) <= tolerance
        ):
            return True
        else:
            return False

    def update_cell_position(self, entity: Entity) -> None:
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

        if not (0 <= grid_y < len(self.maze)) or not (
            0 <= grid_x < len(self.maze[0])
        ):
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

    def is_opposite_direction(self, entity: Entity) -> bool:
        opposite_direction = {
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
            "UP": "DOWN",
            "DOWN": "UP",
        }

        d = entity.direction
        nd = entity.next_direction
        if d is None or nd is None:
            return False

        return d in opposite_direction and opposite_direction[d] == nd

    def update_entity(self, entity: Entity) -> None:
        """Move an entity by pixels, only change direction at cell center."""
        if self.is_opposite_direction(entity):
            assert entity.next_direction is not None
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

    def bfs_distances_uncached(self, source: tuple[int, int]) -> list[int]:
        """Single BFS from source returning flat distance array (uncached fallback)."""
        h = len(self.maze)
        w = len(self.maze[0])
        dist = [-1] * (h * w)
        sy, sx = source
        if not (0 <= sy < h and 0 <= sx < w):
            return dist
        dist[sy * w + sx] = 0
        queue = deque([source])
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        wall_bits = [NORTH, SOUTH, WEST, EAST]
        while queue:
            cy, cx = queue.popleft()
            cd = dist[cy * w + cx]
            cell = self.maze[cy][cx]
            for i, (dy, dx) in enumerate(directions):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not (cell & wall_bits[i]):
                    idx = ny * w + nx
                    if dist[idx] == -1:
                        dist[idx] = cd + 1
                        queue.append((ny, nx))
        return dist

    def bfs_distances(self, source: tuple[int, int]) -> list[int]:
        """Return flat distance array from source to all cells (O(1) cached)."""
        if self.pattern_42 is not None:
            return self.bfs_distances_uncached(source)
        if source not in self._dist_cache:
            self._dist_cache[source] = self.bfs_distances_uncached(source)
        return self._dist_cache[source]

    def bfs_path(
        self, start: tuple[int, int], target: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Find the shortest path from start to target in O(path_length) using distance cache."""
        if start == target:
            return [start]
        if self.pattern_42 is not None:
            queue = deque([start])
            visited = {start}
            parent: dict[tuple[int, int], tuple[int, int]] = {}
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

        w = len(self.maze[0])
        target_dists = self.bfs_distances(target)
        start_dist = target_dists[start[0] * w + start[1]]
        if start_dist < 0:
            return []

        path = [start]
        curr = start
        while curr != target:
            curr_dist = target_dists[curr[0] * w + curr[1]]
            best_nbr = None
            best_dist = curr_dist
            for nbr in self.get_neighbors(curr[0], curr[1]):
                d = target_dists[nbr[0] * w + nbr[1]]
                if 0 <= d < best_dist:
                    best_dist = d
                    best_nbr = nbr
            if best_nbr is None:
                break
            path.append(best_nbr)
            curr = best_nbr

        return path if path[-1] == target else []

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

    def move_inside_prison(self, ghost: Any) -> None:
        self.pattern_42 = ghost.prison_cells

        try:
            if self.is_centered(ghost):
                self.update_cell_position(ghost)
                if not ghost.prison_cells:
                    return
                high_cell = min(ghost.prison_cells, key=lambda cell: cell[0])
                low_cell = max(ghost.prison_cells, key=lambda cell: cell[0])

                current_cell = (ghost.grid_y, ghost.grid_x)

                if ghost.prison_target is None:
                    ghost.prison_target = low_cell

                if current_cell == ghost.prison_target:
                    if ghost.prison_target == low_cell:
                        ghost.prison_target = high_cell
                    else:
                        ghost.prison_target = low_cell

                path = self.bfs_path(current_cell, ghost.prison_target)

                if len(path) >= 2:
                    next_cell = path[1]
                    direction = self.direction_to_next_cell(
                        current_cell,
                        next_cell,
                    )

                    if direction is not None:
                        self.set_direction(ghost, direction)

            self.update_entity(ghost)

        finally:
            self.pattern_42 = None

    def _navigate_bfs(
        self,
        ghost: Any,
        target_grid_y: int,
        target_grid_x: int,
    ) -> None:
        """Move ghost toward target using direct O(1) distance matrix lookup."""
        if self.is_centered(ghost):
            self.update_cell_position(ghost)
            start = (ghost.grid_y, ghost.grid_x)
            target = (target_grid_y, target_grid_x)

            if start != target:
                if self.pattern_42 is not None:
                    path = self.bfs_path(start, target)
                    if len(path) >= 2:
                        next_cell = path[1]
                        direction = self.direction_to_next_cell(
                            start, next_cell
                        )
                        if direction is not None:
                            self.set_direction(ghost, direction)
                else:
                    w = len(self.maze[0])
                    target_dists = self.bfs_distances(target)
                    curr_d = target_dists[start[0] * w + start[1]]
                    if curr_d > 0:
                        best_nbr = None
                        best_dist = curr_d
                        for nbr in self.get_neighbors(start[0], start[1]):
                            d = target_dists[nbr[0] * w + nbr[1]]
                            if 0 <= d < best_dist:
                                best_dist = d
                                best_nbr = nbr
                        if best_nbr is not None:
                            direction = self.direction_to_next_cell(
                                start, best_nbr
                            )
                            if direction is not None:
                                self.set_direction(ghost, direction)

        self.update_entity(ghost)

    def update_ghost_to_target(
        self,
        ghost: Any,
        target_y: int,
        target_x: int,
    ) -> None:
        self.pattern_42 = ghost.prison_cells

        try:
            self._navigate_bfs(ghost, target_y, target_x)
        finally:
            self.pattern_42 = None

    def update_bfs_ghost(self, ghost: Any, player: Any) -> None:
        """Move ghost toward player when ghost is not edible."""
        self.pattern_42 = None
        self._navigate_bfs(ghost, player.grid_y, player.grid_x)

    def update_predictive_ghost(
        self,
        ghost: Any,
        player: Any,
        lookahead: int,
        maze: Optional[Any] = None,
        pellets: Optional[Any] = None,
        ghosts: Optional[Any] = None,
    ) -> None:
        """Predict Pac-Man's future position and navigate the ghost toward it."""
        self.pattern_42 = None

        DIRECTION_DELTA: dict[str, tuple[int, int]] = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1),
        }

        def get_valid_directions(
            y: int, x: int, current_dir: str
        ) -> list[str]:
            valid = []

            for d in DIRECTION_DELTA:
                if self.can_move(y, x, d):
                    valid.append(d)
            return valid

        direction = getattr(player, "next_direction", None) or player.direction
        y, x = player.grid_y, player.grid_x

        if lookahead > 0:
            if self.can_move(y, x, direction):
                dy, dx = DIRECTION_DELTA.get(direction, (0, 0))
                y += dy
                x += dx
                remaining = lookahead - 1
            else:
                remaining = lookahead
        else:
            remaining = 0

        if (
            remaining > 0
            and maze is not None
            and pellets is not None
            and ghosts is not None
        ):
            try:
                from src.logic.expert import PacmanExpert
                from AI_arena.data.formatter import DIRECTIONS

                for _ in range(remaining):
                    valid_directions = get_valid_directions(y, x, direction)

                    if not valid_directions:
                        break

                    if len(valid_directions) == 1:
                        predicted_dir = valid_directions[0]
                    else:

                        class _FakePlayer:
                            pass

                        class _FakeEnv:
                            pass

                        fake_player = _FakePlayer()
                        fake_player.grid_y = y
                        fake_player.grid_x = x
                        fake_player.direction = direction
                        fake_player.next_direction = direction
                        fake_player.powered_timer = float(
                            getattr(player, "powered_timer", 0.0)
                        )

                        fake_env = _FakeEnv()
                        fake_env.movement = self
                        fake_env.player = fake_player
                        fake_env.maze = maze
                        fake_env.pellets = pellets
                        fake_env.ghosts = ghosts

                        expert = PacmanExpert(horizon=remaining)
                        decision = expert.choose_action(fake_env)

                        predicted_dir = DIRECTIONS[decision.action]

                        if predicted_dir not in valid_directions:
                            predicted_dir = valid_directions[0]

                    dy, dx = DIRECTION_DELTA.get(predicted_dir, (0, 0))
                    if self.can_move(y, x, predicted_dir):
                        y += dy
                        x += dx
                        direction = predicted_dir
                    else:
                        break

            except Exception:
                dy, dx = DIRECTION_DELTA.get(direction, (0, 0))
                for _ in range(remaining):
                    if self.can_move(y, x, direction):
                        y += dy
                        x += dx
                    else:
                        break

        self._navigate_bfs(ghost, y, x)

    def update_cnn_ghost(
        self,
        ghost: Any,
        predicted_direction: str,
    ) -> None:
        """Apply a legal CNN direction at a cell center and move the ghost.

        Assumes update_cell_position was already called by the game loop
        when the ghost was detected as centered.
        """
        self.pattern_42 = None
        if self.is_centered(ghost):
            if self.can_move(
                ghost.grid_y,
                ghost.grid_x,
                predicted_direction,
            ):
                self.set_direction(ghost, predicted_direction)
        self.update_entity(ghost)

    # ----------------------------
    # EDIBLE GHOST MOVEMENT
    # Ghost runs away from player
    # ----------------------------

    def get_zone(self, grid_y: int, grid_x: int) -> str:
        middle_grid_y = len(self.maze) // 2
        middle_grid_x = len(self.maze[0]) // 2
        if grid_y < middle_grid_y and grid_x < middle_grid_x:
            return "TOP_LEFT"
        if grid_y < middle_grid_y and grid_x >= middle_grid_x:
            return "TOP_RIGHT"
        if grid_y >= middle_grid_y and grid_x < middle_grid_x:
            return "BOTTOM_LEFT"

        return "BOTTOM_RIGHT"

    def get_zone_bounds(self, zone: str) -> Tuple[
        Tuple[int, int],
        Tuple[int, int],
    ]:
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

    def is_valid_cell(self, grid_y: int, grid_x: int) -> bool:
        for direction in ["LEFT", "RIGHT", "UP", "DOWN"]:
            if self.can_move(grid_y, grid_x, direction):
                return True

        return False

    def choose_runaway_target_by_zone(
        self,
        player: Any,
    ) -> Optional[Tuple[int, int]]:
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

    def update_runaway_ghost(self, ghost: Any, player: Any) -> None:
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

    def get_bfs_next_move(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[list[str | None], int] | None:
        """Return directions and path length from start to target.

        directions_list[i] is the direction from path[i] to path[i+1].
        The first direction is the immediate next move.
        """
        start_yx = (start[1], start[0])
        target_yx = (target[1], target[0])
        path = self.bfs_path(start_yx, target_yx)
        if len(path) < 2:
            return None
        res = []
        for pt in range(len(path) - 1):
            res.append(self.direction_to_next_cell(path[pt], path[pt + 1]))
        return res, len(path)

    def get_runaway_next_move(
        self,
        ghost_position: tuple[int, int],
        player_position: tuple[int, int],
    ) -> tuple[list[str | None], int] | None:
        ghost_yx = (ghost_position[1], ghost_position[0])
        player_yx = (player_position[1], player_position[0])

        target = self.choose_runaway_target(player_yx)
        if target is None:
            return None

        path = self.bfs_path(ghost_yx, target)
        if len(path) < 2:
            return None
        res = []
        for pt in range(len(path) - 1):
            res.append(self.direction_to_next_cell(path[pt], path[pt + 1]))
        return res, len(path)

    def choose_runaway_target(
        self,
        player_yx: tuple[int, int],
    ) -> Optional[Tuple[int, int]]:
        player_zone = self.get_zone(player_yx[0], player_yx[1])

        zones = ["TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"]
        safe_zones = [z for z in zones if z != player_zone]

        while safe_zones:
            random_zone = self.rng.choice(safe_zones)
            (y_min, y_max), (x_min, x_max) = self.get_zone_bounds(random_zone)

            valid_cells = []
            for gy in range(y_min, y_max + 1):
                for gx in range(x_min, x_max + 1):
                    if self.is_valid_cell(gy, gx):
                        valid_cells.append((gy, gx))

            if valid_cells:
                return self.rng.choice(valid_cells)
            safe_zones.remove(random_zone)

        return None
