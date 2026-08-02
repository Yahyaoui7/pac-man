import argparse
import random
import time
from pathlib import Path
from typing import Literal, TextIO, cast
# from AI_arena.data import CNNJSONLDataset, EPISODE_LENGTH

from AI_arena.data_collector.formatters import (
    CNNFormatter,
    StreamWriter,
)
from AI_arena.data_collector.models import (
    LOCAL_PELLET_RADIUS,
    Direction,
    GhostState,
    TrainingSample,
    WorldState,
)
from mazegenerator import MazeGenerator  # type: ignore
from src.logic.movement import MovementSystem  # type: ignore

GHOST_NAMES = ["Blinky", "Pinky", "Inky", "Clyde"]
MIN_MAZE_WIDTH = 10
MAX_MAZE_WIDTH = 25
MIN_MAZE_HEIGHT = 10
MAX_MAZE_HEIGHT = 50
# Include powered states in the synthetic dataset so the models can learn both
# chase and frightened behavior instead of seeing constant zero mode features.
POWERED_SAMPLE_PROBABILITY = 0.35
MAX_FRIGHTENED_TIMER = 10
DEFAULT_CNN_PATH = Path(__file__).parents[1] / "data" / "CNN_DATA.jsonl"

DIRECTION_DELTAS = {
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
    "UP": (-1, 0),
    "DOWN": (1, 0),
}


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def count_local_pellets(
    pellets: list[list[int]], x: int, y: int, radius: int = LOCAL_PELLET_RADIUS
) -> int:
    h = len(pellets)
    w = len(pellets[0])
    count = 0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                count += pellets[ny][nx]
    return count


def choose_frightened_direction(
    movement: MovementSystem,
    ghost_position: tuple[int, int],
    player_position: tuple[int, int],
    available_moves: list[str],
) -> str:
    """Choose a valid move that maximizes distance from the player."""
    gx, gy = ghost_position
    px, py = player_position
    scored_moves = []

    for direction in available_moves:
        dy, dx = DIRECTION_DELTAS[direction]
        next_yx = (gy + dy, gx + dx)
        # Score each legal next cell by its shortest-path distance back to the
        # player. A larger value gives the frightened ghost an escape target.
        path = movement.bfs_path(next_yx, (py, px))
        distance = len(path) if path else float("inf")
        scored_moves.append((distance, direction))

    max_distance = max(distance for distance, _ in scored_moves)
    best_moves = [
        direction
        for distance, direction in scored_moves
        if distance == max_distance
    ]
    return random.choice(best_moves)


def generate_pellets(width, height, valid_cells, player, ghosts):
    pellets = [[0 for _ in range(width)] for _ in range(height)]

    for x, y in valid_cells:
        pellets[y][x] = random.choices(
            population=[0, 1, 2],
            weights=[10, 87, 3],
            k=1,
        )[0]

    px, py = player
    pellets[py][px] = 0

    for gx, gy in ghosts:
        pellets[gy][gx] = 0

    return pellets


def get_randoms() -> TrainingSample:
    # Choose dimensions for every sample so the CNN learns across differently
    # sized mazes. CNNFormatter pads each result to the fixed 25x50 tensor.
    width = random.randint(MIN_MAZE_WIDTH, MAX_MAZE_WIDTH)
    height = random.randint(MIN_MAZE_HEIGHT, MAX_MAZE_HEIGHT)
    seed = random.randint(0, 88888)

    maze = MazeGenerator(
        size=(width, height),
        perfect=False,
        entry_cell=(0, 0),
        exit_cell=(1, 1),
        seed=seed,
    )

    valid_cells = []

    for y in range(height):
        for x in range(width):
            if maze.maze[y][x] != 15:
                valid_cells.append((x, y))

    if len(valid_cells) < 5:
        return get_randoms()

    player = random.choice(valid_cells)

    valid_cells.remove(player)

    ghost_positions = random.sample(valid_cells, 4)

    pellets = generate_pellets(
        width,
        height,
        valid_cells,
        player,
        ghost_positions,
    )
    # Power mode is sampled once per world because one super pellet affects
    # the player and all four ghosts at the same time.
    player_powered = random.random() < POWERED_SAMPLE_PROBABILITY
    frightened_timer = (
        random.randint(1, MAX_FRIGHTENED_TIMER) if player_powered else 0
    )
    player_available_moves = []
    movement = MovementSystem(maze.maze)
    for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
        # MovementSystem accepts coordinates in (row/y, column/x) order.
        if movement.can_move(player[1], player[0], d):
            player_available_moves.append(d)

    ghosts = []

    for name, (gx, gy) in zip(GHOST_NAMES, ghost_positions):
        result = movement.get_bfs_next_move((gx, gy), player)

        if result is not None:
            chase_directions, path_length = result
        else:
            chase_directions, path_length = [], 0

        bfs_path = []
        if result is not None:
            cur_yx = (gy, gx)
            bfs_path = [cur_yx]
            for path_direction in chase_directions:
                if path_direction is None:
                    break
                dy, dx = DIRECTION_DELTAS[path_direction]
                cur_yx = (cur_yx[0] + dy, cur_yx[1] + dx)
                bfs_path.append(cur_yx)
            bfs_path = [(x, y) for y, x in bfs_path]

        available_moves = []
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            if movement.can_move(gy, gx, d):
                available_moves.append(d)

        if not chase_directions or not available_moves:
            continue

        mode: Literal["CHASE", "FRIGHTENED"] = (
            "FRIGHTENED" if player_powered else "CHASE"
        )
        bfs_directions: list[str | None]
        if mode == "FRIGHTENED":
            # Store an escape direction as the supervised label. Reusing the
            # chase BFS label here would teach frightened ghosts to approach.
            bfs_directions = [
                choose_frightened_direction(
                    movement,
                    (gx, gy),
                    player,
                    available_moves,
                )
            ]
        else:
            bfs_directions = chase_directions

        manhattan_dist = manhattan((gx, gy), player)
        local_pellets = count_local_pellets(pellets, gx, gy)
        num_exits = len(available_moves)

        ghosts.append(
            GhostState(
                name=name,
                position=(gx, gy),
                mode=mode,
                distance_to_player=path_length,
                bfs_path=bfs_path,
                bfs_directions=bfs_directions,
                path_length=path_length,
                available_moves=available_moves,
                manhattan_distance=manhattan_dist,
                local_pellet_count=local_pellets,
                num_exits=num_exits,
                frightened_timer=frightened_timer,
                previous_direction="NONE",
            )
        )

    if len(ghosts) != 4:
        return get_randoms()

    sample = TrainingSample(
        metadata={
            "seed": seed,
            "maze_width": width,
            "maze_height": height,
            "tick": 0,
        },
        world=WorldState(
            maze=maze.maze,
            pellets=pellets,
            player_available_moves=player_available_moves,
            player_position=player,
            player_direction="NONE",
            player_powered=player_powered,
            remaining_pellets=sum(
                cell == 1 for row in pellets for cell in row
            ),
            remaining_super_pellets=sum(
                cell == 2 for row in pellets for cell in row
            ),
        ),
        ghosts=ghosts,
    )

    return sample


def advance_sample(
    sample: TrainingSample,
    episode_id: int,
    episode_step: int,
) -> TrainingSample:
    """Advance Pac-Man and all ghosts by one legal teacher-controlled move."""

    maze = sample.world.maze
    pellets = [row.copy() for row in sample.world.pellets]
    movement = MovementSystem(maze)

    player_x, player_y = sample.world.player_position
    player_moves = [
        direction
        for direction in DIRECTION_DELTAS
        if movement.can_move(player_y, player_x, direction)
    ]
    if not player_moves:
        raise ValueError("Pac-Man has no legal move in generated episode")
    player_direction = cast(Direction, random.choice(player_moves))
    player_dy, player_dx = DIRECTION_DELTAS[player_direction]
    player = (player_x + player_dx, player_y + player_dy)
    pellets[player[1]][player[0]] = 0

    ghost_positions = []
    ghost_directions: list[Direction] = []
    for ghost in sample.ghosts:
        if not ghost.bfs_directions or ghost.bfs_directions[0] is None:
            raise ValueError(f"{ghost.name} has no teacher action")
        direction = ghost.bfs_directions[0]
        typed_direction = cast(Direction, direction)
        ghost_directions.append(typed_direction)
        ghost_dy, ghost_dx = DIRECTION_DELTAS[direction]
        ghost_positions.append(
            (
                ghost.position[0] + ghost_dx,
                ghost.position[1] + ghost_dy,
            )
        )

    powered = sample.world.player_powered
    frightened_timer = max(
        0,
        max((ghost.frightened_timer for ghost in sample.ghosts), default=0)
        - 1,
    )
    ghosts = []
    for name, (ghost_x, ghost_y), previous_direction in zip(
        GHOST_NAMES,
        ghost_positions,
        ghost_directions,
    ):
        available_moves = [
            direction
            for direction in DIRECTION_DELTAS
            if movement.can_move(ghost_y, ghost_x, direction)
        ]
        result = movement.get_bfs_next_move(
            (ghost_x, ghost_y),
            player,
        )
        if result is None or not available_moves:
            raise ValueError(f"{name} cannot continue generated episode")
        chase_directions, path_length = result
        mode: Literal["CHASE", "FRIGHTENED"] = (
            "FRIGHTENED" if powered else "CHASE"
        )
        if mode == "FRIGHTENED":
            teacher_directions: list[str | None] = [
                choose_frightened_direction(
                    movement,
                    (ghost_x, ghost_y),
                    player,
                    available_moves,
                )
            ]
        else:
            teacher_directions = chase_directions

        path_yx = movement.bfs_path(
            (ghost_y, ghost_x),
            (player[1], player[0]),
        )
        bfs_path = [(x, y) for y, x in path_yx]
        ghosts.append(
            GhostState(
                name=name,
                position=(ghost_x, ghost_y),
                mode=mode,
                distance_to_player=path_length,
                bfs_path=bfs_path,
                bfs_directions=teacher_directions,
                path_length=path_length,
                available_moves=available_moves,
                manhattan_distance=manhattan((ghost_x, ghost_y), player),
                local_pellet_count=count_local_pellets(
                    pellets,
                    ghost_x,
                    ghost_y,
                ),
                num_exits=len(available_moves),
                frightened_timer=frightened_timer,
                previous_direction=previous_direction,
            )
        )

    return TrainingSample(
        metadata={
            **sample.metadata,
            "episode_id": episode_id,
            "episode_step": episode_step,
        },
        world=WorldState(
            maze=maze,
            pellets=pellets,
            player_available_moves=[
                direction
                for direction in DIRECTION_DELTAS
                if movement.can_move(player[1], player[0], direction)
            ],
            player_position=player,
            player_direction=player_direction,
            player_powered=powered,
            remaining_pellets=sum(
                cell == 1 for row in pellets for cell in row
            ),
            remaining_super_pellets=sum(
                cell == 2 for row in pellets for cell in row
            ),
        ),
        ghosts=ghosts,
    )


def generate_episode(episode_id: int) -> list[TrainingSample]:
    """Generate five consecutive snapshots from one maze and initial state."""

    while True:
        sample = get_randoms()
        episode = []
        try:
            for episode_step in range(EPISODE_LENGTH):
                sample = advance_sample(sample, episode_id, episode_step)
                episode.append(sample)
        except ValueError:
            continue
        return episode


def collect(
    num_samples: int,
    cnn_path: str | Path = DEFAULT_CNN_PATH,
) -> None:
    if num_samples < EPISODE_LENGTH or num_samples % EPISODE_LENGTH:
        raise ValueError(
            f"num_samples must be a positive multiple of {EPISODE_LENGTH}"
        )
    destination = Path(cnn_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cnn_w: TextIO | None = None
    cnn_sw: StreamWriter | None = None

    try:
        # A collection run creates one self-contained dataset with exactly
        # num_samples records instead of silently appending old records.
        cnn_w = destination.open("w")
        cnn_sw = StreamWriter(cnn_w)

        cnn_lines = 0
        start_time = time.time()

        episode_id = 0
        while cnn_lines < num_samples:
            episode = generate_episode(episode_id)
            episode_id += 1
            for sample in episode:
                if cnn_lines >= num_samples:
                    break
                cnn_sw.write_line(CNNFormatter.format_line(sample))
                cnn_lines += 1

            if cnn_lines % 1000 == 0:
                elapsed = time.time() - start_time
                rate = cnn_lines / elapsed if elapsed > 0 else 0
                print(
                    f"[{cnn_lines}/{num_samples}] "
                    f"{rate:.0f} samples/s | "
                    f"CNN: {cnn_lines} lines"
                )

    finally:
        if cnn_w is not None:
            cnn_w.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect sequential CNN ghost-training episodes."
    )
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--output", type=Path, default=DEFAULT_CNN_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    collect(
        num_samples=args.samples,
        cnn_path=args.output,
    )
    dataset = CNNJSONLDataset(args.output)
    print(f"Dataset validated: {len(dataset)} samples")


if __name__ == "__main__":
    main()
