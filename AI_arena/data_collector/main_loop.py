import random
import time
from pprint import pprint
from typing import Literal, TextIO

from AI_arena.data_collector.formatters import (
    CNN_CHANNELS,
    CNNFormatter,
    MLPFormatter,
    StreamWriter,
)
from AI_arena.data_collector.models import (
    LOCAL_PELLET_RADIUS,
    GhostState,
    TrainingSample,
    WorldState,
)
from mazegenerator import MazeGenerator  # type: ignore
from src.logic.movement import MovementSystem  # type: ignore

GHOST_NAMES = ["Blinky", "Pinky", "Inky", "Clyde"]
# Include powered states in the synthetic dataset so the models can learn both
# chase and frightened behavior instead of seeing constant zero mode features.
POWERED_SAMPLE_PROBABILITY = 0.2
MAX_FRIGHTENED_TIMER = 10

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
    width = random.randint(10, 25)
    height = random.randint(10, 50)
    print(f"\nwidth: {width}, height: {height}")
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


def collect(
    num_samples: int,
    mlp_path: str = "MLP_DATA.jsonl",
    cnn_path: str = "CNN_DATA.jsonl",
    debug_first: int = 2,  # Only print first N samples for verification
    single_ghost: bool = True,
) -> None:
    mlp_w: TextIO | None = None
    cnn_w: TextIO | None = None
    mlp_sw: StreamWriter | None = None
    cnn_sw: StreamWriter | None = None

    try:
        mlp_w = open(mlp_path, "w")
        mlp_sw = StreamWriter(mlp_w)

        cnn_w = open(cnn_path, "w")
        cnn_sw = StreamWriter(cnn_w)

        mlp_lines = 0
        cnn_lines = 0
        start_time = time.time()

        for i in range(num_samples):

            sample = get_randoms()

            ghosts_to_write = (
                [0] if single_ghost else list(range(len(sample.ghosts)))
            )

            for g_idx in ghosts_to_write:
                mlp_record = MLPFormatter.format_line(
                    sample, g_idx, single_ghost=single_ghost
                )
                mlp_sw.write_line(mlp_record)
                mlp_lines += 1

            # CNN: one record for all 4 ghosts
            cnn_record = CNNFormatter.format_line(sample)

            # DEBUG: print only first N samples, then never again
            if i < debug_first:
                print(f"\n=== CNN Sample {i + 1} ===")
                for channel_name, channel in zip(
                    CNN_CHANNELS, cnn_record["grid"]
                ):
                    print(f"\n--- Channel: {channel_name} ---")
                    pprint(channel, width=120)

                print("\n--- Non-spatial CNN data ---")
                pprint(
                    {
                        key: value
                        for key, value in cnn_record.items()
                        if key != "grid"
                    },
                    sort_dicts=False,
                    width=120,
                )

            cnn_sw.write_line(cnn_record)
            cnn_lines += 1

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"[{i + 1}/{num_samples}] "
                    f"{rate:.0f} samples/s | "
                    f"MLP: {mlp_lines} lines | "
                    f"CNN: {cnn_lines} lines"
                )

    finally:
        if mlp_w is not None:
            mlp_w.close()
        if cnn_w is not None:
            cnn_w.close()

    # elapsed = time.time() - start_time
    # mode = "single-ghost" if single_ghost else "all-ghosts"
    # feature_count = len(MLPFormatter.feature_names(single_ghost))
    # print(f"\nDone in {elapsed:.1f}s ({mode}, {feature_count} features)")
    # print(f"  MLP: {mlp_lines} lines -> {mlp_path}")
    # print(f"  CNN: {cnn_lines} lines -> {cnn_path}")


def main():

    collect(
        num_samples=1,
        mlp_path="AI_arena/data/MLP_DATA.jsonl",
        cnn_path="AI_arena/data/CNN_DATA.jsonl",
        single_ghost=True,
    )


if __name__ == "__main__":
    main()
