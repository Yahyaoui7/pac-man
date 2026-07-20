import random
from AI_arena.data_collector.models import (
    GhostState,
    Target,
    TrainingSample,
    WorldState,
)
from mazegenerator import MazeGenerator  # type: ignore
from src.logic.movement import MovementSystem  # type: ignore

GHOST_NAMES = ["Blinky", "Pinky", "Inky", "Clyde"]


def generate_pellets(width, height, valid_cells, player, ghosts):
    """
    Generate a random pellet map.

    Values:
        0 -> Empty
        1 -> Normal pellet
        2 -> Super pellet
    """

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

    movement = MovementSystem(maze.maze)

    ghosts = []

    for name, position in zip(GHOST_NAMES, ghost_positions):

        direction, distance = movement.get_bfs_next_move(
            position,
            player,
        )

        ghosts.append(
            GhostState(
                name=name,
                position=position,
                direction=direction,
                mode="CHASE",  # TODO: Implement ghost modes
                distance_to_player=distance,
            )
        )

    sample = TrainingSample(
        metadata={
            "seed": seed,
            "maze_width": width,
            "maze_height": height,
            "tick": 0,  # TODO: Current game tick
        },
        world=WorldState(
            maze=maze.maze,
            pellets=pellets,
            player_position=player,
            player_direction="NONE",  # TODO: Player direction
            player_powered=False,  # TODO: Power pellet state
            remaining_pellets=sum(cell == 1 for row in pellets for cell in row),
            remaining_super_pellets=sum(cell == 2 for row in pellets for cell in row),
        ),
        ghosts=ghosts,
        target=Target(
            Blinky=movement.get_bfs_next_move(
                ghost_positions[0],
                player,
            )[0],
            Pinky=movement.get_bfs_next_move(
                ghost_positions[1],
                player,
            )[0],
            Inky=movement.get_bfs_next_move(
                ghost_positions[2],
                player,
            )[0],
            Clyde=movement.get_bfs_next_move(
                ghost_positions[3],
                player,
            )[0],
        ),
    )

    return sample


def main():
    sample = get_randoms()

    print(sample)

    # Dictionary (good for JSON)
    print(sample.to_dict())


if __name__ == "__main__":
    main()
