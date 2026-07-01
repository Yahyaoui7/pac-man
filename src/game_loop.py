# main game loop

# GameStarter
# │
# ├── Renderer
# ├── InputManager
# ├── LevelManager
# ├── EntityManager
# ├── AudioManager
# ├── UISystem
# └── CollisionSystem

from mazegenerator import MazeGenerator


class GameStarter:
    def run():
        # Create a simple 20x20 maze
        maze_gen = MazeGenerator(
            size=(20, 20), entry_cell=(0, 0), exit_cell=(0, 0), perfect=False, seed=0
        )

        # Get the maze structure
        maze_grid = maze_gen.maze
        shortest_path = maze_gen.shortest_path
        print(maze_grid, shortest_path)
