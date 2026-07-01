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
import pygame

CELL_SIZE = 30


class GameStarter:
    def __init__(self, config):
        self.running = True
        self.config = config
        self.maze = None
        self.curr_level = config["levels"][5]

    def display(self):

        screen = pygame.display.set_mode(
            (
                self.curr_level["width"] * CELL_SIZE,
                self.curr_level["height"] * CELL_SIZE,
            )
        )
        clock = pygame.time.Clock()
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            screen.fill("purple")

        pygame.display.flip()

        clock.tick(60)

        pygame.quit()

    def run(self):
        self.maze = MazeGenerator(
            size=(self.curr_level["width"], self.curr_level["height"]),
            entry_cell=(0, 0),
            exit_cell=(0, 0),
            perfect=False,
            seed=self.curr_level["seed"],
        )

        pygame.init()
        self.display()
