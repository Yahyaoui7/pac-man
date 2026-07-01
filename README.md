main.py
    │
    ▼
GameStarter
    │
    ├──────────────┐
    │              │
    ▼              ▼
InputManager   LevelManager
                    │
                    ▼
               MazeGenerator
                    │
                    ▼
                  Maze
                    │
                    ▼
              EntityManager
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
     Pacman                 Ghosts
                                │
                                ▼
                         CollisionSystem
                                │
                                ▼
                           Renderer
                                │
                                ▼
                          SpriteLibrary