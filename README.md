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





| Game moment              | Sound name idea                   |
| ------------------------ | --------------------------------- |
| Main menu / start game   | **السلام عليكم بعدا هي لولة**     |
| Victory / welcome screen | **merhba biiiik**                 |
| Pac-Man eats ghost       | **FAHHHHHHHHHHHHHH** or **FAAAH** |
| Ghost touches Pac-Man    | **Ayeh ayeh ayeh**                |
| Big dramatic moment      | **VINE BOOM SOUND**               |
| Funny fail / game over   | **Chicken on tree screaming**     |
| Wrong menu click         | **Ach hada ach kadir nta**        |
