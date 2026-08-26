# Project Map

## Project Architecture

`UI -> Hooks -> Services -> Repositories -> Database`

This project implements a Pac-Man game with Reinforcement Learning (PPO) and Supervised Learning (CNN) agents.

> **RL documentation:** detailed, up-to-date docs for the RL system live in
> [`AI_arena/docs/`](AI_arena/docs/README.md) (training loop, environment,
> rewards, model/observation, evaluation & telemetry).

---

## File Index

### `src/logic/movement.py`
Purpose: Movement, wall checks, and BFS pathfinding for player & ghosts.
Key exports:
- `MovementSystem.__init__` -> initialize system and precompute static maze distance cache
- `MovementSystem.clear_cache` -> clear precomputed distance cache
- `MovementSystem.can_move` -> check if move in direction is valid
- `MovementSystem.bfs_distances` -> return shortest distance array from source cell (O(1) cached)
- `MovementSystem.bfs_distances_uncached` -> return distance array via direct BFS traversal
- `MovementSystem.bfs_path` -> return shortest path list from start to target (O(path_len))
- `MovementSystem.update_bfs_ghost` -> advance hunting ghost toward target
- `MovementSystem.update_runaway_ghost` -> advance edible ghost away from player

### `src/logic/level_manager.py`
Purpose: Level configuration and procedural maze generation.
Key exports:
- `LevelManager.build_maze` -> generate procedural maze layout with specified dimensions

### `src/logic/config.py`
Purpose: Constant definitions for wall bitmasks, cell dimensions, and colors.
Key exports:
- `CELL_SIZE` -> grid cell size in pixels
- `NORTH, EAST, SOUTH, WEST` -> bitmasks for wall detection

### `AI_arena/player/player_env.py`
Purpose: Headless Gymnasium-style environment for Pac-Man RL training.
Key exports:
- `PacmanPlayerEnv.__init__` -> initialize environment state and delegates
- `PacmanPlayerEnv.reset` -> reset environment, generate maze, init distances
- `PacmanPlayerEnv.step` -> execute step, update entities, compute rewards

### `AI_arena/player/rewards.py`
Purpose: Stage-dependent reward calculation engine for PPO policy.
Key exports:
- `RewardCalculator.calculate` -> evaluate step events and return total scalar reward

### `AI_arena/player/ghost_controller.py`
Purpose: Ghost behavior management including respawn, frightened, and hunting states.
Key exports:
- `GhostController.update` -> update ghost states and directions per tick

### `AI_arena/player/data/observation.py`
Purpose: Formats spatial grid and extra feature vectors for Pac-Man agent.
Key exports:
- `format_player_observation` -> return grid, extra features, and action mask tensors

### `AI_arena/data/formatter.py`
Purpose: Centralized observation formatter for CNN models.
Key exports:
- `ObservationFormatter.format_observation` -> create unified CNN tensors
