# Project Map

## Core Logic (`src/logic`)

File: src/logic/config.py
Purpose: Game configuration constants and settings.
Key exports:
- CELL_SIZE -> Default cell size in pixels
- GameConfig -> Game settings dataclass

File: src/logic/movement.py
Purpose: Movement and pathfinding logic for player and ghosts.
Key exports:
- MovementSystem -> Entity movement and BFS pathfinding controller

File: src/logic/level_manager.py
Purpose: Maze building and level progression logic.
Key exports:
- LevelManager -> Generates and manages maze layout

File: src/logic/inputmanager.py
Purpose: Keyboard input handling and direction mapping.
Key exports:
- InputManager -> Captures player keyboard inputs

## Graphics & Entities (`src/graphics`)

File: src/graphics/entitys/player.py
Purpose: Pac-Man player entity state and rendering.
Key exports:
- Player -> Pac-Man player object

File: src/graphics/entitys/ghost.py
Purpose: Ghost entity states and prison behaviors.
Key exports:
- Ghost -> Ghost entity object

File: src/graphics/entitys/entity_manager.py
Purpose: Manages player, ghost, and pellet interactions.
Key exports:
- EntityManager -> Main entity interaction coordinator

## AI Arena (`AI_arena`)

File: AI_arena/cnn_model.py
Purpose: CNN neural network architecture for Ghost AI.
Key exports:
- GhostCNN -> PyTorch spatial CNN model for ghosts

File: AI_arena/cnn_controller.py
Purpose: Real-time inference controller for Ghost CNN.
Key exports:
- CNNGhostController -> Runs ghost CNN model predictions

File: AI_arena/cnn_dataset.py
Purpose: PyTorch dataset and constants for CNN data.
Key exports:
- CNNJSONLDataset -> Dataset parser for JSONL records
- create_cnn_dataloader -> Creates PyTorch DataLoader

File: AI_arena/pacman_ghost_env.py
Purpose: Headless RL environment template for ghost model training.
Key exports:
- PacmanGhostEnv -> Headless environment for ghost RL

File: AI_arena/pacman_player_env.py
Purpose: Headless RL environment for training Pac-Man player against BFS ghosts.
Key exports:
- PacmanPlayerEnv -> Headless environment for player RL

File: AI_arena/player_cnn_model.py
Purpose: Actor-Critic neural network architecture for Pac-Man player model.
Key exports:
- PlayerActorCritic -> PyTorch PPO model for player

File: AI_arena/player_rl_training.py
Purpose: PPO training script for Pac-Man player model.
Key exports:
- train_player_ppo -> Main training loop for player RL

File: AI_arena/player_controller.py
Purpose: Real-time inference controller for trained Pac-Man player RL model.
Key exports:
- CNNPlayerController -> Runs player model predictions
