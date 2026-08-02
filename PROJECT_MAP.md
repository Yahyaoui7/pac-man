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

File: AI_arena/data/constants.py
Purpose: Shared observation dimensions, channels, and game constants.
Key exports:
- CNN_CHANNEL_COUNT -> Spatial grid channel count
- EXTRA_FEATURE_COUNT -> Extra feature vector length

File: AI_arena/data/dataset.py
Purpose: PyTorch dataset and JSONL data loader utilities.
Key exports:
- CNNJSONLDataset -> Dataset parser for JSONL records
- create_cnn_dataloader -> Creates PyTorch DataLoader

File: AI_arena/data/formatter.py
Purpose: Unified observation formatter for spatial grid and features.
Key exports:
- ObservationFormatter -> Formats game states into unified tensors

File: AI_arena/models/cnn_backbone.py
Purpose: Shared spatial CNN encoder and feature fusion trunk.
Key exports:
- PacmanCNNBackbone -> Unified spatial feature backbone

File: AI_arena/models/cnn_ghost.py
Purpose: Ghost CNN neural network architecture.
Key exports:
- GhostCNN -> Action logits predictor for 4 ghosts

File: AI_arena/models/cnn_player.py
Purpose: Actor-Critic neural network architecture for Pac-Man.
Key exports:
- PlayerActorCritic -> PPO actor-critic model for player

File: AI_arena/player/player_env.py
Purpose: Headless RL environment for Pac-Man player training.
Key exports:
- PacmanPlayerEnv -> Headless environment for player RL

File: AI_arena/player/player_controller.py
Purpose: Live inference controller for Pac-Man player model.
Key exports:
- CNNPlayerController -> Runs player model predictions

File: AI_arena/player/player_training.py
Purpose: PPO training script for Pac-Man player model.
Key exports:
- train_player_ppo -> PPO training loop for player RL

File: AI_arena/player/play_player_ai.py
Purpose: Visual evaluation launcher for live Pac-Man AI player.
Key exports:
- main -> Launches Pygame UI with active AI player

File: AI_arena/player/plot_training_curves.py
Purpose: Script to generate training curve plots and auto-numbered markdown reports.
Key exports:
- plot_all -> Generates reward, pellet completion, and loss graphs
- write_readme -> Auto-generates report README with experiment stats

File: AI_arena/ghosts/ghost_env.py
Purpose: Headless RL environment template for ghost training.
Key exports:
- PacmanGhostEnv -> Headless environment for ghost RL

File: AI_arena/ghosts/ghost_controller.py
Purpose: Live inference controller for Ghost CNN model.
Key exports:
- CNNGhostController -> Runs ghost model predictions

File: AI_arena/ghosts/ghost_training.py
Purpose: Training pipeline for Ghost CNN models.
Key exports:
- train -> Training function for ghost CNN

