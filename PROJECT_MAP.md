# PROJECT_MAP

Source of truth for repository structure, module responsibilities, and key function exports.

---

## AI_arena/models

### File: AI_arena/models/cnn_backbone.py
Purpose: Spatial CNN encoder and feature fusion trunk.
Key exports:
- PacmanCNNBackbone -> CNN encoder class fusing spatial grid and extra features

### File: AI_arena/models/cnn_player.py
Purpose: Actor-Critic and Supervised Learning model definitions.
Key exports:
- PlayerActorCritic -> PPO policy and value network
- PlayerImitationCNN -> Supervised imitation learning policy classifier
- load_sl_weights_into_ppo -> Load pre-trained SL weights into PPO actor network

### File: AI_arena/models/player_sl_best.pt
Purpose: Best pre-trained Supervised Learning model checkpoint.

---

## AI_arena/player/utils

### File: AI_arena/player/utils/logger.py
Purpose: Training file logger and cross-platform keyboard signal listener.
Key exports:
- TrainingLogger -> Append log messages to file and stdout
- QuitListener -> Listen for non-blocking 'q' keypress to save checkpoint

### File: AI_arena/player/utils/metrics.py
Purpose: Training metric formatting and breakdown stats calculator.
Key exports:
- BD_LABELS -> Dictionary mapping reward component keys to display labels
- format_breakdown_line -> Format log string for episode reward components
- compute_positive_stats -> Average positive rewards over episode window
- compute_negative_stats -> Average negative penalties over episode window

---

## AI_arena/player/plotting

### File: AI_arena/player/plotting/parser.py
Purpose: Training log text parser converting regex matches to numpy arrays.
Key exports:
- parse_log -> Extract metric arrays from training log file

### File: AI_arena/player/plotting/charts.py
Purpose: Matplotlib diagnostic chart generators.
Key exports:
- plot_all -> Render overview and diagnostic charts to PNGs

### File: AI_arena/player/plotting/report.py
Purpose: Automated markdown report generator.
Key exports:
- generate_markdown_report -> Build summary README.md for report folder

---

## AI_arena/player

### File: AI_arena/player/expert.py
Purpose: Risk-aware BFS lookahead expert teacher.
Key exports:
- PacmanExpert -> Search teacher generating oscillation-free action labels

### File: AI_arena/player/player_collector.py
Purpose: Demonstration data collection pipeline.
Key exports:
- collect_demonstrations -> Generate JSONL expert trajectory dataset

### File: AI_arena/player/imitation_dataset.py
Purpose: JSONL dataset loader for supervised training.
Key exports:
- PlayerImitationDataset -> PyTorch Dataset streaming expert samples

### File: AI_arena/player/observation.py
Purpose: State observation feature formatter.
Key exports:
- format_player_observation -> Format grid and 35-dim feature vector

### File: AI_arena/player/player_env.py
Purpose: Headless Pac-Man RL environment wrapper.
Key exports:
- PacmanPlayerEnv -> Gym-like environment for Pac-Man training

### File: AI_arena/player/player_training.py
Purpose: High-level PPO CLI training entrypoint.
Key exports:
- train_player_ppo -> PPO training loop with warm-start support

### File: AI_arena/player/plot_training_curves.py
Purpose: High-level CLI entrypoint for diagnostic reports.
Key exports:
- main -> Execute log parsing, chart generation, and report building

---

## AI_arena/data

### File: AI_arena/data/constants.py
Purpose: Central constants for network dimensions and entity limits.
Key exports:
- CNN_CHANNEL_COUNT -> Number of spatial grid input channels (12)
- EXTRA_FEATURE_COUNT -> Number of dense feature vector dimensions (35)
- ACTION_COUNT -> Number of legal movement actions (4)

### File: AI_arena/data/formatter.py
Purpose: Raw spatial grid and observation tensor formatter.
Key exports:
- ObservationFormatter -> Utility class mapping maze layout to tensors

---

## src/logic

### File: src/logic/level_manager.py
Purpose: Procedural maze generation and layout scaling.
Key exports:
- LevelManager -> Build and clamp maze grid dimensions

### File: src/logic/movement.py
Purpose: Directional collision detection and BFS distance maps.
Key exports:
- MovementSystem -> Grid movement and BFS graph navigation
