# 🌟 Neon Pac-Man Arcade & Neural AI Engine

A modernized, neon-themed **Pac-Man** arcade game built with **Pygame** and powered by deep neural networks for both Pac-Man and the Ghosts.

This polished branch contains the complete game logic, graphics, audio, and neural network inference engines—isolated cleanly from training environments, reward calculations, and data collectors.

---

## ⚡ Quick Start

### Prerequisites
- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or standard Python virtual environment

### Installation

```bash
# Clone and enter directory
git clone <repo-url>
cd old_pacman

# Switch to the polished gameplay branch
git checkout polished-game

# Install dependencies
uv sync
# or with make:
make install
```

---

## 🕹️ How to Play

### 1. Manual Arcade Mode
Play as Pac-Man using the keyboard:
```bash
uv run python pac_man.py
# or
make run
```

### 2. AI Autopilot Mode (Pac-Man AI)
Watch the trained Actor-Critic neural network with lookahead tactical search play autonomously:
```bash
uv run python pac_man.py --ai-player
# or
make run-ai
```

### 3. Pure Neural Network Mode (No Lookahead Search)
Run Pac-Man strictly on 1-step neural network policy inferences:
```bash
uv run python pac_man.py --ai-player --no-search
# or
make run-ai-fast
```

### 4. Neural Ghost Mode
Enable the neural network model for the four ghosts:
```bash
uv run python pac_man.py --ai-ghosts
# or
make run-ghost-ai
```
*(Note: If `AI_arena/models/ghost_ai.pt` is not present, the game automatically and seamlessly falls back to smart scripted BFS chasing).*

### 5. Full Autonomous Battle
AI Pac-Man vs AI Ghosts:
```bash
uv run python pac_man.py --ai-player --ai-ghosts
# or
make run-full-ai
```

---

## 🎮 Controls & Shortcuts

| Action | Key / Input |
|---|---|
| **Move Pac-Man** | Arrow Keys (`Up`, `Down`, `Left`, `Right`) or `W`, `A`, `S`, `D` |
| **Toggle AI / Manual** | `P` or `A` *(switch instantly during live play)* |
| **Pause Game** | `Spacebar` |
| **Speed Boost Cheat** | `S` |
| **Invincibility Cheat**| `I` |
| **Freeze Ghosts Cheat**| `G` |
| **Skip Level Cheat**   | `L` |
| **Quit Game**          | `ESC` or Window Close |

---

## 🧠 AI Engine Architecture

### Pac-Man Autopilot
- **Model**: `PlayerActorCritic` CNN with a recurrent GRU memory layer (`AI_arena/models/cnn_player.py`).
- **Observation**: 6-channel spatial tensor encoding walls, pellets, power pellets, player, and ghost locations, combined with a 50-dimensional scalar feature vector (distances to targets, traps, and visit frequencies).
- **Tactical Lookahead Search**: A high-performance beam-search planner simulates multiple moves ahead, preventing suicide corners, dead-end traps, and oscillation.
- **Cell Snapping**: Decisions are queried when Pac-Man centers on a new grid cell, eliminating frame-to-frame direction flicker.

### Ghost Engine
- **Neural Network**: `GhostCNN` (`AI_arena/models/cnn_ghost.py`) predicting simultaneous directional moves for Blinky, Pinky, Inky, and Clyde. Supports dynamic INT8 quantization via TorchScript.
- **Scripted AI Fallback**: If neural weights are absent, ghosts run classic BFS pursuit (chasing Pac-Man or flanking) and flee when Pac-Man consumes a power pellet.
- **Plugging in New Ghost Weights**: Place your trained `ghost_ai.pt` or `ghost_ai_quantized.pt` in `AI_arena/models/` and run with `--ai-ghosts`.

---

## 📂 Project Structure

```
├── pac_man.py                     # Polished arcade game launcher & CLI
├── config.json                    # Level configurations & dimensions
├── Makefile                       # Quick commands for game execution
├── pyproject.toml                 # Dependencies and build settings
├── assets/                        # Visual sprites, neon fonts, and UI assets
├── src/
│   ├── game_loop.py               # Main Pygame game starter and loop
│   ├── graphics/                  # Rendering, states (Home, Playing, GameOver), entities
│   ├── logic/                     # Movement mechanics, BFS pathfinding, collision detection
│   └── sounds/                    # Arcade audio effects and music manager
└── AI_arena/
    ├── models/
    │   ├── player_rl_best.pt      # Trained neural network weights for Pac-Man
    │   ├── cnn_player.py          # Pac-Man Actor-Critic neural model
    │   ├── cnn_ghost.py           # Ghost multi-agent neural model
    │   └── cnn_backbone.py        # Shared visual convolutional backbone
    ├── player/
    │   ├── player_controller.py   # Pac-Man AI inference controller
    │   ├── search_planner.py      # Lookahead tactical search engine
    │   └── data/observation.py    # Live observation tensor constructor
    ├── ghosts/
    │   └── ghost_controller.py    # Ghost AI inference controller
    └── data/
        ├── formatter.py           # Spatial grid tensor generator
        └── constants.py           # Input dimensions & channel counts
```
