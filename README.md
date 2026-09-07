_This project has been created as part of the 42 curriculum by mmenniou, nyahyaou._

# Neon Pac-Man Arcade

A modernized, neon-themed arcade recreation of the classic **Pac-Man** game built in Python with **Pygame**, featuring procedural maze integration, persistent leaderboard tracking, evaluator cheat modes, and a hybrid deep neural network autopilot.

---

## Table of Contents

- [Description](#description)
- [Instructions](#instructions)
- [Configuration](#configuration)
- [Highscore](#highscore)
- [Implementation](#implementation)
- [General Software Architecture](#general-software-architecture)
- [Project Management](#project-management)
- [Resources & AI Usage](#resources--ai-usage)
- [Documentation Index](#documentation-index)

---

## Description

The objective of this project is to recreate Namco's iconic 1980 arcade game **Pac-Man** using modern software engineering practices, clean Object-Oriented Programming (OOP), and an extensible architecture.

### Key Highlights:

- **Neon Visual Aesthetics**: Custom vector-style neon sprites, glowing animations, dynamic status HUD, and sound-ducked arcade audio.
- **Procedural Labyrinths**: Dynamic level generation through integration with an external `A-Maze-ing` package.
- **Persistent Highscores**: Top-10 leaderboard saved to disk with player name input and validation.
- **Evaluator Cheat Suite**: Integrated hotkeys for invincibility, ghost freeze, speed boost, extra lives, level skipping, and hunter mode.
- **Hybrid AI Autopilot**: An optional deep learning player controller running on ONNX Runtime and beam search, toggleable in real time.
- **Standalone Distribution**: Fully self-contained Linux executable packaged for Itch.io deployment (~82 MB).

---

## Instructions

### 1. Prerequisites

- **Python**: Version 3.10 or later.
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `pip`.

### 2. Installation

To install the project dependencies:

```bash
make install
# Or directly via uv:
uv sync
```

### 3. Execution

Run the game using the Makefile or command line:

```bash
make run
# Or via python:
python3 pac-man.py config.json
# (Alternative syntax: python3 pac_man.py config.json)
```

### 4. Code Quality & Linting

To run static analysis and verify compliance with `flake8` and `mypy`:

```bash
make lint
```

For strict checking:

```bash
make lint-strict
```

### 5. Cleaning Artifacts

To clean bytecode caches and temporary files:

```bash
make clean
```

### 6. In-Game Controls & Cheats

| Action              | Key / Shortcut               | Description                               |
| ------------------- | ---------------------------- | ----------------------------------------- |
| **Move Pac-Man**    | `Arrow Keys` or `W, A, S, D` | 4-way corridor navigation                 |
| **Pause Game**      | `Spacebar` or `ESC`          | Opens in-game pause menu                  |
| **Quit Game**       | `Q`                          | Exits the application                     |
| **🤖 AI Autopilot** | **`Ctrl + A`**               | **Cheat**: Toggles neural autopilot       |
| **Invincibility**   | **`I`**                      | **Cheat**: Immune to ghost damage         |
| **Ghost Freeze**    | **`F`**                      | **Cheat**: Freezes all ghosts in place    |
| **Speed Boost**     | **`B`**                      | **Cheat**: Doubles player speed           |
| **Extra Life**      | **`L`**                      | **Cheat**: Adds +1 life                   |
| **Level Skip**      | **`K`**                      | **Cheat**: Clears level immediately       |
| **Hunter Mode**     | **`H`**                      | **Cheat**: Allows eating dangerous ghosts |

---

## Configuration

The game is configured via a JSON file (e.g., `config.json`) supporting `#` comment lines.

### Supported Keys & Defaults:

- **`lives`** (`int`, default: `3`): Initial player lives.
- **`points_per_pacgum`** (`int`, default: `10`): Score per normal dot.
- **`points_per_super_pacgum`** (`int`, default: `50`): Score per power pellet.
- **`points_per_ghost`** (`int`, default: `200`): Score per eaten ghost.
- **`highscore_filename`** (`str`, default: `"highscores.json"`): Storage path for scores.
- **`levels`** (`list`): Array of at least 10 level definitions, each containing:
  - `width` (`int`): Grid width (clamped between 10 and 43).
  - `height` (`int`): Grid height (clamped between 10 and 23).
  - `seed` (`int`): Level generation seed (Level 1 fixed to 42; subsequent levels varied).
  - `level_max_time` (`int`, default: `90`): Level countdown timer in seconds.

### Faulty Config Handling:

Unknown keys are ignored. Missing or invalid keys clamp to safe defaults with clear console warnings—never raising an unhandled Python traceback.

_Detailed documentation_: [docs/configuration.md](docs/configuration.md)

---

## Highscore

The persistent highscore system is implemented via `HighScoreManager` (`src/logic/score.py`):

- **Persistence**: Serialized to a JSON file (`highscores.json`) on disk.
- **Validation**: Player names are sanitized to a maximum of 10 characters (alphanumeric and spaces only). Scores are validated as non-negative integers.
- **Top 10 Depth**: Keeps and sorts the Top 10 scores descending.
- **End-Game Integration**: Upon game completion (Victory or Game Over), the game presents an interactive name input screen before displaying the updated leaderboard.
- **Why JSON?**: A lightweight JSON structure allows inspection during evaluations while ensuring portability across operating systems.

_Detailed documentation_: [docs/highscores.md](docs/highscores.md)

---

## Implementation

The game is built in Python 3.10+ using **Pygame** for rendering, input dispatch, and audio playback:

- **Corridor Snapping & Movement**: Continuous coordinate updates coupled with discrete tile-center turning to ensure smooth cornering without wall clipping.
- **Classic Ghost Personalities**:
  - **Blinky**: Direct pursuit.
  - **Pinky**: Ambush lookahead.
  - **Inky**: Flanking vector triangulation.
  - **Clyde**: Distance-based proximity scatter.
- **Pellet & Power Pellet Mechanics**: Corner placement for super pellets, timed frightened mode for ghosts with flashing warning indicators, and special ability modifiers.
- **Zero-PyTorch Inference**: High-performance ONNX Runtime engine executing a full-precision FP32 model in pure NumPy at rock-solid 60 FPS.

---

## General Software Architecture

The codebase adheres strictly to the **State Pattern** and modular separation:

```
old_pacman/
├── pac-man.py / pac_man.py    # Main game entry points
├── Makefile                   # Automation (install, run, debug, clean, lint)
├── config.json                # Game configuration with 10+ levels
├── src/
│   ├── game_loop.py           # Game lifecycle and display management
│   ├── graphics/              # State machine, HUD, and sprite renderers
│   │   ├── renderer.py        # Base state & screen transitions
│   │   ├── states/            # Home, Playing, Pause, GameOver, Victory, HighScore, NameInput
│   │   └── entitys/           # Player, Ghost, EntityManager, animations
│   ├── logic/                 # Movement, collision, level loading, score
│   │   ├── parsing.py         # Resilient comment-aware JSON parser
│   │   ├── level_manager.py   # Maze generator adapter
│   │   ├── movement.py        # Snapping, BFS shortest paths
│   │   └── score.py           # Highscore serialization
│   └── sounds/                # Audio manager with ducking
├── AI_arena/                  # Standalone pure ONNX AI subsystem
│   ├── models/player_model.onnx # Trained neural policy
│   ├── player/                # ONNX controller & lookahead planner
│   └── data/                  # Pure-NumPy observation formatter
└── docs/                      # Technical specification documents
```

_Detailed documentation_: [docs/architecture.md](docs/architecture.md)

---

## Project Management

A structured project management approach was maintained throughout the development lifecycle:

- **Work Tracking**: Phased delivery tracking core gameplay, maze integration, highscores, AI training, and release packaging.
- **Documentation**: All management records, acceptance test matrices, and risk analyses are maintained in the dedicated directory:
  👉 [**Project Management Directory (docs/)**](docs/)

---

## Resources & AI Usage

### 1. References & Documentation

- _The Pac-Man Dossier_ by Jamey Pittman: In-depth behavioral breakdown of original arcade ghost AI and corridor tile logic.
- _Pygame Community Documentation_: Surface blitting, sound mixer channels, and event queue management.
- _ONNX Runtime Documentation_: Multi-threaded CPU graph optimization and tensor session management.

### 2. Description of AI Usage

In accordance with 42 AI guidelines, AI tools were leveraged deliberately for specific development phases:

- **Reinforcement Learning Architecture Design**: Exploring network topology options (CNN spatial feature extractors combined with recurrent GRU memory for corridor history).
- **Search Optimization**: Designing the heuristic scoring functions for forward beam search lookahead (anti-oscillation penalties, dead-end traps).
- **Static Analysis Remediation**: Accelerating `mypy` strict type-hint coverage across game state transitions and `flake8` compliance.
- **Standalone Packaging**: Identifying and resolving PyInstaller bundle path resolution issues (`sys._MEIPASS`) when decoupling the runtime from PyTorch.

---

## Documentation Index

For in-depth technical specifications, please consult the dedicated documents in `docs/`:

- 🏛️ [General Architecture](docs/architecture.md)
- ⚙️ [Configuration Specification](docs/configuration.md)
- 🌀 [Maze Generator Integration](docs/maze_generation.md)
- 🏆 [Highscore Subsystem](docs/highscores.md)
- 🕹️ [Controls & Cheat Suite](docs/cheats_and_controls.md)
- 📦 [Packaging & Itch.io Deployment](docs/packaging_and_itch.md)
- 🧠 [AI Architecture & Training Branches Guide](docs/ai_and_training_branches.md)
