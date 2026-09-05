# 🌟 Neon Pac-Man Arcade

A modernized, neon-themed **Pac-Man** arcade game built with **Pygame** and enhanced with a trained deep neural network autopilot for Pac-Man.

Ready for publishing on platforms like **Itch.io**: launches with a single command and all controls, cheats, and AI toggles are controlled seamlessly from within the game.

---

## ⚡ Quick Start

### Installation

```bash
# Install dependencies
uv sync
# or:
make install
```

### Run the Game

```bash
make run
# or:
python pac_man.py
```

---

## 🕹️ Controls & Cheats

All game mechanics and special features are controlled directly from within the game:

| Action | Key / Shortcut | Description |
|---|---|---|
| **Move Pac-Man** | `Arrow Keys` or `W, A, S, D` | Classic arcade directional controls |
| **Pause Game** | `Spacebar` or `ESC` | Pause and access pause menu |
| **🤖 AI Autopilot** | **`Ctrl + A`** | **Cheat:** Toggles the trained AI neural autopilot (Hybrid Lookahead Search + Neural Net) on/off |
| **Invincibility** | `I` | **Cheat:** Makes Pac-Man immune to ghost collisions |
| **Freeze Ghosts** | `F` | **Cheat:** Freezes all ghosts in place |
| **Speed Boost** | `B` | **Cheat:** Doubles Pac-Man's movement speed |
| **Extra Life** | `L` | **Cheat:** Grants an additional life |
| **Skip Level** | `K` | **Cheat:** Instantly clears all pellets and advances level |

Active cheats are visually rendered in neon badges at the bottom of the screen.

---

## 🧠 AI Engine Architecture

- **Model**: `PlayerActorCritic` CNN with recurrent GRU memory (`AI_arena/models/cnn_player.py`) using pre-trained weights (`AI_arena/models/player_rl_best.pt`).
- **Tactical Lookahead Search**: Simulated beam search evaluates ghost pursuit vectors, pellet yields, and dead-end traps multiple steps ahead to prevent corner traps.
- **In-Game Toggle**: Activating the `Ctrl + A` cheat seamlessly switches control between human input and AI navigation in real time.
- **Ghost AI**: Prepared for the ghost model integration; currently runs polished, smart scripted BFS pursuit and runaway mechanics.

---

## 📂 Project Structure

```
├── pac_man.py                     # Clean single-command game launcher
├── config.json                    # Level configurations & dimensions
├── Makefile                       # make run / install / clean / lint
├── assets/                        # Visual sprites, neon fonts, and UI assets
├── src/
│   ├── game_loop.py               # Main Pygame game starter and loop
│   ├── graphics/                  # Rendering, states (Home, Instructions, Playing, Pause), entities
│   ├── logic/                     # Movement mechanics, BFS pathfinding, input manager, cheats
│   └── sounds/                    # Arcade audio effects and music manager
└── AI_arena/
    ├── models/
    │   ├── player_rl_best.pt      # Trained neural network weights for Pac-Man
    │   ├── cnn_player.py          # Pac-Man Actor-Critic neural architecture
    │   ├── cnn_ghost.py           # Ghost multi-agent neural architecture (dormant for now)
    │   └── cnn_backbone.py        # Shared visual convolutional backbone
    ├── player/
    │   ├── player_controller.py   # Pac-Man AI inference controller
    │   ├── search_planner.py      # Lookahead tactical search engine
    │   └── data/observation.py    # Live observation tensor constructor
    └── ghosts/
        └── ghost_controller.py    # Ghost AI inference controller
```
