# Controls & Evaluator Cheat Suite

To facilitate thorough peer-review and testing, the game features an integrated cheat mode directly operable from the keyboard during active gameplay.

---

## 1. Standard Player Controls

| Key | Action |
|---|---|
| `Arrow Keys` or `W, A, S, D` | Steer Pac-Man (Up, Down, Left, Right) |
| `Spacebar` or `ESC` | Pause / Resume game (opens Pause Menu) |
| `Q` | Quit game application |

---

## 2. Reviewer Cheat Mode

During active gameplay (`PlayingState`), peer reviewers can activate the following hotkeys to test edge cases, ghost behaviors, audio effects, and level progressions:

| Hotkey | Cheat Function | Evaluation Utility |
|---|---|---|
| **`I`** | **Invincibility** | Pac-Man cannot take damage or lose lives from ghost collisions. Perfect for inspecting maze layouts without death interruptions. |
| **`F`** | **Ghost Freeze** | All ghosts freeze in place immediately. Allows testing maze geometry, pellet counts, and special abilities safely. |
| **`B`** | **Speed Boost** | Doubles Pac-Man's movement speed for rapid traversal. |
| **`L`** | **Extra Life** | Adds +1 life to Pac-Man immediately. |
| **`K`** | **Level Skip** | Instantly clears remaining pellets and advances to the next level (ideal for verifying multi-level progression). |
| **`H`** | **Hunter Mode** | Allows Pac-Man to eat ghosts even when they are not in frightened mode. |
| **`Ctrl + A`** | **🤖 AI Autopilot** | Hands over control of Pac-Man to the hybrid Deep Learning + Beam Search AI controller in real time. |

---

## 3. Visual Feedback

- Active cheats are rendered as glowing neon badges in the in-game HUD banner at the bottom of the screen.
- When AI Autopilot is active, a real-time HUD bubble displays the AI's current move choice and directional probability distribution.
