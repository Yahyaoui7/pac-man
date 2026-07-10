# src/logic/ -- Game Logic & Configuration

This package contains the non-visual game systems: configuration, input handling, maze/level management, movement AI, scoring, and utility helpers.

## Files

| File | Role |
|------|------|
| `config.py` | `GameConfig` / `LevelConfig` dataclasses + maze direction constants |
| `parsing.py` | `Parser` -- reads and validates `config.json` |
| `level_manager.py` | `LevelManager` -- generates mazes, manages per-level timer |
| `movement.py` | `MovementSystem` -- pixel movement, wall checks, BFS ghost AI |
| `inputmanager.py` | `InputManager` -- maps pygame events to `InputState` |
| `score.py` | `ScoreManager` (runtime) + `HighScoreManager` (JSON persistence) |
| `utils.py` | Timer helpers: `now()`, `after(ms)`, `expired(end_time)` |

---

## config.py -- Game Configuration

### Constants

- `TOP_BAR_HEIGHT = 30` -- HUD bar height in pixels
- `CELL_SIZE = 30` -- base size of one maze cell (dynamically recalculated per level)
- `PADDING = 20` -- screen margin around the maze
- Direction bitmasks: `NORTH = 1`, `EAST = 2`, `SOUTH = 4`, `WEST = 8` -- used in the maze cell encoding

### Dataclasses

**`LevelConfig`**: Per-level settings
- `width`, `height` -- maze dimensions (clamped to 10-63 range at load)
- `seed` -- RNG seed for maze generation
- `level_max_time` -- time limit in seconds

**`GameConfig`**: Global game settings
- `lives` -- starting lives (default 3)
- `points_per_pacgum` -- score per normal pellet
- `points_per_super_pacgum` -- score per super gum
- `points_per_ghost` -- score per eaten ghost
- `highscore_filename` -- JSON file for persistent scores
- `levels` -- list of `LevelConfig`

---

## parsing.py -- Config File Parser

`Parser` reads `config.json` (or any JSON config file) and converts it to a `GameConfig`.

- Supports `#` comment lines (stripped before parsing)
- Falls back to `DEFAULT_CONFIG` on any error (file not found, invalid JSON, bad values)
- Validates each field: positive integers for counts/dimensions, strings for filenames
- Nested `levels` array: each entry becomes a `LevelConfig`

---

## level_manager.py -- Maze Generation & Level Timer

`LevelManager` handles maze creation and per-level time tracking.

**Maze generation**: Uses the `mazegenerator` library (`MazeGenerator`) to create procedural mazes from a seed. Dimensions are clamped to `10 <= w <= 33`, `10 <= h <= 63`.

**Level flow**:
1. `load_level(index)` -- generates maze, sets `remaining_time`
2. `update_time(dt)` -- counts down each frame
3. `is_time_out()` -- returns True when time reaches 0

---

## movement.py -- Movement System & Ghost AI

`MovementSystem` controls all entity movement. It reads the maze to check walls and runs BFS pathfinding for ghost AI.

### Entity Movement

`update_entity(entity)`:
1. At cell center: check queued `next_direction`, apply if valid (no wall)
2. If current direction hits a wall, stop
3. Otherwise: advance pixel position by `entity.speed`

Wall checks use the maze bitmask: e.g., `cell & EAST` means there IS a wall to the east.

### Ghost AI (3 modes)

**Hunt** (`update_bfs_ghost`): Ghost not edible -- chase the player
- At each cell center, BFS from ghost to player
- Follow the first step of the shortest path

**Frightened** (`update_runaway_ghost`): Ghost edible -- flee from player
- Divide maze into 4 quadrants (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT)
- Pick a random valid cell in the opposite quadrant from the player
- BFS toward that target; pick a new target when reached

**Eaten** (`update_ghost_to_target`): Ghost was caught -- return to spawn
- BFS from current position to spawn coordinates
- Follow the path back

### Helper Methods

- `can_move(row, col, direction)` -- wall check using bitmask
- `bfs_path(start, target)` -- shortest path via BFS, returns list of cells
- `get_neighbors(row, col)` -- valid adjacent cells
- `is_centered(entity)` -- checks if entity is at exact cell center

---

## inputmanager.py -- Input Handling

`InputManager` maps raw pygame events to a clean `InputState` dataclass each frame.

**InputState fields**:
- `move_up/down/left_right` -- held arrow keys (continuous)
- `pause_pressed` -- ESC key (one-shot)
- `action_pressed` -- SPACE key (one-shot, used for ability activation)
- `mouse_pos`, `mouse_pressed`, `mouse_clicked` -- mouse state for UI buttons
- `quit_requested` -- Q key or window close

---

## score.py -- Scoring & Highscores

### ScoreManager

Runtime score tracking. Methods: `add_normal_pellet()`, `add_super_pacgum()`, `add_ghost()`, `add_time_bonus(time)`. Points come from `GameConfig`.

### HighScoreManager

Persistent top-10 leaderboard stored in `.highscores.json`.
- `add_score(name, score)` -- inserts, sorts descending, keeps top 10, saves
- `get_top_scores()` -- returns the sorted list
- `load_scores()` / `save_scores()` -- JSON file I/O

---

## utils.py -- Timer Helpers

Three small functions built on `pygame.time.get_ticks()`:

- `now()` -- current time in ms
- `after(ms)` -- timestamp when a timer should expire
- `expired(end_time)` -- True if current time >= end_time

Used for invincibility frames (1.5s after death) and other timed effects.
