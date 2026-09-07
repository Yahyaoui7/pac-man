# Maze Generation & Package Integration

In compliance with Chapter V.4 of the subject, this project does **not** implement a custom procedural generator. Instead, it integrates the external `A-Maze-ing` package assigned from another group as-is.

---

## 1. External Package Details

- **Package**: `mazegenerator` (provided as `mazegenerator-2.0.2-py3-none-any.whl`).
- **Integration Layer**: `LevelManager` in `src/logic/level_manager.py`.

The external library is imported directly without internal modification:
```python
from mazegenerator import MazeGenerator
```

---

## 2. Interface Adaptation & Parameters

The `LevelManager.build_maze` adapter invokes the generator using the exact parameters required by the Pac-Man specification:

```python
maze = MazeGenerator(
    size=(width, height),
    perfect=False,        # MUST be False to create loops and corridors suitable for Pac-Man
    entry_cell=(0, 0),
    exit_cell=(-1, -1),
    seed=seed,            # Deterministic for level 1 (e.g. 42), varied for subsequent levels
)
```

### Key Parameter Rationale:
- **`perfect=False`**: A "perfect" maze has exactly one path between any two points (a tree structure with dead ends). Pac-Man requires interconnected loops so the player and ghosts can navigate circular escape routes without hitting unavoidable dead-ends.
- **`size=(width, height)`**: Clamped by `LevelManager.clamp_dimensions` to ensure the labyrinth fits within graphical viewport constraints (`10 <= width <= 43`, `10 <= height <= 23`).
- **`seed`**: Level 1 uses a fixed seed (`seed=42`) for reproducibility during evaluations; subsequent levels use increasing seeds or random seeds.

---

## 3. Error Handling & Fallbacks

If the external maze generator encounters an internal error or fails to generate a maze:
1. The error is caught gracefully in a `try...except` block.
2. A clear error message is logged to the console.
3. The game falls back to a safe, pre-verified default maze layout so gameplay remains uninterrupted.
