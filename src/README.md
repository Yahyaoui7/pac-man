# src/ -- Source Code

This directory contains the full game code, organized into four packages plus the main game loop.

## Architecture

```
pac_man.py  (entry point)
    |
    v
GameStarter  (src/game_loop.py)
    |
    +-- StateManager       -> HomeState / PlayingState / PauseState / GameOverState ...
    +-- LevelManager       -> generates mazes per level config
    +-- EntityManager      -> pellets, player, ghosts, collision drawing
    +-- SoundManager       -> sound effects + background music
    +-- InputManager       -> keyboard / mouse input
    +-- ScoreManager       -> scoring + highscore persistence
```

## Packages

| Package | Purpose |
|---------|---------|
| `src/graphics/` | Rendering, sprites, entity lifecycle, game states |
| `src/logic/` | Game config, maze generation, movement/AI, scoring, input |
| `src/sounds/` | Audio: sound effects and background music |
| `src/UI/` | Reusable UI widgets (buttons, text input) |

## Game Loop

`GameStarter.run()` drives the main loop at 60 FPS:

1. Collect events via `InputManager`
2. Update the current `State` (movement, collision, timers)
3. Draw the current `State` to screen
4. `clock.tick(60)`

## Key Files

| File | Role |
|------|------|
| `game_loop.py` | `GameStarter` -- owns the window, clock, and all managers |
| `graphics/renderer.py` | State pattern: `HomeState`, `PlayingState`, `PauseState`, `GameOverState`, `VictoryState`, `NameInputState`, `HighScoreState` |
| `graphics/entity_manager.py` | Pellet grid, player/ghost entities, ability system, draw order |
| `graphics/graphic_lib.py` | `SpriteLibrary` singleton -- loads, scales, caches all sprite frames |
| `logic/config.py` | `GameConfig` / `LevelConfig` dataclasses + maze direction constants |
| `logic/parsing.py` | `Parser` -- reads `config.json`, validates, falls back to defaults |
| `logic/level_manager.py` | `LevelManager` -- generates mazes via `mazegenerator`, manages per-level timer |
| `logic/movement.py` | `MovementSystem` -- pixel movement, wall checks, BFS pathfinding for ghosts |
| `logic/inputmanager.py` | `InputManager` -- maps pygame events to an `InputState` dataclass |
| `logic/score.py` | `ScoreManager` (runtime) + `HighScoreManager` (persistent JSON) |
| `logic/utils.py` | Small timer helpers: `now()`, `after(ms)`, `expired(end_time)` |
| `sounds/soud_manager.py` | `SoundManager` -- loads all .mp3 effects + music, ducking support |
| `UI/button.py` | `Button` widget -- clickable rect with hover highlight |
| `UI/menu.py` | `TextInput` widget -- player name entry on game over |
