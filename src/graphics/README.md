# src/graphics/ -- Rendering, Sprites, Entities

This package handles everything the player sees: sprite loading, entity lifecycle, pellet grids, ghost AI visuals, and all game screen states.

## Files

| File | Role |
|------|------|
| `graphic_lib.py` | `SpriteLibrary` singleton + `Animation` playback + enums (`PacmanMode`, `Facing`, `GhostColor`, `GhostState`) |
| `entity_manager.py` | `EntityManager` (pellet grid, ghost/player spawn, update loop), `Player`, `Ghost`, `Entity` base class |
| `renderer.py` | State pattern: `StateManager` + every game screen (`HomeState`, `PlayingState`, `PauseState`, `GameOverState`, `VictoryState`, `NameInputState`, `HighScoreState`) |

---

## graphic_lib.py -- Sprite Loading & Animation

### Enums

- **`PacmanMode`**: `NORMAL` (3 walking frames, looping), `PUNCH` (10 frames, 3 walk + 7 attack), `KICK` (6 frames, 3 walk + 3 attack)
- **`Facing`**: `LEFT` / `RIGHT` -- horizontal flip only, never vertical
- **`GhostColor`**: `RED` (Blinky), `PINK` (Pinky), `CYAN` (Inky), `ORANGE` (Clyde)
- **`GhostState`**: `HUNT` (chasing), `FRIGHTENED` (blue, edible), `EATEN` (eyes only, returning home)

### SpriteLibrary

Singleton that loads and caches all sprite frames once at startup.

**Pacman sprites** (`assets/pacman_sprites/`):
- Each mode has its own folder (`normal/`, `puncher/`, `kicker/`) with `meta.json` listing frame filenames
- Frames are split into **walk** (first 3) and **attack** (remaining) subsets at load time
- `SPRITE_SCALE = 0.9` for normal mode, `SPRITE_SCALE_ATTACK = 1.8` (2x) for punch/kick
- Factory methods: `new_animation(mode)`, `new_walk_animation(mode)`, `new_attack_animation(mode)`

**Ghost sprites** (`assets/ghost_sprites/`):
- `meta.json` maps color + direction to frame files
- Hunt: 4 colors x 2 directions x 2 frames
- Frightened: 3 blue running frames
- Eaten: 2 frames for up/down/side (side mirrored for LEFT)
- `GHOST_SPRITE_SCALE = 1.4`

### Animation

`Animation` is a lightweight per-entity playback state referencing a shared `Surface` list:
- `frame_duration_ms` -- time per frame
- `loop` -- whether to cycle or play once
- `overrides` -- per-frame duration overrides (impact frames held longer)
- `update(dt_ms)` advances the timer; `current_frame` returns the current `Surface`

---

## entity_manager.py -- Entities & Pellet Grid

### EntityManager

Owns the game board and all entities.

**Pellet grid** (`pellets[][]`):
- `0` = empty, `1` = normal pellet, `2` = super gum (power pellet)
- 4 corners always get super gums; first 2 corners get guaranteed special abilities (1 punch + 1 kick), other 2 are normal
- Center cell and fully-walled cells get `0`

**Super gum abilities**:
- `ABILITY_NONE` -- normal super gum, just frightens ghosts
- `ABILITY_PUNCH` -- grants punch attack for the fright duration (7s)
- `ABILITY_KICK` -- grants kick attack for the fright duration (7s)

**Update loop** (`update(maze, dt)`):
1. Check if player stepped on a pellet (eat it, add score, play sound)
2. If super gum: frighten all ghosts for 7s, activate ability if special
3. Tick powered-mode timer, end powered mode when expired
4. Update player animation, ghost fright timers, ghost animations

**Drawing** (`draw(screen)`):
1. Draw all pellets (normal as small circles, super gums as pulsing circles; punch = red, kick = blue, normal = peach)
2. Draw player
3. Draw ghosts

### Entity (base class)

Common fields for Player and Ghost:
- `x`, `y` -- pixel position (center of cell)
- `grid_x`, `grid_y` -- current grid cell
- `speed` -- pixels per frame
- `direction`, `next_direction` -- current and queued movement
- `facing` -- `LEFT` or `RIGHT` (horizontal only)

### Player

Extends `Entity` with:
- `mode` -- current `PacmanMode`
- `powered_mode` -- active ability mode (PUNCH/KICK) or `None`
- `powered_timer` -- countdown synced with ghost fright duration
- `is_attacking` -- whether attack animation is playing
- `animation` -- current `Animation` state

**Ability lifecycle**:
1. `start_powered_mode(mode, duration)` -- enters walk loop (first 3 frames)
2. `trigger_attack()` -- on ghost collision, plays attack frames (non-looping)
3. Attack finishes -> returns to walk loop
4. `end_powered_mode()` -- reverts to NORMAL after timer expires

### Ghost

Extends `Entity` with:
- `is_edible` -- True when frightened (blue)
- `is_eaten` -- True when caught during fright
- `frightened_timer` -- countdown (starts at 7.0s)
- `runaway_target` -- random safe zone for frightened movement

**State transitions**: `HUNT` -> `FRIGHTENED` -> `EATEN` -> back to `HUNT`

---

## renderer.py -- Game States (State Pattern)

`StateManager` maintains a stack of `State` objects. Only the top state receives `update()` and `draw()`.

### States

| State | Screen | Key behavior |
|-------|--------|-------------|
| `HomeState` | Main menu | Start, Instructions, Highscores, Exit buttons |
| `PlayingState` | Active gameplay | Movement, collision, pellet eating, ghost AI, HUD |
| `PauseState` | Pause overlay | Resume, Home Menu buttons |
| `GameOverState` | Game over | Score display, name entry, highscores |
| `VictoryState` | Level complete | Final score, save score, home |
| `NameInputState` | Name entry | Typing field for highscore |
| `HighScoreState` | Top 10 scores | Score list, home button |

### PlayingState flow

1. `enter()` -- loads level maze, creates entities, resizes window
2. `update()` -- processes input -> moves player -> moves ghosts (BFS chase / runaway / return) -> updates entities -> checks collisions -> updates timer
3. `check_collision()` -- circle-distance check:
   - Ghost edible -> mark eaten, add score, trigger attack animation
   - Ghost not edible -> lose life (unless invincible), reset position
4. `draw()` -- HUD bar -> maze walls -> entities -> message overlay
