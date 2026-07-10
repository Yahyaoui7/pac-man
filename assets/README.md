# assets/ -- Game Assets

This directory contains all visual assets (sprite sheets) used by the game.

## Directory Structure

```
assets/
  pacman_sprites/
    normal/       -- Pac-Man walking animation (3 frames)
    puncher/      -- Punch attack animation (10 frames: 3 walk + 7 attack)
    kicker/       -- Kick attack animation (6 frames: 3 walk + 3 attack)
  ghost_sprites/
    meta.json     -- Maps color+direction to frame filenames
    *.png         -- Ghost sprite frames
```

---

## pacman_sprites/ -- Pac-Man Sprites

Each mode has its own folder with a `meta.json` listing frame filenames and source dimensions.

### Frame Layout (per mode)

The first 3 frames (`frame_00.png` to `frame_02.png`) are the **walking animation** shared across all modes. Frames after that are the **attack animation** specific to each mode.

| Mode | Folder | Total Frames | Walk | Attack | Scale |
|------|--------|-------------|------|--------|-------|
| Normal | `normal/` | 3 | 3 | 0 | 0.9x |
| Punch | `puncher/` | 10 | 3 | 7 | 1.8x (2x) |
| Kick | `kicker/` | 6 | 3 | 3 | 1.8x (2x) |

### meta.json Format

```json
{
  "source": "puncher-removebg-preview.png",
  "frame_size": [134, 139],
  "frames": ["frame_00.png", "frame_01.png", ...]
}
```

- `source` -- original sprite sheet (for reference)
- `frame_size` -- pixel dimensions of a single frame
- `frames` -- ordered list of frame filenames

### How Frames Are Used

1. **Normal mode**: All 3 frames loop as the walking animation (160ms/frame)
2. **Powered walk** (after eating punch/kick super gum): First 3 frames loop for the 7-second fright duration
3. **Attack** (on ghost collision): Remaining frames play once (punch: 70ms/frame, kick: 80ms/frame) with impact-frame pauses, then return to walk loop

---

## ghost_sprites/ -- Ghost Sprites

Single flat folder with all ghost frames and a `meta.json` index.

### State Sprites

| State | Files | Count | Usage |
|-------|-------|-------|-------|
| Hunt | `{color}_{dir}-{i}.png` | 16 (4 colors x 2 dirs x 2 frames) | Ghost chasing player |
| Frightened | `running_{i}.png` | 3 | Blue ghost fleeing (edible) |
| Eaten | `eye_{dir}-{i}.png` | 6 (up/down/side x 2 frames) | Eyes returning to spawn |

### meta.json Structure

```json
{
  "colors": {
    "red":    { "l": [...], "r": [...] },
    "pinc":   { "l": [...], "r": [...] },
    "cyan":   { "l": [...], "r": [...] },
    "orange": { "l": [...], "r": [...] }
  },
  "frightened": ["running_0.png", "running_1.png", "running_2.png"],
  "eaten": {
    "up":   ["eye_up-0.png", "eye_up-1.png"],
    "down": ["eye_down-0.png", "eye_down-1.png"],
    "side": ["eye_side-0.png", "eye_side-1.png"]
  }
}
```

Side-facing eaten sprites are mirrored horizontally for `LEFT` direction at load time.
