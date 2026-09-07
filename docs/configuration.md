# Configuration Specification & Fault Tolerance

The game is configured via a flexible JSON configuration file supplied as an argument upon launch:

```bash
python3 pac-man.py config.json
```

---

## 1. Supported Keys & Defaults

| Key | Type | Default | Description |
|---|---|---|---|
| `lives` | Integer | `3` | Number of starting lives for Pac-Man |
| `points_per_pacgum` | Integer | `10` | Score granted for consuming a regular pellet |
| `points_per_super_pacgum` | Integer | `50` | Score granted for consuming a power pellet |
| `points_per_ghost` | Integer | `200` | Score granted for consuming an edible ghost |
| `highscore_filename` | String | `"highscores.json"` | Relative path to persistent highscore storage |
| `levels` | Array | `[...]` | List of level configuration objects (minimum 10) |
| `levels[].width` | Integer | `21` | Grid width of the generated maze (clamped to 10–43) |
| `levels[].height` | Integer | `21` | Grid height of the generated maze (clamped to 10–23) |
| `levels[].seed` | Integer | `42` | Seed for deterministic procedural maze generation |
| `levels[].level_max_time` | Integer | `90` | Countdown timer (in seconds) to complete the level |

---

## 2. Comment Handling

In accordance with subject requirements, the configuration loader strips comments before decoding JSON:
- Lines starting with `#` are treated as comments and discarded.
- In-line comments and whitespace are handled gracefully without causing JSON decode exceptions.

Example configuration snippet with comments:
```json
# Neon Pac-Man Configuration
{
    "lives": 3,
    "points_per_pacgum": 10,
    "points_per_super_pacgum": 50,
    "points_per_ghost": 200,
    "highscore_filename": "highscores.json",
    # Level definitions:
    "levels": [
        { "width": 15, "height": 12, "seed": 42, "level_max_time": 120 },
        { "width": 17, "height": 14, "seed": 84, "level_max_time": 140 }
    ]
}
```

---

## 3. Faulty Configuration Recovery (No Tracebacks)

The `Parser` class (`src/logic/parsing.py`) strictly adheres to zero-crash principles:
1. **Missing File**: If the provided path does not exist, an informative error message is logged to `sys.stderr` and fallback defaults (`DEFAULT_CONFIG`) are applied.
2. **Malformed JSON / Syntax Errors**: If the file contains invalid JSON syntax, line number and error descriptions are logged, and the game safely falls back to default values.
3. **Invalid Data Types or Missing Keys**:
   - Non-integer scores/lives are clamped to positive defaults.
   - Missing levels default to a standard 21x21 layout with seed 42.
   - Unknown keys are ignored without penalty.
4. **No Tracebacks**: Under no circumstances will a malformed configuration terminate the application with an unhandled Python exception.
