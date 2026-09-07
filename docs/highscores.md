# Persistent Highscore System

The highscore subsystem provides durable, cross-session leaderboard tracking stored in JSON format on disk.

---

## 1. Requirements & Design Choices

| Requirement | Implementation | Rationale |
|---|---|---|
| **Storage Medium** | JSON file (`highscores.json`) | Human-readable, easily inspectable during peer defense, and portable across platforms. |
| **Top 10 Retention** | `self.max_scores = 10` | Maintains standard arcade leaderboard depth, sorting descending by score. |
| **Player Name Rules** | `[:10]`, alphanumeric + space | Strips invalid characters, enforces 10-character limit, defaults to `"Player"` if empty. |
| **Score Validation** | `max(0, int(score))` | Guarantees non-negative integer representation. |
| **Fault Resilience** | Safe fallback to empty list | Missing files, empty files, or corrupted JSON data are caught without tracebacks. |

---

## 2. File Format

Highscores are serialized as an array of objects:

```json
[
    { "name": "NEON_PRO", "score": 15400 },
    { "name": "PAC_MASTER", "score": 12850 },
    { "name": "ARCADE99", "score": 9300 },
    { "name": "GHOSTBUSTER", "score": 7600 },
    { "name": "RETRO_FAN", "score": 5400 }
]
```

---

## 3. User Experience & Lifecycle

1. **Startup**: `HighScoreManager.load_scores()` reads the configured file (`config.highscore_filename`) on game initialization.
2. **Main Menu**: Players can click **HIGHSCORES** to view the Top 10 leaderboard with ranks, names, and scores.
3. **End of Game (Victory or Game Over)**:
   - The game presents `NameInputState`.
   - The player types their name (supporting letters, digits, spaces, and backspace).
   - Upon pressing `ENTER`, the score is inserted, sorted, saved to disk, and the leaderboard is immediately displayed.
