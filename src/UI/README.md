# src/UI/ -- User Interface Widgets

Reusable UI components used by the game screens (menus, overlays, name entry).

## Files

| File | Role |
|------|------|
| `button.py` | `Button` -- clickable rect with text and hover highlight |
| `menu.py` | `TextInput` -- text field for player name entry |

---

## button.py -- Button Widget

A simple rectangular button with hover detection and click handling.

**Constructor**: `Button(x, y, width, height, text, font)`

**Behavior**:
- `update(input_state)` -- checks if mouse is hovering (highlight changes color) and if clicked. Returns `True` on click.
- `draw(screen)` -- renders the filled rect with centered text. Hover state uses a lighter gray.

**Colors**: Normal `(70, 70, 70)`, hover `(120, 120, 120)`, text white.

Used in: `HomeState` (menu buttons), `PauseState` (resume/home), `GameOverState` (name/highscore/home), `VictoryState` (save/home), `HighScoreState` (home).

---

## menu.py -- TextInput Widget

A text input field for capturing the player's name on game over or victory screens.

**Constructor**: `TextInput(x, y, width, height, font)`

**Behavior**:
- `handle_event(event)` -- processes KEYDOWN events:
  - alphanumeric/space characters appended (max 10 chars)
  - BACKSPACE removes last character
  - RETURN submits if name is non-empty (returns `True`)
- `draw(screen)` -- renders the field with:
  - Dark background with colored border (yellow when active, gray when inactive)
  - Text with a flashing cursor (`_` toggles every 500ms)

**Limits**: Max 10 characters, alphanumeric + spaces only, must be non-empty to submit.
