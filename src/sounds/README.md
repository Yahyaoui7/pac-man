# src/sounds/ -- Audio System

This directory contains the `SoundManager` class and all audio assets (.mp3 files).

## Sound Manager (`soud_manager.py`)

`SoundManager` wraps `pygame.mixer` to handle both sound effects and background music.

### Sound Effects

Loaded as `pygame.mixer.Sound` objects at startup, stored in `self.sounds` dict:

| Key | Event | Volume |
|-----|-------|--------|
| `eat_normal_pellet` | Player eats a regular pellet | 0.3 |
| `eat_super_pacgum` | Player eats a super gum | 0.6 |
| `eat_ghost` | Player eats an edible ghost | 0.7 |
| `player_death` | Ghost catches the player | 0.8 |
| `level_complete` | All pellets eaten | 0.7 |
| `victory` | Game won | 0.7 |
| `game_over` | Lives depleted | 0.8 |
| `menu_select` | Button hover | 0.4 |
| `menu_confirm` | Button click | 0.5 |
| `pause` | Game paused | 0.5 |

### Background Music

Loaded via `pygame.mixer.music` (one track at a time):

| Key | Context | Volume |
|-----|---------|--------|
| `menu_intro` | Main menu entrance | 0.3 |
| `menu_music` | Main menu background | 0.3 |
| `game_intro` | Level start jingle | 0.4 |
| `game_music` | During gameplay | 0.15 |
| `game_over_music` | Game over screen | 0.4 |
| `victory_music` | Victory screen | 0.4 |

### Audio Ducking

`play_sound_with_duck(name)` temporarily lowers music volume (to 0.1) when a sound effect plays, so the effect is audible over the music. Used for pellet-eating and ghost-eating sounds during gameplay.

### API

- `play_sound(name)` -- play a one-shot sound effect
- `play_music(name, loop=True)` -- switch background music
- `stop_music()` / `pause_music()` / `resume_music()` -- music control
- `play_sound_with_duck(name)` -- play effect with music ducking

## Audio Assets

All `.mp3` files in this directory. The collection includes standard Pac-Man sounds (munching, ghost eating) and custom comedic/meme sound effects (Arabic voice clips, anime effects, screams).
