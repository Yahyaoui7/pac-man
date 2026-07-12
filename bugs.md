Based on my thorough analysis of the codebase against the subject.txt requirements, here's a comprehensive breakdown of tasks for Jira:

---

## BUGS

### Critical
1. **Makefile `debug` rule is broken** — `uv run python -m pdb -m pac_man.py` should be `uv run python -m pdb pac_man.py`
2. **WASD keys not implemented** — Subject requires "arrow keys or WASD" but only arrow keys work (`src/logic/inputmanager.py:36-39`)
3. **NameInputState rejects spaces** — Subject says "alphanumeric and spaces only" but `event.unicode.isalnum()` blocks spaces (`src/graphics/renderer.py:880`)
4. **Player.maze attribute never set** — `is_valid_spawn()` references `self.maze` but Player never receives it (`src/graphics/entity_manager.py:399`)
5. **EntityManager creates duplicate SoundManager** — Creates its own `SoundManager()` instead of using the one from `GameStarter` (`src/graphics/entity_manager.py:41`)
6. **Ghost respawn timer not implemented** — Subject says "respawn to their corner after a while when eaten" but eaten ghosts immediately path back with no delay

### Minor
7. **Filename typo** — `soud_manager.py` should be `sound_manager.py`
8. **Typo in level_manager.py:51** — "faile to load" should be "failed to load"
9. **HomeState mutates config directly** — Lines 144-145 modify `curr_level.height/width` in place, corrupting the config for subsequent games
10. **`sys.setrecursionlimit(99999999)`** — Dangerous; should not be needed if BFS is used correctly

---

## MISSING FEATURES (Mandatory from Subject)

11. **Only 3 levels in config** — Subject requires "at least 10 levels"
12. **README.md is incomplete** — Missing all required sections: Description, Instructions, Resources, Configuration, Highscore, Maze Generation, Implementation, Software Architecture, Project Management. Current README is just an architecture diagram and sound table.
13. **No `--ignore-missing-imports` in lint target** — The Makefile `lint` rule is missing this flag that's in the subject spec (line 129 of subject.txt shows it should be included)

---

## CODE QUALITY (mypy + flake8)

### mypy Errors (60+ errors)
14. **Missing type annotations in `score.py`** — All functions lack type hints (`ScoreManager.__init__`, `add_time_bonus`, `HighScoreManager.__init__`, `load_scores`, `save_scores`, `get_top_scores`)
15. **Missing type annotations in `movement.py`** — 12+ functions lack type hints (`__init__`, `set_direction`, `is_centered`, `update_cell_position`, `can_move`, `update_entity`, `update_ghost_to_target`, `update_bfs_ghost`, `get_zone`, `get_zone_bounds`, `is_valid_cell`, `choose_runaway_target_by_zone`, `update_runaway_ghost`)
16. **Missing type annotations in `soud_manager.py`** — All functions lack return types
17. **Missing type annotations in `graphic_lib.py`** — `Animation.__init__`, `reset`, `_duration_for`, `SpriteLibrary.__init__`, `load_ghosts`
18. **Missing type annotations in `button.py`** — All 3 methods lack type hints
19. **`Optional` not used correctly in `graphic_lib.py:293,295`** — `color: GhostColor = None` should be `color: Optional[GhostColor] = None`
20. **`union-attr` errors in `entity_manager.py`** — `Player | None` accessed without None checks (lines 101, 123, 132-135, 137, 209)

### flake8 Violations (25+ E501)
21. **Line length violations** — Multiple files exceed 79 chars: `entity_manager.py` (10 violations), `graphic_lib.py` (7), `soud_manager.py` (5), `parsing.py` (1), `movement.py` (1), `menu.py` (1)

---

## RECOMMENDATIONS / ENHANCEMENTS

22. **Remove dead/commented code** — `movement.py:274-304` (commented-out corner logic), `renderer.py:127,256` (commented-out sound calls), `soud_manager.py:125-130` (commented-out update method)
23. **Add docstrings** — Many classes and public methods lack PEP 257 docstrings (required by subject III.3)
24. **Magic numbers need constants** — Fright duration `7.0` (entity_manager.py:116), invincibility timer `1500` (renderer.py:455), ghost speed `1` (entity_manager.py:449), player speed `3` (entity_manager.py:279)
25. **.gitignore is incomplete** — Missing `.mypy_cache/`, `.venv/`, `*.egg-info/`, `.highscores.json`, `.DS_Store`
26. **Sound file error handling** — `SoundManager.__init__` will crash if any sound file is missing (no try/except around `pygame.mixer.Sound()` calls)
27. **Level time behavior undefined** — When timer runs out and lives > 0, the level resets ("TRY AGAIN") but this isn't documented; subject says "you can decide what happens"
28. **Add project management docs** — Subject Chapter VIII requires timeline, progress tracking, risk analysis, team organization, acceptance test plan in a dedicated subdirectory

---

## SUMMARY BY PRIORITY

| Priority | Count | Category |
|----------|-------|----------|
| P0 - Must Fix | 6 | Bugs (WASD, debug rule, name spaces, Player.maze, ghost respawn, config mutation) |
| P1 - Must Have | 3 | Missing features (10+ levels, README, lint flag) |
| P2 - Should Fix | 8 | mypy errors (type annotations across all files) |
| P3 - Should Clean | 6 | flake8, dead code, docstrings, magic numbers, gitignore, error handling |
| P4 - Nice to Have | 3 | Sound error handling, time behavior docs, project management docs |