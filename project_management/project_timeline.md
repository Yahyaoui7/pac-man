# Project Timeline & Progress Tracking

This document shows the actual development timeline of the Pac-Man project, derived from Git commit history and Jira task tracking.

## Project Duration
- **Start:** July 1, 2026
- **End:** September 7, 2026
- **Total Duration:** ~10 weeks

---

## Phase 1: Foundation & Core Engine (July 1–4)
**Commits:** 27 | **Contributors:** Mouad, Nabil

| What was done | Who |
|---|---|
| Project structure setup, Makefile, pyproject.toml | Mouad |
| Config JSON parsing & validation | Nabil (`SCRUM-25`, `SCRUM-26`) |
| A-Maze-ing maze generation integration | Mouad (`SCRUM-27`, `SCRUM-28`, `SCRUM-29`) |
| Game loop, level system, window rendering | Mouad (`SCRUM-30`, `SCRUM-31`) |
| Player class creation, pixel-based movement | Nabil (`SCRUM-34`, `SCRUM-35`) |
| Ghost creation and initial movement | Nabil (`SCRUM-39`, `SCRUM-40`) |
| Sound manager & music integration | Mouad, Nabil |

---

## Phase 2: Gameplay Mechanics (July 5–10)
**Commits:** 34 | **Contributors:** Mouad, Nabil

| What was done | Who |
|---|---|
| Pellet system (drawing, eating, scoring) | Mouad (`SCRUM-87`) |
| Ghost chase & flee (BFS pathfinding) | Nabil (`SCRUM-41`, `SCRUM-42`) |
| Entity collision system | Mouad (`SCRUM-88`) |
| Super-pacgum → ghost edible mode | Nabil |
| Ghost respawn after eaten | Mouad (`SCRUM-43`) |
| Timer, invincibility, leveling system | Mouad (`SCRUM-32`) |
| Pause & Game Over states | Mouad (`SCRUM-33`, `SCRUM-56`, `SCRUM-57`) |
| Score & highscore management | Nabil (`SCRUM-46`, `SCRUM-48` to `SCRUM-53`) |
| Sound effects (ghost eaten, game over) | Nabil (`SCRUM-89`) |
| Custom sprite animations (Pac-Man, ghosts) | Mouad (`SCRUM-98` to `SCRUM-104`) |
| **Bugs fixed:** Collision detection (`SCRUM-93`, `SCRUM-96`, `SCRUM-105`), window resize (`SCRUM-106`) | Mouad |

---

## Phase 3: UI Polish & Refactoring (July 11–18)
**Commits:** 36 | **Contributors:** Mouad, Nabil

| What was done | Who |
|---|---|
| Main menu creation | Mouad (`SCRUM-54`) |
| Victory screen | Nabil (`SCRUM-58`) |
| Highscores screen in menu | Nabil (`SCRUM-59`) |
| Full keyboard/mouse navigation | Mouad (`SCRUM-112`) |
| Button manager system | Mouad |
| Cheat modes (invincibility, freeze, speed, skip, lives) | Mouad (`SCRUM-60` to `SCRUM-64`) |
| Ghost eaten eyes animation | Nabil (`SCRUM-109`) |
| Ghost prison respawn logic | Nabil |
| Code structure refactoring | Mouad (`SCRUM-111`) |
| Flake8/mypy linting fixes | Mouad |
| **Bugs fixed:** Player movement (`SCRUM-94`), ghost respawn (`SCRUM-107`, `SCRUM-113`) | Nabil |

---

## Phase 4: AI Research & Data Collection (July 19–31)
**Commits:** 30 | **Contributors:** Mouad, Nabil, Oussama

| What was done | Who |
|---|---|
| CNN architecture research & planning | Nabil (`SCRUM-130`) |
| MLP formating and features setup | Mouad |
| MLP training implementation | Oussama (`SCRUM-131`) |
| CNN data collection pipeline | Nabil |
| Data collection loop for training | Mouad |
| CNN base architecture | Mouad |
| Supervised Learning (SL) model initial work | Nabil |
| Best model with quantization and integration | Mouad |

---

## Phase 5: RL Training Pipeline (Aug 1–12)
**Commits:** 81 | **Contributors:** Mouad, Nabil

| What was done | Who |
|---|---|
| Reinforcement Learning environment setup | Mouad |
| RL training setup & data creation | Mouad, Nabil |
| Reward system design & fixes | Mouad |
| Oscillation problem diagnosis & 4 structural fixes | Mouad |
| Model architecture iterations (CNN backbone, Actor-Critic) | Mouad |
| Training checkpoint management | Mouad, Nabil |
| SL player model training | Nabil |
| Performance optimizations (pre-allocated tensors, throttled BFS) | Mouad |
| Best model checkpoints saved | Mouad |

---

## Phase 6: Polished Game Branch & Ghost AI Model (Aug 18 – Sep 7)
**Commits:** 65 | **Contributors:** Mouad, Nabil

| What was done | Who |
|---|---|
| Created `polished-game` branch (clean runtime without training code) | Mouad |
| AI Pac-Man activation bound to `Ctrl+A` cheat | Mouad |
| Tactical lookahead search engine (`search_planner.py`) | Mouad |
| Ghost hunter cheat mode with predictive navigation | Nabil |
| Ghost hunter performance optimization (predictive caching) | Nabil |
| Mystery gamble pellet system | Mouad |
| SL ghost model training (dedicated `SL_ghosts_model` branch) | Nabil |
| Adversarial ghost model training | Nabil |
| Final documentation, cleanup, `.flake8` config | Mouad |

---

## Commit Activity Chart

```
Week 1  (Jul 01-04): ██████████████████████████░ 27 commits
Week 2  (Jul 05-10): ██████████████████████████████████░ 34 commits
Week 3  (Jul 11-18): ████████████████████████████████████░ 36 commits
Week 4  (Jul 19-25): ██████████████████████████████░ 30 commits
Week 5  (Jul 31-Aug 05): █████████████████████████████████████████░ 41 commits
Week 6  (Aug 06-12): ████████████████████████████████████████░ 40 commits
     (Aug 13-17): ░ 0 commits (break)
Week 7  (Aug 18-27): ██████████░ 10 commits
Week 8  (Aug 28-Sep 01): ████████████████████████████████████░ 36 commits
Week 9  (Sep 02-07): ██████████████████████████████████████████████░ 46 commits
```

**Total: ~300 commits across 54 active development days.**
