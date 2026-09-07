# Risk Analysis & Mitigation

This document identifies the technical risks encountered during the Pac-Man project and the mitigation strategies the team applied.

## Risk 1: A-Maze-ing Package Integration

| | |
|---|---|
| **Risk** | The project required integrating the external `A-Maze-ing` maze generator package without modifying its source code. The generator could produce incompatible layouts, crash on certain dimensions, or return mazes without valid paths. |
| **Impact** | High — a broken maze means the game cannot start. |
| **Likelihood** | Medium |
| **Mitigation** | Mouad built an abstraction layer that validates the maze output before passing it to the renderer. The `config.json` provides explicit level seeds per level, and the game falls back to random generation for subsequent levels (`SCRUM-27`, `SCRUM-28`, `SCRUM-29`). Error handling was added to catch generator exceptions cleanly (`SCRUM-29`). |
| **Outcome** | ✅ Resolved. Maze generation works reliably across levels. |

## Risk 2: Ghost AI Collision Clipping

| | |
|---|---|
| **Risk** | Pac-Man and ghosts could clip through each other between frames due to the grid-to-pixel movement system, causing missed collisions or unfair deaths. |
| **Impact** | High — breaks core gameplay. |
| **Likelihood** | High |
| **Mitigation** | Nabil implemented tile-center collision detection and coordinate snapping. Multiple bugs were discovered and fixed via Jira: `SCRUM-93` (score not updating on collision), `SCRUM-96` (collision detection failure), `SCRUM-105` (collision count calculated after move instead of on-the-spot). Each bug was tracked, assigned, and resolved. |
| **Outcome** | ✅ Resolved after 3 bug-fix iterations. |

## Risk 3: Ghost Respawn Logic

| | |
|---|---|
| **Risk** | After a ghost is eaten during super-pacgum mode, it must return to the center spawn box, wait, then respawn. The respawn timing and pathfinding could break across level transitions. |
| **Impact** | Medium — ghosts could get stuck or respawn in walls. |
| **Likelihood** | Medium |
| **Mitigation** | Nabil tracked this through `SCRUM-43`, `SCRUM-107`, and `SCRUM-113`. The ghost respawn was fixed multiple times (visible in Git: `fix ghost respawn after being eaten`, `fix: resolve IndexError in find_player_spawn by using actual maze bounds on level transitions`). Eyes animation was added to show ghost returning to spawn (`SCRUM-109`). |
| **Outcome** | ✅ Resolved. Ghosts reliably respawn across all levels. |

## Risk 4: Neural Network Model Instability

| | |
|---|---|
| **Risk** | The CNN-based Pac-Man autopilot could oscillate (get stuck in loops), ignore pellets, or run directly into ghosts. Training on random mazes with varying dimensions added complexity. |
| **Impact** | High — the AI bonus feature would not work. |
| **Likelihood** | High |
| **Mitigation** | The team iterated heavily on the RL training pipeline over 5+ weeks (Aug 1 – Sep 5). Git shows explicit fixes: `resulving the oscillation problem`, `fix: break oscillation attractor — 4 structural fixes`, `trying to fix the ignorance of pellets and head first ghosts`. A tactical lookahead search (`search_planner.py`) was layered on top of the raw neural output to prevent fatal moves. |
| **Outcome** | ✅ Resolved. The autopilot (`Ctrl+A`) works reliably in-game. |

## Risk 5: Merge Conflicts Between Parallel Workstreams

| | |
|---|---|
| **Risk** | Mouad and Nabil worked simultaneously on different systems (UI vs. ghosts, engine vs. sound) that touched shared files. Multiple branches (`Player_RL`, `SL_ghosts_model`, `polished-game`) increased conflict risk. |
| **Impact** | Medium — could cause regressions or lost work. |
| **Likelihood** | High |
| **Mitigation** | The team used feature branches and merged frequently. Git history shows 10+ merge commits with explicit conflict resolution (`fixing the merge conflicts`, `conflict the merge`, `merging the bug fix`). Communication at the 1337 campus enabled real-time coordination. |
| **Outcome** | ✅ Managed. No work was lost. |
