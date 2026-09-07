# Acceptance Test Plan

This document lists the features tested before release, mapped to their Jira tickets and the bugs found during testing.

## 1. Configuration & Setup

| Test | Jira Ticket | Result |
|---|---|---|
| Game launches with `python pac_man.py` | — | ✅ Pass |
| `config.json` is parsed correctly (dimensions, lives, level seeds) | `SCRUM-25`, `SCRUM-26` | ✅ Pass |
| A-Maze-ing generates a valid maze from config seeds | `SCRUM-27` | ✅ Pass |
| Subsequent levels generate random mazes | `SCRUM-28` | ✅ Pass |
| Maze generator errors are handled without traceback | `SCRUM-29` | ✅ Pass |

## 2. Player Mechanics

| Test | Jira Ticket | Result |
|---|---|---|
| Pac-Man moves with Arrow Keys and WASD | `SCRUM-35` | ✅ Pass |
| Pac-Man cannot move through walls | `SCRUM-36` | ✅ Pass |
| Player respawns correctly after death | `SCRUM-37` | ✅ Pass |
| Lives decrease on ghost collision | `SCRUM-38` | ✅ Pass |
| **BUG found:** Player movement broken when returning in path | `SCRUM-94` | 🐛 Fixed |

## 3. Ghost AI

| Test | Jira Ticket | Result |
|---|---|---|
| 4 ghosts spawn and move autonomously | `SCRUM-39`, `SCRUM-40` | ✅ Pass |
| Ghosts chase Pac-Man using BFS pathfinding | `SCRUM-41` | ✅ Pass |
| Super-pacgum triggers ghost edible/flee mode | `SCRUM-42` | ✅ Pass |
| Eaten ghosts return to spawn box and respawn | `SCRUM-43` | ✅ Pass |
| Eaten ghost shows eyes animation while returning | `SCRUM-109` | ✅ Pass |
| **BUG found:** Ghost respawn broken after being eaten | `SCRUM-113` | 🐛 Fixed |
| **BUG found:** Ghost reappears incorrectly after specific time | `SCRUM-107` | 🐛 Fixed |

## 4. Pacgums & Scoring

| Test | Jira Ticket | Result |
|---|---|---|
| Standard pacgums placed throughout maze | — | ✅ Pass |
| Super-pacgums placed in corners | `SCRUM-45` | ✅ Pass |
| Score increases when eating pacgums | `SCRUM-46` | ✅ Pass |
| Score increases when eating ghosts (escalating: 200, 400, 800, 1600) | `SCRUM-48` | ✅ Pass |
| Score shown above ghost when eaten | `SCRUM-95` | ✅ Pass |
| **BUG found:** Score not updating on player-entity collision | `SCRUM-93` | 🐛 Fixed |
| **BUG found:** Collision count calculated after move, not on-the-spot | `SCRUM-105` | 🐛 Fixed |

## 5. Highscore System

| Test | Jira Ticket | Result |
|---|---|---|
| Highscore JSON file created and loaded | `SCRUM-49`, `SCRUM-50` | ✅ Pass |
| Score saved at game end | `SCRUM-51` | ✅ Pass |
| Only top 10 scores kept | `SCRUM-52` | ✅ Pass |
| Player name validated (max 10 chars, allowed characters) | `SCRUM-53` | ✅ Pass |

## 6. UI & Game States

| Test | Jira Ticket | Result |
|---|---|---|
| Main menu displays on launch with Start, Highscores, Instructions, Exit | `SCRUM-54` | ✅ Pass |
| Pause menu works (Spacebar/ESC) | `SCRUM-56` | ✅ Pass |
| Game Over screen displays correctly | `SCRUM-57` | ✅ Pass |
| Victory screen displays when all pacgums eaten | `SCRUM-58` | ✅ Pass |
| Highscores viewable from main menu | `SCRUM-59` | ✅ Pass |
| Full keyboard navigation (no mouse required) | `SCRUM-112` | ✅ Pass |
| **BUG found:** Window resizes weirdly on launch | `SCRUM-106` | 🐛 Fixed |
| **BUG found:** Button issues on victory screen | — | 🐛 Fixed |

## 7. Cheat Modes

| Test | Jira Ticket | Result |
|---|---|---|
| Invincibility toggle (`I`) | `SCRUM-60` | ✅ Pass |
| Level skip (`K`) | `SCRUM-61` | ✅ Pass |
| Ghost freeze (`F`) | `SCRUM-62` | ✅ Pass |
| Extra lives (`L`) | `SCRUM-63` | ✅ Pass |
| Speed boost (`B`) | `SCRUM-64` | ✅ Pass |
| AI Autopilot toggle (`Ctrl+A`) | — | ✅ Pass |

## 8. Sound

| Test | Jira Ticket | Result |
|---|---|---|
| Sound plays on menu enter | `SCRUM-86` | ✅ Pass |
| Sound plays during gameplay | `SCRUM-89` | ✅ Pass |
| Sound plays on ghost eaten, game over, pause | `SCRUM-89` | ✅ Pass |

## Summary

| Category | Total Tests | Passed | Bugs Found & Fixed |
|---|---|---|---|
| Config & Setup | 5 | 5 | 0 |
| Player | 5 | 5 | 1 |
| Ghost AI | 6 | 6 | 2 |
| Pacgums & Scoring | 6 | 6 | 2 |
| Highscore | 4 | 4 | 0 |
| UI & States | 7 | 7 | 2 |
| Cheat Modes | 6 | 6 | 0 |
| Sound | 3 | 3 | 0 |
| **Total** | **42** | **42** | **7** |
