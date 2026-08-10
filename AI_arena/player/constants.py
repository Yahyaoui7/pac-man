"""Game constants for Pac-Man RL environment."""

from __future__ import annotations

DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")

GHOST_SPECS = [
    ("Blinky", (255, 0, 0)),
    ("Pinky", (255, 182, 193)),
    ("Inky", (0, 255, 255)),
    ("Clyde", (255, 165, 0)),
]

MAZE_WIDTH_MIN = 10
MAZE_WIDTH_MAX = 43
MAZE_HEIGHT_MIN = 10
MAZE_HEIGHT_MAX = 23

MAX_PHYSICS_TICKS = 300
GHOST_RESPAWN_TICKS = 25
MAZE_STEP_MULTIPLIER = 9.0

# ── Reward constants ──
STEP_REWARD = -0.4
DEATH_REWARD = -20.0
OSCILLATION_REWARD = -2.0
COMPLETION_REWARD = 500.0
PELLET_REWARD = 1.0
EAT_GHOST_REWARD = 40.0
SUPER_PELLET_REWARD = 2.0
LIVES = 100

MILESTONE_REWARDS = {
    0.10: 5.0,
    0.20: 5.0,
    0.30: 5.0,
    0.40: 5.0,
    0.50: 10.0,
    0.60: 10.0,
    0.70: 10.0,
    0.80: 15.0,
    0.90: 20.0,
}
