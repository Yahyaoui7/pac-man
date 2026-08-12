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
MAZE_WIDTH_MAX = 15
MAZE_HEIGHT_MIN = 10
MAZE_HEIGHT_MAX = 15

MAX_PHYSICS_TICKS = 300
GHOST_RESPAWN_TICKS = 25
MAZE_STEP_MULTIPLIER = 12.0

# ── Reward constants ──
# ← CHANGED: step penalty is mild — time pressure, not torture
STEP_REWARD = -0.002

# ← CHANGED: death should feel bad, but not break training
DEATH_REWARD = -200.0

# ← CHANGED: oscillation is a gentle nudge, not a sledgehammer
# A random agent will still oscillate; we don't want to drown the signal
OSCILLATION_REWARD = -15.0

# ← CHANGED: completion is the jackpot — make it impossible to ignore
COMPLETION_REWARD = 1500.0

# ← CHANGED: pellets are the primary income source
PELLET_REWARD = 1.5

EAT_GHOST_REWARD = 90.0
SUPER_PELLET_REWARD = 5.0

# ← CHANGED: fewer lives = shorter episodes = more completion attempts per hour
# 100 lives means every episode is 100% truncation. The agent never gets a "fresh start."
LIVES = 20

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
