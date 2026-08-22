"""Game constants for Pac-Man RL environment — Survival Mode."""

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
MAX_PHYSICS_TICKS = 40
GHOST_RESPAWN_TICKS = 2
MAZE_STEP_MULTIPLIER = 12.0

# ═══════════════════════════════════════════════════════════════════
# STAGE 2 BALANCED REWARDS (Survival + Completion)
# ═══════════════════════════════════════════════════════════════════

STEP_REWARD = -0.01
DEATH_REWARD = -150.0
OSCILLATION_REWARD = -10.0
COMPLETION_REWARD = 5000.0
EAT_GHOST_REWARD = 150.0
PELLET_REWARD = 3.0
SUPER_PELLET_REWARD = 5.0

SURVIVAL_TRUNCATION_BASE = 10.0
SURVIVAL_TRUNCATION_PELLET_BONUS = 20.0

# ═══════════════════════════════════════════════════════════════════
# Stage 2 survival shaping
# ═══════════════════════════════════════════════════════════════════

CLOSE_DODGE_REWARD = 1.0
ESCAPE_BOX_REWARD = 4.0
BAIT_SUPER_PELLET_REWARD = 5.0
BAIT_SUPER_PELLET_RADIUS = 4
CORNERED_MIN_MOVES = 4
NEAR_GHOST_DIST = 2

LIVES = 2

# ═══════════════════════════════════════════════════════════════════
# Milestones
# ═══════════════════════════════════════════════════════════════════
MILESTONE_REWARDS = {
    0.25: 20.0,
    0.50: 50.0,
    0.75: 100.0,
    0.90: 200.0,
}
