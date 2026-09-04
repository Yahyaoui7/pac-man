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
MAZE_WIDTH_MAX = 30
MAZE_HEIGHT_MIN = 10
MAZE_HEIGHT_MAX = 23
MAX_PHYSICS_TICKS = 40
GHOST_RESPAWN_TICKS = 2
MAZE_STEP_MULTIPLIER = 14.0

# ═══════════════════════════════════════════════════════════════════
# STAGE 2 BALANCED REWARDS (Survival + Completion)
# ═══════════════════════════════════════════════════════════════════

STEP_REWARD = -0.1
DEATH_REWARD = -150.0  # Reduced: -50 was dominating gradient, masking intermediate reward signal
OSCILLATION_REWARD = -2.0  # Soft direction-flip penalty when safe
COMPLETION_REWARD = 1000.0
EAT_GHOST_REWARD = 35.0
PELLET_REWARD = 1.0  # Restored strong pellet clearing incentive
SUPER_PELLET_REWARD = 8.0

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

LIVES = 5

# ═══════════════════════════════════════════════════════════════════
# Telemetry (leading indicators for trap-avoidance learning)
# ═══════════════════════════════════════════════════════════════════
# After leaving a cornered+threatened state, a death within this many
# steps retroactively marks the escape attempt as failed.
ESCAPE_CONFIRM_STEPS = 8

# ═══════════════════════════════════════════════════════════════════
# Milestones
# ═══════════════════════════════════════════════════════════════════

MILESTONE_REWARDS = {
    0.25: 50.0,  # 25% of map cleared
    0.50: 100.0,  # 50%
    0.75: 200.0,  # 75%
    0.90: 400.0,  # 90%
}
