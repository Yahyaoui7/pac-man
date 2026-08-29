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
MAZE_WIDTH_MAX = 40
MAZE_HEIGHT_MIN = 10
MAZE_HEIGHT_MAX = 20
MAX_PHYSICS_TICKS = 40
GHOST_RESPAWN_TICKS = 2
MAZE_STEP_MULTIPLIER = 12.0

# ═══════════════════════════════════════════════════════════════════
# STAGE 2 BALANCED REWARDS (Survival + Completion)
# ═══════════════════════════════════════════════════════════════════

STEP_REWARD = -0.1
DEATH_REWARD = -350.0
OSCILLATION_REWARD = -10.0
COMPLETION_REWARD = 1000.0
EAT_GHOST_REWARD = 150.0
PELLET_REWARD = 1.5
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

# ═══════════════════════════════════════════════════════════════════
# Survival shaper magnitudes (re-enabled + tuned Aug 28)
# ═══════════════════════════════════════════════════════════════════

# Per-step alive bonus near threats — small so it doesn't overwhelm pellet signal.
DENSE_SURVIVAL_REWARD = 0.3
# Scaling for escape credit after a close call.
EVASION_ESCAPE_BASE = 1.0
# Scale for near-threat prediction penalty.
PREDICTIVE_THREAT_NEAR = 1.0
# Per-step reward for surviving near a threat at safe distance (2-3).
THREAT_MASTERY_SURVIVE = 0.3
# Large reward for surviving to the episode timeout.
SURVIVAL_TRUNCATION_BONUS = 150.0

LIVES = 2

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
    0.50: 20.0,
    0.75: 50.0,
    0.85: 100.0,
    0.95: 200.0,
}
