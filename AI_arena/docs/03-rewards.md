# 03 — Rewards (`rewards.py` + `constants.py`)

Last updated: **2026-08-23**

`RewardCalculator.calculate(...)` runs every env step and returns
`(total_reward, breakdown_dict)`. The breakdown is summed per episode and
printed in the training log (`format_breakdown_line`), so each term below is
directly observable.

## Active terms (Stage 2)

| Breakdown key | Trigger | Value |
|---|---|---|
| `death` | Pac-Man dies | **−350** |
| `complete` | Level cleared | **+5000** + min(remaining_steps × 0.1, 100) |
| `pellet` | Normal pellet | +1.5 + 2.0 × cleared-fraction (+4 extra once >75% cleared) |
| `super_pellet` / `super_bait` | Power pellet | +5 (+ up to 6 scaled by #threatening ghosts — bait bonus) |
| `milestone` | 50/75/85/95% pellets (once each) | +20/+50/+100/+200 |
| `ghost` | Eat edible ghost | +150 |
| `step` | Every step | −0.1 (0 on pellet/super steps) |
| `hunger` | >25 steps without a pellet | −0.5 |
| `zone_stagnation` | 12 steps in same 3×3 block, then per step after | −5 flat, then −0.5/step |
| `oscillation` | A→B→A or 4-cell loop, no pellet eaten, threat dist > 4 | −10 |
| `predictive_threat` | Hunting ghost near (Manhattan ≤ 8), not powered | d≥5: +0.30 · d=4: +0.15 · d=3: −0.5 · d=2: −1.5 · d=1: −4.0 |
| `zone_control` | Cornered (≤1 open neighbour) while not powered | threatened: −10, then −2.5/extra consecutive step; safe corner: −0.1 |
| `survival_truncation` | Episode ends on step budget | +200 + cleared% × 50 |
| `ghost_proximity` *(stage>1)* | Static repulsion field by min BFS ghost distance | d=1 −4 · d=2 −1.5 · d=3 −0.5 · d=4 −0.2 · d=5 −0.05; plus moving closer: −0.4 × max(0, 6−d) |

Scale intuition: one completion ≈ 16 deaths; one death ≈ 230 pellets.
The reward is deliberately survival-first: dying costs more than any single
episode can earn back through pellets alone.

## Defined but currently DISABLED (not called from `calculate()`)

These exist for experiments — enable by adding the call in `calculate()`:

| Method | Purpose | Note |
|---|---|---|
| `_exploration_reward` | +1 per brand-new tile visited | needs `visited_this_episode` maintenance if enabled |
| `_evasion_skill_reward` | Credit for increasing distance to a close ghost (d≤4) | pairs with `last_min_ghost_dist` tracking |
| `_threat_mastery_reward` | Sustained survival while ghosts within Manhattan 2–3 | uses `consecutive_threat_steps` |
| `_ghost_lure_reward` | Approaching edible ghosts while powered | hunting behavior shaping |
| `_dense_survival_reward` | Small per-step bonuses when threatened | overlaps with `predictive_threat` |
| `_region_cleared/_dirty_penalty/_backtrack_penalty/_incomplete_penalty` | 4×4-region hygiene & timeout penalty | region events still computed in env; only rewards are off |

Prepared-but-unused state also present: `in_danger_zone`,
`danger_zone_entry_step/pos` (a danger-zone mechanic never implemented).

## Helpers worth knowing

- `_is_cornered(px, py, maze)` — trap detector: ≤1 open neighbour via maze
  wall-bitmask. Public wrappers `is_cornered()` / `count_threatening()`
  exist for the env's telemetry (no reward coupling).
- `_count_threatening_ghosts(px, py, ghosts)` — non-prison, non-edible,
  **Manhattan** distance ≤ 8. Returns counts + min distances to both
  threatening and edible ghosts.
- Stage gating: `ghost_proximity` and the `last_min_ghost_dist` update only
  run when `stage > 1`.

## Constants (`player/constants.py`)

All numeric values above live here (`DEATH_REWARD`, `COMPLETION_REWARD`,
`MILESTONE_REWARDS`, ...). Also:

- `LIVES = 2` → an episode terminates on the **first death**
  (`died >= max(1, LIVES−1)` in the env).
- `MAZE_STEP_MULTIPLIER = 12.0` → episode step budget = maze area × 12.
- Unused constants kept for history: `CLOSE_DODGE_REWARD`,
  `ESCAPE_BOX_REWARD`, `BAIT_SUPER_PELLET_*`, `CORNERED_MIN_MOVES`,
  `NEAR_GHOST_DIST`, `SURVIVAL_TRUNCATION_*` (the active truncation bonus is
  hardcoded as 200/50 inside `_survival_truncation_reward`).
- `ESCAPE_CONFIRM_STEPS = 8` — telemetry window after leaving a trap within
  which a death retroactively fails the escape attempt.
