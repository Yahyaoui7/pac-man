# 03 — Rewards (`rewards.py` + `constants.py`)

Last updated: **2026-08-25** — *Minimal Signal Mode*

`RewardCalculator.calculate(...)` runs every env step and returns
`(total_reward, breakdown_dict)`. The breakdown is summed per episode and
printed in the training log (`format_breakdown_line`, sparse: zero terms are
omitted), so each term below is directly observable.

## Minimal Signal Mode (current)

Target stated by the project: **stay alive + eat pellets + zero oscillation**,
with the cleanest possible reward. Active terms:

| Breakdown key | Trigger | Value |
|---|---|---|
| `death` | Pac-Man dies | **−350** |
| `complete` | Level cleared | **+1000** + min(remaining_steps × 0.1, 100) *(lowered from 5000 — frequent tiny-map completions favour a flatter stream)* |
| `pellet` | Normal pellet | +1.5 + 2.0 × cleared-fraction (+4 extra once >75% cleared) |
| `super_pellet` / `super_bait` | Power pellet | +5 (+ up to 6 bait bonus) |
| `milestone` | 50/75/85/95% pellets (once each) | +20/+50/+100/+200 |
| `ghost` | Eat edible ghost | +150 |
| `exploration` | First visit to a tile this episode (**ON** Aug 25 — the carrot for "force more exploration") | **+1 per brand-new tile**, one-time, unfarmable |
| `bfs` | Potential shaping (**ON**) | `3 × (γΦ′ − Φ)`, Φ = −BFS dist to nearest pellet ⇒ ~±3/cell; telescoping ⇒ unfarmable |
| `zone_stagnation` | 12 steps in same 3×3 block, then per step | −5 flat, then −0.5/step (anti-camping) |
| `oscillation` | A→B→A or 4-cell loop without pellet eaten; unconditional; **escalating**: −10 × min(consecutive-offence streak, 3), streak resets on any clean step | −10 → −20 → −30 (cap); ε-explorer steps exempt and streak-neutral |
| `ghost_proximity` | **Re-enabled Aug 26** (ladder contingency B: Death% plateaued ~88% >100 upd vs full-power ghosts). Static field by min BFS ghost distance + approach penalty | d=1 −4 · d=2 −1.5 · d=3 −0.5 · d=4 −0.2 · d=5 −0.05; moving closer: −0.4 × (6−d). Off while powered (hunting stays free) |

Everything else is intentionally disabled in this mode: `step` tax, `hunger`,
`predictive_threat`, `evasion_skill`, `super_bait` extras beyond the base,
`zone_control`, `ghost_proximity`, `survival_truncation`, region terms.
Rationale: one dense gradient toward pellets (`bfs`), one hard signal against
wiggling (`oscillation`), one guard against camping (`zone_stagnation`), plus
the three outcome events (death / completion / pellets). Camping or wiggling
now earns exactly nothing; progress pays continuously; outcomes pay big.

## Historical mode (pre Aug-25): full survival-first stack

Before Minimal Signal Mode these were also active: step tax −0.1 (0 on pellet
steps), hunger −0.5 (>25 pellet-less steps), predictive threat ±(0.15…4.0) by
ghost distance, ghost-proximity field (d=1 −4 … d=5 −0.05 + approach penalty
−0.4×(6−d)), zone-control corner penalties (−10 then −2.5/step), survival
truncation +200+, oscillation only when `threat_dist > 4`. All methods remain
implemented — re-enable by uncommenting their calls in `calculate()`.

Also still implemented but never called in any recent mode:
`_exploration_reward` (+1/new tile), `_evasion_skill_reward`,
`_threat_mastery_reward`, `_ghost_lure_reward`, `_dense_survival_reward`,
region hygiene terms.

## Helpers worth knowing

- `_is_cornered(px, py, maze)` — trap detector: ≤1 open neighbour via maze
  wall-bitmask. Public wrappers `is_cornered()` / `count_threatening()` exist
  for the env's telemetry (no reward coupling).
- `_count_threatening_ghosts(px, py, ghosts)` — non-prison, non-edible,
  **Manhattan** distance ≤ 8.

## Constants (`player/constants.py`)

All numeric values live here (`DEATH_REWARD`, `COMPLETION_REWARD`,
`MILESTONE_REWARDS`, ...). Also:

- `LIVES = 2` → an episode terminates on the **first death**
  (`died >= max(1, LIVES−1)` in the env).
- `MAZE_STEP_MULTIPLIER = 12.0` → episode step budget = maze area × 12.
- `ESCAPE_CONFIRM_STEPS = 8` — telemetry escape-confirm window.
- Unused constants kept for history: `CLOSE_DODGE_REWARD`,
  `ESCAPE_BOX_REWARD`, `BAIT_SUPER_PELLET_*`, `CORNERED_MIN_MOVES`,
  `NEAR_GHOST_DIST`, `SURVIVAL_TRUNCATION_*`.
