# 02 — Environment (`player_env.py`)

Last updated: **2026-08-23**

`PacmanPlayerEnv` is a headless Pac-Man where the RL policy controls the
player against 4 BFS-driven ghosts. One env `step()` = Pac-Man moving from
cell center to cell center (up to `MAX_PHYSICS_TICKS = 40` physics ticks).

## Constructor

| Arg | Default | Notes |
|---|---|---|
| `seed` | None | Seeds the episode RNG (`self.rng`) |
| `max_steps` | None | If unset, per-episode limit = maze_area × 12 |
| `stage` | 1 | 1: ghosts imprisoned, no collisions; 2: full game |
| `device` | cpu | Device of returned observation tensors |
| `use_bfs_shaping` | False | Optional potential-based reward shaping |
| `maze_{w,h}_{min,max}` | None | Curriculum overrides of default sizes (w 10–30, h 10–20) |
| `start_pellets` | None | Completion curriculum: episode starts with one of these pellet counts (sampled per reset), placed at BFS-farthest walkable cells as **normal** pellets. `None` = classic full map. The eval benchmark never uses this |

Side effects at init: SDL dummy video driver, pygame init, sprite library
load. Delegates created here and shared for the env's lifetime:
`RewardCalculator(stage)` and `GhostController(movement=None, rng=self.rng)` —
**ghosts share the env's RNG**, so seeding the env seeds ghost confusion.

## Episode lifecycle

### `reset()`
1. Clears all per-episode state: histories, oscillation counters, region
   tracking, event counts, **telemetry**, reward breakdown.
2. Draws from `self.rng`: `maze_w`, `maze_h`, then `current_seed ∈ [1, 44444]`.
3. Builds the maze via `LevelManager.build_maze(w, h, seed=current_seed)`.
   ⚠ The third-party `mazegenerator` reseeds Python's *global* `random`
   with that seed during generation — a known side effect; nothing else in
   the RL path consumes global random afterwards.
4. Creates `MovementSystem(self.maze)` and **reseeds
   `movement.rng` with `current_seed`** — it is used for frightened-ghost
   flee targets and would otherwise make eval runs irreproducible.
5. Spawns player at maze center, ghosts in the four corners
   (`EntityFactory`), fills pellets: every walkable cell gets value 1,
   corners get super pellets (value 2), center + spawn cell excluded.
   With `start_pellets` set, instead exactly N normal pellets are placed
   within a per-episode **distance band** from spawn — near (4–9),
   mid (10–17) or far (18+) BFS steps, chosen uniformly — so completion
   starts achievable and gradually demands long-range navigation
   (`_create_pellets(count)`). The +5000 completion reward therefore fires
   regularly and becomes learnable. `info["start_pellets"]` reports the
   episode's count.
6. Initializes visit counts / visited heatmap, region bookkeeping (4×4
   regions), distance-snapshot state.
7. Returns `(grid, features, valid_actions)`.

### `step(action)` — full pipeline in order
1. **Validate action**, update `same_action_count` (stuck detector feature),
   set `player.next_direction`.
2. **Pre-move BFS snapshot** (`movement.bfs_distances` from player cell):
   nearest pellet, nearest active (non-prison, non-edible) ghost, nearest
   power-pellet distances → stored as "previous" values feeding delta features.
   `min_ghost_dist_before` is captured here (stage > 1).
3. **Physics loop** up to 40 ticks:
   - `_update_entities()`: move player, decay `powered_timer` (−0.1/tick,
     ends powered mode → ghosts non-edible), tick all ghosts via
     `GhostController`.
   - On any grid-cell change: `_check_events()` — pellet/super eating
     (super → `start_powered_mode(PUNCH, 45.0)` + all ghosts edible),
     level complete when `remaining_pellets <= 0`, collisions
     (`stage > 1`): edible ghost → eaten + prison + respawn timer
     (`GHOST_RESPAWN_TICKS=2`); dangerous ghost → `pacman_died`, player and
     all ghosts reset positions.
   - Stop when a new cell center is reached or episode ends.
4. **Post-move BFS**: `min_ghost_dist_after` / `threat_dist` (BFS distance to
   nearest hunting ghost; stage > 1 only).
5. **Behavioral events**: oscillation (2-cell A→B→A flip; 4-cell loop on an
   empty cell), backtracking (recent empty cell revisited while safe),
   hunger counter reset on eating, 4×4-region dirty/cleared transitions.
6. **Truncation check** (`step_count >= max_steps`) → `events["truncated"]`.
7. **Reward** via `RewardCalculator.calculate(...)` — see
   [03-rewards.md](03-rewards.md). Breakdown accumulated into
   `episode_reward_breakdown`.
8. **Danger telemetry** (`_update_telemetry`, stage > 1) — see below.
9. **Episode counters** and termination:
   - `terminated` = died ≥ `max(1, LIVES − 1)` (**one death ends the
     episode**, since `LIVES = 2`) OR level completed.
   - `truncated` = step budget exhausted without termination.
10. Build `info`; when `done` it carries `episode_event_counts`,
    `episode_reward_breakdown`, and `info["telemetry"]`.

### `set_seed(seed)`
Reseeds the episode RNG. Used by the eval harness before every benchmark
episode so identical seeds always produce identical mazes/ghosts.

## Danger telemetry (`_update_telemetry`)

Cheap per-step counters with **no reward coupling** — they measure whether
trap-avoidance is being learned long before it shows in score:

| Counter | Meaning |
|---|---|
| `cornered_steps` / `cornered_entries` | Steps spent with ≤1 open neighbour cell **and** ≥1 hunting ghost within Manhattan 8 / how many times that state was entered |
| `escape_success` / `escape_failure` | After leaving a trap: survived `ESCAPE_CONFIRM_STEPS` (8) steps ⇒ success; death inside the trap or within the confirm window ⇒ failure |
| `deaths_cornered` | Deaths attributable to a trap (in-trap or within window) |
| `min_ghost_dist_sum/cnt` | Running mean of BFS distance to nearest hunting ghost |
| `approach_steps` | Steps moving *closer* (BFS) to a hunting ghost while not powered |

State machine flags: `_in_corner_threat` (currently trapped) and
`_open_escape_deadline` (pending escape confirmation). Death handling
attributes the death and closes both.

Interpretation guide lives in
[05-evaluation-and-telemetry.md](05-evaluation-and-telemetry.md).

## Observations (`_get_observation`)

Returns three tensors (details in [04-model-and-observation.md](04-model-and-observation.md)):
- `grid` `[1, 7, 25, 50]` — walls/pellets/super/player/hunting/edible/visit-heat
- `features` `[1, 65]`
- `valid_actions` `[1, 4]` — mask from `movement.can_move`

Optional `use_reverse_mask` (default off) would forbid reversing direction.

## Ghost AI summary (`ghost_controller.py`)

Per physics tick, stage 2, each ghost is in exactly one mode:
- **Prison**: waits out its respawn timer, then re-enters play.
- **Frightened** (edible): moves away from player via BFS
  (`update_runaway_ghost`) at half speed (accumulator 0.5/tick).
- **Hunting**: 30% chance ("confusion") picks a random valid direction
  (never straight reverse), otherwise full-speed BFS chase
  (`update_bfs_ghost`, accumulator 0.75/tick).

All randomness flows through the env's seeded RNG.
