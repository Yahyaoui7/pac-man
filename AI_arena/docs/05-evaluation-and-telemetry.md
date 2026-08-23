# 05 — Evaluation, Telemetry & Stall Detection

Last updated: **2026-08-23**

This is the "is it actually learning?" machinery. It exists because
trap-avoidance / anti-surrounding needs 10k–20k episodes to move win-rate or
pellet% — these tools give a readable verdict at ~1–3k episodes.

## The core idea

Separate **exposure** from **skill**:
- Getting cornered while a ghost hunts you is largely *luck* (maze + ghost
  spawns). Exposure rate is not a skill measure.
- Escaping that situation alive *is* the skill. Escape success rate can rise
  long before the agent learns to avoid traps in the first place.

Expected learning trajectory of the indicators:

| Metric | Trend if learning | Where |
|---|---|---|
| `Life:` moves survived + % of step budget used (`steps / (maze_area×12)`) | ↑ first alongside Esc — **the survival-rate metric**; raw move counts are misleading across maze sizes, so judge the % | SURV line + eval |
| `Esc:` escape success rate | ↑ early — earliest trap-specific signal | SURV line + eval |
| `CDth:` deaths-while-cornered share of deaths | ↓ early | SURV line + eval |
| `Corn:` cornered steps per episode | ↓ (avoidance emerging) | SURV line + eval |
| `MinD:` mean min BFS ghost distance | ↑ (keeps margins) | SURV line + eval |
| `Appr:` % steps moving closer to hunting ghosts | ↓ | eval JSON only |
| death%, pellet%, comp% | move last | EVAL line |

Note on `Life:`: while death% is 100%, every episode ends by dying, so
moves-lived *is* the survival time. Once truncations/completions appear, Life
saturates toward ~100% and outcomes (comp%/death%) become the differentiator.

Rule of thumb for killing a run: if eval score AND Esc%/Corn%/CDth are all
flat over ~250 updates, the run is not learning this skill — change reward or
curriculum instead of waiting.

## Per-step telemetry (`player_env._update_telemetry`, stage > 1)

Counters per episode (returned as `info["telemetry"]` when done):
`cornered_steps`, `cornered_entries`, `escape_success`, `escape_failure`,
`deaths_cornered`, `min_ghost_dist_sum/cnt`, `approach_steps`.

Definitions: *trapped* = ≤1 open neighbour cell (wall bitmask) AND ≥1 hunting
ghost within Manhattan 8. Leaving a trap opens an 8-step confirm window
(`ESCAPE_CONFIRM_STEPS`); surviving it = success, dying inside the trap or in
the window = failure (and counts toward `deaths_cornered`). Deaths reset the
state machine.

Aggregation across episodes: `compute_survival_stats(episodes)` in
`utils/metrics.py` → rates/per-episode means; `format_survival_line(stats)`
renders the `SURV | ...` log line every training update (window = update's
episodes, falling back to the rolling last-100).

## Fixed-seed benchmark (`utils/evaluate.py`)

- 20 episodes on seeds `10000..10019` (`EVAL_SEED_BASE` — never change;
  identical mazes make checkpoint comparisons paired).
- Greedy argmax policy (deterministic), fresh GRU state per episode,
  separate env instance.
- Composite score (defined once in `eval_score`):
  `score = pellet% + 40 × completion_rate − 30 × death_rate`
- Every run appends a full record (incl. survival stats + per-episode detail)
  to `AI_arena/evals/eval_history.json`.
- **Comparability rule:** scores are only valid *within identical game
  parameters*. Each record stores `env` provenance (`stage`,
  `ghost_confusion`, …) — changing difficulty (e.g. confusion 0.30 → 0.01)
  starts a new effective baseline; never compare across it.

### From the CLI

```bash
uv run python -m AI_arena.player.utils.evaluate                 # latest best ckpt
uv run python -m AI_arena.player.utils.evaluate --compare       # + history table
# options: --checkpoint PATH --episodes N --stage 2 --device cpu --sample --no-save
```

### From training

Runs automatically inside the save block every `EVAL_INTERVAL=50` updates and
logs `EVAL @NNN: score ... | pellet ... | death ... | life Nmv/N% | esc ...[n] | corn .../ep | MinD ...`.
Cost ≈ one extra update per 50 (~2%).

### Best-checkpoint ownership

Until the first eval completes, `*_best.pt` follows train-window pellet%
(legacy). After the first eval, the eval score owns it exclusively
(`eval_best_active` flag) — pellet% is too noisy/lagging to pick the best
survival policy.

## Determinism guarantees

For identical seeds, episodes are bit-reproducible given the same policy:

1. Maze size + maze seed come from the env RNG (`set_seed` before each
   episode).
2. `MovementSystem.rng` (frightened flee targets) is reseeded with the maze
   seed in `reset()` — previously unseeded, which broke reproducibility.
3. Ghost confusion draws from the same env RNG.
4. Greedy actions; CPU float32 forward passes.

⚠ `mazegenerator` reseeds Python's *global* random during maze generation
(side effect). Harmless today because nothing downstream uses global random,
but keep it that way.

## Stall detection (training loop)

State tracked at each eval: `best_eval_score`, `best_eval_update`,
`last_meaningful_improve_upd` (last eval where score beat the previous best
by ≥ `EVAL_MIN_IMPROVEMENT = 2.0`), `last_stall_warn_update`.

Warning fires once per stall stretch when:
`update − last_meaningful_improve_upd ≥ EVAL_STALL_PATIENCE × EVAL_INTERVAL`
(default 5 × 50 = 250 updates).

Deliberate details:
- "Best" high-water marks can drift up on noise (+0.01); stall detection keys
  off *meaningful* improvements only, so slow-noise drift doesn't mask a
  plateau.
- The banner clears itself on the next meaningful improvement.
- The banner explicitly asks the human to check whether SURV lines are flat —
  flat score + flat Esc%/Corn% is the kill signal.

## Reading eval history offline

```python
import json
hist = json.load(open("AI_arena/evals/eval_history.json"))
for r in hist:
    s = r["survival"]
    print(r["update"], round(r["eval_score"], 1), r["checkpoint"],
          f"esc={s['escape_rate']:.2f} corn={s['cornered_steps_per_ep']:.2f}")
```

Each record also contains `episodes_detail` (per-seed results) for drilling
into specific failure mazes.
