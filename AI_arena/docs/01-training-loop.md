# 01 — Training Loop (`player_training.py`)

Last updated: **2026-08-23**

Recurrent PPO (actor-critic + GRU memory) with hard action masking.
Entry point: `python -m AI_arena.player.player_training` (from repo root).

## Configuration (top of file, edit by hand)

| Constant | Value | Notes |
|---|---|---|
| `STAGE` | 2 | Stage 1 = ghosts jailed, no collisions; Stage 2 = full game |
| `NUM_UPDATES` | 1000 | PPO updates per run |
| `ROLLOUT_STEPS` | 3000 | Env steps collected per update |
| `SEQ_LEN` | 16 | BPTT chunk length for the GRU |
| `NUM_SEQUENCES` | 187 | = ROLLOUT_STEPS // SEQ_LEN; rollout is truncated to fit exactly |
| `MINIBATCH_SEQS` | 16 | Sequence chunks per minibatch (~64 frames) |
| `PPO_EPOCHS` | 2 | Passes over the rollout |
| `LEARNING_RATE` | 2e-4 | Adam |
| `GAMMA` / `GAE_LAMBDA` | 0.99 / 0.95 | Discount / GAE smoothing |
| `CLIP_EPS` | 0.2 | PPO ratio clip |
| `ENTROPY_COEF` | 0.015 | Raised to 0.04 during the exploration-collapse rescue (Aug 2026), annealed back once Ent recovered to ~0.65 |
| `ROLLOUT_EPSILON` | 0.06 | ε-uniform exploration during data collection; 0 disables (see §Rollout). Was 0.12 during the rescue phase — annealed alongside entropy once exploration was healthy |
| `START_PELLETS` | (1, 2, 3) | Completion curriculum: episodes start with few pellets placed in per-episode distance bands (near 4–9 / mid 10–17 / far 18+ BFS steps). `None` = full map. Eval benchmark unaffected |
| `USE_BFS_SHAPING` | True | Potential-based shaping `γΦ(s′)−Φ(s)`, Φ = −BFS-dist to nearest pellet (~±3/cell). Self-scales with pellet density (fades as mazes fill). Keep ON through all curriculum stages |

### Graduation ladder (Aug 25 plan)

Graduate when **all** hold over ≥30 updates / ≥2 evals:
**Death% < 50** · pellets/ep ≥ 60% of stage max · Complete% ≥ 15 · Osc% ≤ 20.
Change ONE variable per step (the pellet schedule); freeze confusion & rewards.

| Stage | START_PELLETS | Expected on entry | If stuck after ~150 upd |
|---|---|---|---|
| S0 | (1,2,3) | current phase | see contingencies below |
| S1 | (4,6,10) | Death spikes ↑ (longer exposure), recovers within ~100 upd | Death >80% sustained → enable `_ghost_proximity_penalty` ONLY (survival teacher), keep rest minimal |
| S2 | (15,25) | Truncated% may climb (longer episodes, no timeout reward) | Truncated >25% → re-enable `_survival_truncation_reward`; pellets high but Complete flat → grazing trap, accept or raise completion bonus |
| S3 | None (full map) | corner super-pellets return automatically → powered mode + ghost-hunting (+150) reappears untrained; expect temporary regression blip | give it 200 upd before judging |

Contingencies (any stage):
- Ent < 0.2 → double ENTROPY_COEF temporarily (not ε)
- Camping (Life↑ pellets↓ Truncated↑) → zone_stagnation −0.5→−1/step after block entry
- Osc% rebounds >25% → verify ε value active; consider streak cap 3→4
- Death >90% for 150 upd → drop back ONE stage (no shame, it's a ladder)
| `VALUE_COEF` | 0.25 | Value-loss weight |
| `MAX_GRAD_NORM` | 0.5 | Global grad clip |
| `SEED` | 42 | Training env seed |
| `SAVE_INTERVAL` | 50 | Checkpoint + eval cadence (updates) |
| `RESUME` | True | Load latest checkpoint before training |
| `SL_WARMSTART` | False | Init from supervised weights + KL leash to them |

Eval constants (`EVAL_*`) are documented in
[05-evaluation-and-telemetry.md](05-evaluation-and-telemetry.md).

## Startup (`train()`)

1. Device selection (CUDA if available) + AMP GradScaler (CUDA only).
2. `TrainingLogger` — PID-locked append to `training_log.txt`; a second
   training process refuses to start while the lock is alive.
3. Checkpoint paths: `AI_arena/models/player_rl_stage{STAGE}.pt` (latest) and
   `..._best.pt` (best).
4. **Resume order** when `RESUME=True`: stage checkpoint → best checkpoint →
   `data/` fallbacks. First existing candidate wins
   (`load_checkpoint_into_policy` tolerates feature-count changes by slicing
   the projection layer).
5. If nothing loaded and `SL_WARMSTART=True`: load SL weights, freeze the CNN
   backbone, create a frozen reference policy (KL leash `kl_coef=0.20`,
   entropy 0.001, LR 5e-5). Otherwise: fresh random weights.

## Per-update flow

### 1. Rollout collection (3000 steps)
- Samples actions from a masked categorical: invalid moves get logit `-1e4`.
- **ε-explorer** (`ROLLOUT_EPSILON > 0`): branch-explicit sampling — with
  prob ε sample a uniformly random *valid* action, otherwise sample from π;
  the stored log-prob is always the true mixture probability
  `(1−ε)·π(a) + ε·uniform(a)`, so PPO ratios stay correct. Each step is
  labelled `explore` when it came from the uniform branch, and
  `PacmanPlayerEnv.step(..., explore=True)` **exempts those steps from the
  oscillation penalty** (random detours must stay unpunished). The Osc%
  metric still counts them — expect inflated Osc% while ε is active; judge
  policy oscillation only after ε is lowered. Rationale: once the policy
  saturates (Ent ≈ 0.1), the entropy bonus's gradient vanishes and — in a
  punishing environment — deviations die fast, so the policy re-sharpens.
  The ε-floor guarantees exploratory data no matter how deterministic π
  gets. Expect worse short-term metrics after enabling/raising it; judge on
  evals ≥100 updates later.
- Every `SEQ_LEN` steps, the current GRU hidden state is stored
  (`rollout_seq_hiddens`) so BPTT chunks can start from the right memory.
- Hidden state resets to `None` on episode end or mid-episode death.
- Each finished episode appends an `ep_record` (reward, pellets, pct, steps,
  osc stats, maze size, event counts, reward breakdown, **telemetry**) to both
  `recent_episodes` (rolling 100) and `save_window_episodes`.

### 2. GAE (`compute_gae`)
- Standard GAE with two distinct flags:
  - `terminated` — true terminal (death or level complete): bootstrap value is
    zeroed via `next_non_terminal`.
  - `dones` — includes time-limit truncations: cuts advantage propagation but
    still bootstraps from the final observation's value (computed right after
    rollout as `next_value`). This avoids penalizing timeouts as deaths.
- Advantages are normalized globally across the update.

### 3. Reshape into sequence chunks
All tensors are cut to `NUM_SEQUENCES * SEQ_LEN` steps and viewed as
`(187, 16, ...)`. Chunk-start hidden states come from the pre-recorded
`b_seq_hiddens`.

### 4. PPO epochs
- Per minibatch (16 random sequence chunks):
  forward with `dones=mb_resets` so the GRU zeroes memory at episode borders,
  masked logits, clipped surrogate loss, `smooth_l1` value loss,
  entropy bonus, optional KL-vs-reference loss, AMP backward, grad clip 0.5.
- Total loss: `policy + 0.25*value − eff_ent_coef*entropy (+ 0.20*kl)`.

### 5. Logging
Two lines per update (fields glossarized in [README.md](README.md)):
the main `Upd ...` line and the `SURV | ...` danger-skill line computed by
`compute_survival_stats` over the same window.

### 6. Best-checkpoint logic
- Until the **first** eval runs: best-by-train-window pellet% (legacy).
- After the first eval: the fixed-seed eval score owns `*_best.pt`
  (`eval_best_active` flag). Train-window "best" saves stop permanently.

### 7. Periodic save + eval + stall detection (every `SAVE_INTERVAL`)
- Saves latest checkpoint, then runs the benchmark eval and appends it to
  `AI_arena/evals/eval_history.json`. Logs an `EVAL @NNN:` line with score
  delta vs previous eval, and `EVAL: new best ...` on high-water marks.
- Stall detection: tracks the last update where the eval score improved by ≥
  `EVAL_MIN_IMPROVEMENT` (2.0) points over the previous best
  (`last_meaningful_improve_upd`). If that exceeds
  `EVAL_STALL_PATIENCE * EVAL_INTERVAL` updates (default 250), a prominent
  `STALL WARNING` banner is logged once per stall stretch. It clears
  automatically on the next meaningful improvement.

> Why this exists: trap-avoidance needs 10k–20k episodes to show up in
> win-rate/pellet%. The eval + SURV telemetry give a readable verdict at
> ~1–3k episodes so dead runs can be killed early instead of discovered late.

### 8. Shutdown paths
- `'q'` (QuitListener thread) or Ctrl-C: save latest checkpoint, log summary.
- `finally` block always logs which checkpoint holds the best score and where
  the eval history lives.

## Operational notes

- ~37 s/update on CPU at these settings; one eval adds ~30 s every 50 updates.
- The lock file `training_log.lock` must not exist for a new session; stale
  locks are auto-removed only if the PID is dead.
- Memory: rollout tensors are explicitly deleted each update +
  `torch.cuda.empty_cache()` on CUDA.
