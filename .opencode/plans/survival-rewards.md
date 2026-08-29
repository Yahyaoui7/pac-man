# Plan: Fix Survival Rewards for Pac-Man AI (500-update quick test)

## Context
The model is stuck at 100% death rate — it collects pellets but never learns to
survive ghost chases. There are 13 disabled survival reward shapers in
`rewards.py`. The env already computes the needed state (ghost BFS distances,
threat counts), but the shapers are never called. Two latent bugs also block
them from working.

## Bugs discovered during research
1. `_dense_survival_reward` writes to `breakdown["survival_truncation"]`
   instead of a dedicated `dense_survival` key (which doesn't exist in the
   breakdown dict) → would crash if enabled. Fix by adding the key + updating.
2. `last_min_ghost_dist` tracking is gated behind `self.stage > 1` and only
   written in the proximity block. `_evasion_skill_reward` needs it every step.
   Move it outside the gate.

## File 1: `AI_arena/player/constants.py`
Add after `NEAR_GHOST_DIST = 2`:
```python
DENSE_SURVIVAL_REWARD = 0.3
EVASION_ESCAPE_BASE = 1.0
PREDICTIVE_THREAT_NEAR = 1.0
THREAT_MASTERY_SURVIVE = 0.3
SURVIVAL_TRUNCATION_BONUS = 150.0
```

## File 2: `AI_arena/player/rewards.py`
- Import the 5 new constants.
- Add `"dense_survival": 0.0` to the `breakdown` dict.
- Fix `_dense_survival_reward` (lines ~359-372) to write `breakdown["dense_survival"]`.
- Tune `_evasion_skill_reward` (lines ~269-288) with `EVASION_ESCAPE_BASE`, cap total.
- Soften far-side of `_predictive_threat_reward` (lines ~245-267) so it doesn't
  fight pellet collection; keep steep dist-1/dist-2 penalties.
- `_survival_truncation_reward` (lines ~351-357): use `SURVIVAL_TRUNCATION_BONUS`.
- `_threat_mastery_reward` (lines ~311-329): extend to 8 consecutive steps,
  use `THREAT_MASTERY_SURVIVE`.
- In `calculate()` (lines ~484-516):
  - Move `last_min_ghost_dist` update OUTSIDE the `stage > 1` gate.
  - Inside `if self.stage > 1:` call, in this order:
    `_ghost_proximity_penalty`, `_predictive_threat_reward`,
    `_evasion_skill_reward`, `_threat_mastery_reward`,
    `_dense_survival_reward`, `_survival_truncation_reward`.
  - Order matters: evasion/lure must read OLD `last_min_ghost_dist` before it's
    updated at the end of the step.

## File 3: `AI_arena/player/player_env.py`
No functional change required. Note: `min_ghost_dist_before/after` are BFS
(Dijkstra) distances while `_count_threatening_ghosts` uses Manhattan (<=8).
Shapers use `min_threat_dist` (Manhattan) for consistency with threat counting.

## File 4: `AI_arena/player/player_training.py`
Line 35: `NUM_UPDATES = 1000` → `500`.

## File 5: Broken `ghost_training` imports
File `AI_arena/ghosts/ghost_training.py` was deleted, but imports remain:
- `AI_arena/__init__.py`: `from AI_arena.ghosts.ghost_training import train as train_ghost_cnn` (broken) + stale `__all__` entries.
- `AI_arena/ghosts/__init__.py`: same broken import.
Remove those import lines and clean stale `__all__` entries.

## Verification
1. `python -c "import AI_arena"` — must succeed.
2. Short dry-run to confirm no crashes.
3. Full run NUM_UPDATES=500; check `AI_arena/evals/eval_history.json`:
   - death_rate should drop below 100%
   - avg_pellet_pct may drop (survival vs pellet tradeoff)
   - survival telemetry (cornered, escape) improves

## Risks
- Over-rewarding survival → model parks and never eats pellets. Dense reward
  (0.3/step) kept small; truncation bonus only fires at timeout.
- May need retuning after first eval if death rate flat but pellets crater.
