# 04 — Model & Observation

Last updated: **2026-08-23**

## Observation contract (per step)

Built by `format_player_observation` (`AI_arena/player/data/observation.py`)
on top of the shared `ObservationFormatter` (`AI_arena/data/formatter.py`).

### 1. Grid — `[1, 7, 25, 50]`

| Channel | Content |
|---|---|
| 0 | Walkable cells (`maze != 15`) |
| 1 | Normal pellets |
| 2 | Power pellets |
| 3 | Player position (one-hot cell) |
| 4 | Hunting ghosts (count per cell; prison ghosts included as non-edible) |
| 5 | Edible ghosts |
| 6 | Visit-count heatmap this episode (saturates at 10 visits) |

Mazes are drawn top-left into a fixed 25×50 canvas; smaller mazes leave a
zero border. The network therefore also sees maze size implicitly.

### 2. Feature vector — `[1, 65]` (order matters)

| Group | Dim | Content |
|---|---|---|
| Player direction | 4 | one-hot of UP/DOWN/LEFT/RIGHT |
| Ghost directions | 16 | 4 ghosts × 4 dirs |
| Ghost edible flags | 4 | |
| Frightened timers | 4 | normalized by `POWER_TIMER_MAX = 30` |
| Valid actions mask | 4 | copy of the action mask |
| Pellets remaining | 2 | normal + power, fraction of initial count |
| Power timer | 1 | player's powered timer / 30 |
| Ghost BFS distances | 4 | `(d+1)/max_dim` per ghost |
| Nearest power pellet BFS dist | 1 | normalized |
| Nearest normal pellet BFS dist | 1 | normalized |
| Maze size | 3 | w/50, h/25, area-ish/1000 |
| Powered flag | 1 | |
| Local pellets | 4 | pellet/super in adjacent cell (UP/DOWN/L/R) |
| Local danger | 4 | hunting ghost adjacent **or** 1-step away through that exit; suppressed while powered |
| Deltas vs previous step | 3 | nearest-pellet / ghost / power-pellet distance change (negative = moved closer) |
| Steps since pellet | 1 | capped at 100 |
| Last offsets | 2 | current − last position (y, x) / max_dim |
| Second-last offsets | 2 | |
| Region completion | 1 | fraction cleared in current 4×4 region |
| Region dirty flag | 1 | region has ≤2 pellets left |
| Just-died flag | 1 | decays −0.05/step after death |
| Same-action streak | 1 | capped at 20 |

A length assertion guards this list: changing it requires updating
`PLAYER_EXTRA_FEATURE_COUNT` and `EXTRA_FEATURE_COUNT`
(`AI_arena/data/constants.py`).

### 3. Action mask — `[1, 4]`

Boolean tensor over UP/DOWN/LEFT/RIGHT from `movement.can_move`. Training and
eval mask logits with `-1e4` before sampling/argmax.

## Policy: `PlayerActorCritic` (`AI_arena/models/cnn_player.py`)

```
grid [B,7,25,50] ─► CNN ─► flatten 10,400 ─┐
                                           ├─► proj Linear(10,465→128)+ReLU ─► GRU(128→128) ─► out Linear(128→128)+ReLU
features [B,65] ───────────────────────────┘                                                                    ├─► actor Linear(128→4)
                                                                                                                └─► critic Linear(128→1)
hidden: [1,B,128], passed between steps
```

CNN stack (`PacmanCNNBackbone`): Conv 7→32 → ResBlock(d=1) → ResBlock(d=2) →
Conv 32→64 dilation 4 (receptive field ≈ whole small maze) → ResBlock(d=4) →
stride-2 Conv 64→64 → 1×1 Conv to 32. Dropout is **0** for RL (0.1 only in
the imitation variant `PlayerImitationCNN`).

GRU memory rules:
- Single-step mode (rollout/inference): hidden carried across steps, reset to
  zero by the caller on episode boundaries.
- Sequence mode (PPO update): chunks of `SEQ_LEN=16`; if `dones` passed, the
  backbone zeroes hidden state *inside* the chunk at reset steps — memory
  never leaks across episodes.

Checkpoint utilities:
- `load_checkpoint_into_policy(policy, path)` → bool. Maps SL key
  `action_head*` → `actor*`, copies shape-matched tensors, and tolerates
  feature-count changes for `backbone.proj.0.weight` by copying the overlap
  slice (new columns stay at init).
- `load_sl_weights_into_ppo(...)` — warm-start helper used by training's
  `SL_WARMSTART` mode.
