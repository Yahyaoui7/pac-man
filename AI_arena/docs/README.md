# RL System Documentation

Last updated: **2026-08-23** — covers the PPO player pipeline as of the
fixed-seed evaluation + danger-telemetry changes.

> Rule of thumb: **when you change behavior, update the matching doc in the
> same commit.** These files exist so we can tell "the model changed because
> we changed something" apart from noise.

## File index

| Doc | Covers |
|-----|--------|
| [01-training-loop.md](01-training-loop.md) | `AI_arena/player/player_training.py` — config, rollouts, GAE, BPTT-PPO, logging, checkpoints, eval & stall detection |
| [02-environment.md](02-environment.md) | `AI_arena/player/player_env.py` — episode lifecycle, step pipeline, observations, telemetry, seeding |
| [03-rewards.md](03-rewards.md) | `AI_arena/player/rewards.py` + `constants.py` — every reward term, active vs disabled |
| [04-model-and-observation.md](04-model-and-observation.md) | `AI_arena/models/` + `player/data/observation.py` — CNN+GRU actor-critic and the 65-feature vector |
| [05-evaluation-and-telemetry.md](05-evaluation-and-telemetry.md) | `utils/evaluate.py`, `utils/metrics.py`, survival metrics, stall detection, determinism |
| [06-reading-losses-and-learning.md](06-reading-losses-and-learning.md) | How to read policy/value loss, entropy & LR — deciding if the model is learning |

Superseded docs: `player_rl_code_explanation.md` (repo root) describes an old
12-channel / 37-feature iteration and outdated filenames. Trust these docs
instead; that file is kept only for history.

## System at a glance

```
                    ┌──────────────────────────────────────────────┐
                    │ player_training.py  (train())                │
                    │                                              │
   obs ────────────►│ policy(grid, features, hidden) → action      │
                    │    ▲                                         │
                    │    │ rollout 3000 steps                      │
                    │    │                                         │
                    │  GAE → BPTT-PPO update (seq chunks of 16)    │
                    │    │                                         │
                    │    ├── every 50 upd: fixed-seed EVAL         │
                    │    │      → eval_history.json + best ckpt    │
                    │    └── stall detection → STALL WARNING       │
                    └───────────────┬──────────────────────────────┘
                                    │ step(action) / reset()
                    ┌───────────────▼──────────────────────────────┐
                    │ PacmanPlayerEnv                              │
                    │  physics ticks → events → RewardCalculator   │
                    │                        → danger telemetry    │
                    │  ghosts driven by GhostController (BFS hunt) │
                    └──────────────────────────────────────────────┘
```

Components:

| Component | File | Role |
|---|---|---|
| Environment | `AI_arena/player/player_env.py` | Headless game: maze gen, physics, pellets, collisions, rewards, telemetry |
| Rewards | `AI_arena/player/rewards.py` | Pure reward computation per step (`RewardCalculator`) |
| Ghost AI | `AI_arena/player/ghost_controller.py` | BFS hunting w/ 30% confusion, frightened flee via BFS, prison respawn |
| Policy | `AI_arena/models/cnn_player.py` | `PlayerActorCritic` = CNN backbone + GRU + actor/critic heads |
| Backbone | `AI_arena/models/cnn_backbone.py` | ResNet-style CNN encoder + GRU(128), done-masked hidden resets |
| Observation | `AI_arena/player/data/observation.py` + `AI_arena/data/formatter.py` | Grid `[1,7,25,50]` + feature vector `[1,65]` + action mask `[1,4]` |
| Training | `AI_arena/player/player_training.py` | Recurrent PPO loop |
| Eval harness | `AI_arena/player/utils/evaluate.py` | Fixed-seed greedy benchmark → `AI_arena/evals/eval_history.json` |
| Metrics/formatting | `AI_arena/player/utils/metrics.py` | Breakdown lines, `compute_survival_stats`, `format_survival_line` |
| Logger / quit | `AI_arena/player/utils/logger.py` | PID-locked file+stdout logger, `'q'` graceful stop |

## Reading a training log line

```
Upd 042/1000 | Tot Ep: 655 | Avg Epoch Rwd: -812.3 | Max Pellets: 141 (78.3%) | Avg Pellets: 88.2 (51.0%) | Osc%: 3.1% | Avg Rwd: -804.5 | Step: ... | Death: ... | ... | Loss (P/V): -0.0123/0.4567 | Time: 1550.1s (36.9s/upd) | Complete:  2.3% | Truncated: 11.6% | Avg Maze: 312.4 (21.5x14.5)
   SURV | Death:  97.7% | Corn:  0.9/ep ( 1.4%) | Esc: 55% [ 29] | CDth: 18% | MinD:  6.41
```

| Field | Meaning |
|---|---|
| `Avg Epoch Rwd` | Mean episode reward over episodes finished *in this update's window* (since last checkpoint save) |
| `Max/Avg Pellets (x%)` | Pellet count and % of total; Max = window best, Avg = rolling last-100-episodes |
| `Osc%` | Share of oscillating steps per episode (A→B→A flips / 4-cells loops) |
| `Ent:` | Mean policy entropy (nats) during the update — 1.386 = random, →0 deterministic. See [06](06-reading-losses-and-learning.md) |
| Breakdown fields (`Step:` `Pellet:` ...) | Window sum per reward component. **Sparse:** zero components are omitted, so only influential terms appear — safe to add/remove rewards without log churn |
| `Complete:` / `Truncated:` | Level-completion / timeout rate within the update window |
| `Avg Maze` | Average maze area sampled by the curriculum (explains reward swings) |
| SURV line | Danger-skill leading indicators — see [05-evaluation-and-telemetry.md](05-evaluation-and-telemetry.md). `Life:` = moves survived + % of the episode step budget (`maze_area × 12`) consumed, so it stays comparable across maze sizes. With death at 100%, Life **is** the survival metric |

Every 50 updates an `EVAL @...:` line appears: the same policy scored greedily
on 20 identical benchmark mazes. That number — not the training-window stats —
is the ground truth for "is this run learning?".
