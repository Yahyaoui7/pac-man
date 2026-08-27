# Player RL — Pac-Man Agent Training

Reinforcement Learning pipeline that trains **Pac-Man** to survive and clear
levels against 4 BFS-hunting ghosts, using **Recurrent PPO** (CNN + GRU
actor-critic) with hard action masking.

> Detailed docs live in [`../docs/`](../docs/) (training loop, environment,
> rewards, model, evaluation, loss reading). This README is the quick-start
> and summary.

---

## 1. Quick start

```bash
# from repo root
uv sync                                          # install deps
uv run python -m AI_arena.player.player_training  # start/resume training
```

- Training auto-resumes from the latest checkpoint (`RESUME=True`).
- Press `q` or Ctrl-C for a graceful stop (checkpoint is saved).
- A PID lock (`training_log.lock`) prevents two training sessions at once.

### Other commands

```bash
# Benchmark a checkpoint on 20 fixed-seed mazes (+ history table)
uv run python -m AI_arena.player.utils.evaluate --compare

# Options: --checkpoint PATH --episodes N --stage 2 --device cpu --sample --no-save

# Plot reward/pellet curves from the eval history
uv run python -m AI_arena.player.utils.plot_training_curves

# Watch a trained checkpoint play the real game
uv run python -m AI_arena.player.utils.play_player_ai
```

---

## 2. Architecture at a glance

```
                    ┌──────────────────────────────────────────────┐
                    │ player_training.py  (train())                │
                    │                                              │
   obs ────────────►│ policy(grid, features, hidden) → action      │
                    │    ▲                                         │
                    │    │ rollout 3000 steps                      │
                    │  GAE → BPTT-PPO update (seq chunks of 16)    │
                    │    ├── every 50 upd: fixed-seed EVAL         │
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

| Component | File | Role |
|---|---|---|
| Environment | `player_env.py` | Headless game: maze gen, physics, pellets, collisions, telemetry |
| Rewards | `rewards.py` + `constants.py` | Pure per-step reward computation |
| Ghost AI | `ghost_controller.py` | BFS hunting w/ 30% confusion; frightened flee; prison respawn |
| Policy | `../models/cnn_player.py` | CNN backbone → GRU(128) → actor(4) / critic(1) heads |
| Observation | `data/observation.py` | Grid `[1,7,25,50]` + features `[1,65]` + action mask `[1,4]` |
| Trainer | `player_training.py` | Recurrent PPO loop |
| Eval harness | `utils/evaluate.py` | Fixed-seed greedy benchmark → `../evals/eval_history.json` |

---

## 3. Observation & model

Per step the policy receives three tensors:

1. **Grid `[1, 7, 25, 50]`** — channels: walkable cells, normal pellets,
   power pellets, player position, hunting ghosts, edible ghosts, visit heatmap.
2. **Feature vector `[1, 65]`** — player/ghost directions, edible flags,
   frightened timers, valid actions, BFS distances to ghosts/pellets/power,
   local danger, distance deltas vs previous step, region progress, ...
3. **Action mask `[1, 4]`** — `[UP, DOWN, LEFT, RIGHT]`; blocked moves get
   logit `-1e8` so the policy can never walk into a wall.

Model (`PlayerActorCritic`):

```
grid [B,7,25,50] ─► ResNet-style CNN ─► flatten ─┐
                                                 ├─► proj(→128) ─► GRU(128) ─► out(→128)
features [B,65] ─────────────────────────────────┘                                ├─► actor → 4 logits
                                                                                  └─► critic → V(s)
```

The GRU gives the agent memory across steps; hidden state resets on episode
end or death.

---

## 4. Training loop (per update)

```text
1. Rollout      collect 3000 env steps (masked categorical sampling,
                plus ε=0.06 uniform exploration during rollouts)
2. GAE          advantages with γ=0.99, λ=0.95
                (time-limit truncations bootstrap instead of counting as death)
3. Reshape      cut into 187 sequence chunks of SEQ_LEN=16 (BPTT for the GRU)
4. PPO update   2 epochs over random minibatches of 16 chunks
                loss = policy_clip + 0.25·value − ent_coef·entropy
5. Log          one Upd-line + one SURV danger-telemetry line
6. Every 50 upd save checkpoint + run fixed-seed EVAL + stall detection
```

Key config (top of `player_training.py`, edit by hand):

| Constant | Value | Meaning |
|---|---|---|
| `STAGE` | 2 | 1 = ghosts jailed/no collisions; 2 = full game |
| `NUM_UPDATES` | 1000 | PPO updates per run |
| `ROLLOUT_STEPS` | 3000 | env steps collected per update |
| `LEARNING_RATE` | 2e-4 | Adam |
| `ENTROPY_COEF` | 0.015 | exploration bonus (anti-collapse guard) |
| `ROLLOUT_EPSILON` | 0.06 | ε-uniform exploration during data collection |
| `START_PELLETS` | (1, 2, 3) | pellet curriculum; `None` = full map |
| `USE_BFS_SHAPING` | True | potential-based shaping toward nearest pellet |
| `SAVE_INTERVAL` | 50 | checkpoint + eval cadence |
| `SEED` | 42 | training env seed |

~37 s/update on CPU.

---

## 5. Reward system ("Minimal Signal Mode")

Goal: **stay alive + eat pellets + zero oscillation**, cleanest possible signal.

| Term | Trigger | Value |
|---|---|---|
| `death` | Pac-Man dies | −350 |
| `complete` | Level cleared | +1000 + time bonus |
| `pellet` | Normal pellet eaten | +1.5 + 2.0 × cleared-fraction (+4 once >75%) |
| `super_pellet` | Power pellet | +5 |
| `milestone` | 50/75/85/95% pellets | +20 / +50 / +100 / +200 |
| `ghost` | Eat an edible ghost | +150 |
| `exploration` | First visit of a tile this episode | +1 per new tile |
| `bfs` | Shaping: γΦ(s′)−Φ(s), Φ = −BFS dist to nearest pellet | ~±3/cell |
| `zone_stagnation` | Camping 12+ steps in one 3×3 block | −5 then −0.5/step |
| `oscillation` | A→B→A wiggle without eating | −10 → −20 → −30 escalating |
| `ghost_proximity` | Near hunting ghosts / moving closer | d=1: −4 … d=5: −0.05 |

All numeric values live in `constants.py`. Disabled shapers (predictive
threat, evasion skill, dense survival, ...) remain implemented in `rewards.py`
and can be re-enabled by uncommenting their calls in `calculate()`.

---

## 6. Curriculum & graduation

Two ladders:

1. **Stage**: `STAGE=1` (learn to eat safely) → `STAGE=2` (full game).
2. **Pellet count**: `(1,2,3)` → `(4,6,10)` → `(15,25)` → `None` (full map).

Graduate when **all** hold for ≥30 updates / ≥2 evals:

```text
Death% < 50 · pellets ≥ 60% of stage max · Complete% ≥ 15 · Osc% ≤ 20
```

Change ONE variable per step. If stuck >150 updates:
- Ent < 0.2 → temporarily double `ENTROPY_COEF`
- Death >90% sustained → enable only `_ghost_proximity_penalty`
- Still stuck → drop back one stage.

---

## 7. Is it learning? (evidence hierarchy)

Trust in this order:

1. **`EVAL @NNN:` score** — greedy policy on 20 identical seeds
   (score = pellet% + 40×completion − 30×death). The ground truth.
2. **SURV telemetry** — leading indicators of survival skill:
   `Esc↑ Corn↓ CDth↓ MinD↑`.
3. Rolling train averages — noisy.
4. Losses/entropy — diagnose *why*, not *whether*.

Example log lines:

```
Upd 042/1000 | Avg Pellets: 88.2 (51.0%) | Osc%: 3.1% | Ent: 0.62 | Loss (P/V): -0.012/0.456 | Complete: 2.3% | Truncated: 11.6%
   SURV | Death: 97.7% | Corn: 0.9/ep (1.4%) | Esc: 55% [29] | CDth: 18% | MinD: 6.41
EVAL @100: score 15.6 | pellet 45.6% | death 100% | esc 67% [9] | MinD 9.42
```

Healthy signs: P-loss ≈ ±0.01, V-loss stable plateau (ours: 27–55),
entropy drifting slowly from ln(4)≈1.386 toward 0.2–0.6 while eval rises.

A `STALL WARNING` fires if no meaningful eval improvement (≥2.0) happens for
250 updates — flat score AND flat SURV stats = kill/change something.

---

## 8. Checkpoints & artifacts

| Path | Content |
|---|---|
| `../models/player_rl_stage{STAGE}.pt` | latest checkpoint |
| `../models/player_rl_stage{STAGE}_best.pt` | best-by-eval-score checkpoint |
| `../evals/eval_history.json` | full eval records incl. survival stats |
| `training_log.txt` | append-only training log (PID-locked) |

Comparability rule: eval scores are only comparable within identical game
parameters — changing ghost confusion/rewards starts a new baseline.

---

## 9. File map

```text
AI_arena/player/
├── README.md                  ← you are here
├── constants.py               all reward/env constants
├── entity_factory.py          spawns player + ghosts
├── ghost_controller.py        BFS hunt / flee / prison logic
├── player_env.py              headless environment (reset/step)
├── player_training.py         recurrent PPO trainer (entry point)
├── rewards.py                 RewardCalculator
├── player_controller.py       live inference for the real game
├── data/
│   ├── observation.py         grid + feature vector builder
│   └── expert.py, ...         imitation-learning helpers
└── utils/
    ├── evaluate.py            fixed-seed benchmark CLI
    ├── metrics.py             survival stats + formatting
    ├── logger.py              PID-locked file logger, 'q' to quit
    ├── play_player_ai.py      watch the checkpoint play
    └── plot_training_curves.py
```




a pool of mazes that the model plays u until he wins when he wins a maze we replace that maze  20
