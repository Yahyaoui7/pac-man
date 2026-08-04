# Training Report 005 — PPO Stage-1

Generated: 2026-08-04 07:54  
Log file: `RL_logs.txt`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | 1 → 1188 (1188 logged) |
| Total Episodes | 4271 |
| Rollout Steps / Update | 512 |
| PPO Epochs | 4 |
| Mini-batch Size | 64 |
| Learning Rate | 3e-4 |
| Gamma (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| Clip ε | 0.2 |
| Entropy Coef | 0.02 |
| Value Coef | 0.5 |
| Max Grad Norm | 0.5 |

---

## Observation Format

Each step the model receives **3 tensors**:

### 1. Grid (`[1, C, H, W]` CNN input)
| Channel | Content |
|---------|---------|
| 0 | Walls (maze structure) |
| 1 | Pellets (normal, value 1) |
| 2 | Super-pellets (value 2) |
| 3 | Player position |
| 4 | Ghosts (all combined) |
| 5 | BFS distance-to-player heatmap (normalized) |

### 2. Extra Features (`[1, F]` MLP input)
| Feature | Description |
|---------|-------------|
| player_grid_x | Normalized column position |
| player_grid_y | Normalized row position |
| remaining_pellets | Fraction of pellets left |
| ghost_count | Number of active ghosts |
| ghost_min_dist | Closest ghost BFS distance (normalized) |
| powered_mode | 1 if player is in powered/attack mode |

### 3. Valid Actions (`[1, 4]` mask)
Binary mask over `[UP, DOWN, LEFT, RIGHT]` — invalid moves are masked to −∞ before softmax.

---

## Reward System

| Event | Reward |
|-------|--------|
| Every step (base penalty) | **−0.2** |
| Oscillating move (reversed within 6 steps) | **−0.3** *(active from report_002)* |
| First visit to a new grid tile | **+0.5** |
| Pellet eaten | **+5.0** |
| Super-pellet eaten | **+15.0** |
| Ghost eaten (in powered mode) | **+30.0** |
| Level completed | **+100.0** |
| Pac-Man died | **−20.0** |

**Net examples:**
- Step forward into a new pellet tile: `−0.2 + 0.5 + 5.0 = +5.3`
- Step forward into a new empty tile: `−0.2 + 0.5 = +0.3`
- Step forward into an already-visited tile: `−0.2`
- Oscillating move (back-track): `−0.2 − 0.3 = −0.5` *(active from report_002)*

---

## Results Summary

| Metric | Value | At Update |
|--------|-------|-----------|
| Best raw avg reward | 8.9 | 2 |
| Best smoothed avg reward | -4.9 | 94 |
| Best avg pellet % | 74.5% | 2 |
| Best max pellet % | 100.0% | 22 |
| Final avg reward | -12.9 | 1188 |
| Final avg pellet % | 68.2% | 1188 |
| Final max pellet % | 85.1% | 1188 |

---

## Reward Trend Diagnostics

Reward-vs-pellet correlation (smoothed): **0.92**
Episode-to-window reward volatility (std): **29.3**

Time spent in each trend regime (by update count):

| Regime | % of run |
|--------|----------|
| Improving | 21% |
| Plateau | 35% |
| Declining | 38% |

### Segments

| Updates | Regime | Avg slope (Δreward/upd) |
|---------|--------|---------------------------|
| 8–35 | plateau | +0.0007 |
| 36–53 | declining | -0.0234 |
| 69–89 | improving | +0.0274 |
| 105–204 | declining | -0.0548 |
| 209–267 | declining | -0.0352 |
| 268–285 | plateau | -0.0001 |
| 286–345 | improving | +0.0609 |
| 357–388 | declining | -0.0254 |
| 391–467 | declining | -0.0470 |
| 468–486 | plateau | -0.0001 |
| 487–608 | improving | +0.0560 |
| 624–680 | declining | -0.0250 |
| 681–696 | plateau | +0.0007 |
| 697–717 | improving | +0.0266 |
| 718–888 | plateau | -0.0027 |
| 889–932 | declining | -0.0441 |
| 933–965 | plateau | -0.0054 |
| 966–1035 | declining | -0.0375 |
| 1036–1063 | plateau | +0.0040 |
| 1069–1129 | plateau | +0.0078 |
| 1130–1149 | improving | +0.0218 |
| 1150–1188 | plateau | +0.0054 |

### What this run suggests

- Reward is declining across ~38% of the run. That usually means an unstable update (LR too high, value loss diverging, or a shaping term that's easy to exploit in a way that eventually collapses behavior) rather than something more training will fix.
- Reward and pellet completion move together closely (corr=0.92). Reward increases are coming from actually eating more pellets — the shaping is aligned with the real objective.
- Per-episode reward is noisy relative to the smoothed trend (episode-to-window std ≈ 29.3). Individual episodes still swing a lot — consider whether the maze size/seed randomization is creating too much variance per update, or whether more PPO epochs/rollout steps would help it settle.

---

## Plots

| File | Description |
|------|-------------|
| `00_overview.png` | Combined 3-panel summary (reward / pellets / value loss) |
| `01_avg_reward.png` | Average episode reward, shaded by trend regime |
| `02_pellet_completion.png` | Pellet completion % (avg & max) |
| `03_value_loss.png` | Value network loss |
| `04_reward_trend.png` | Local reward slope — where training is/isn't progressing |
| `05_reward_vs_pellets.png` | Reward vs pellet completion, dual-axis |
| `06_epoch_vs_window_reward.png` | Instantaneous epoch reward vs the smoothed sliding-window average |

![Overview](00_overview.png)

---

## Notes / Observations

<!-- Add manual notes about this run here -->
- Training data covers updates 1–1188 (4271 episodes).
- Log window size: last 20 completed episodes per update.
- Oscillation penalty introduced in this run to combat node-to-node back-and-forth behavior.
