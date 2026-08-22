# Training Report 003 — PPO Stage-1

Generated: 2026-08-21 21:12  
Log file: `training_log.txt`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | 1 → 84 (84 logged) |
| Total Episodes | 570 |
| Rollout Steps / Update | 1024 |
| PPO Epochs | 4 |
| Mini-batch Size | 64 |
| Learning Rate | 1e-4 |
| Gamma (γ) | 0.99 |
| GAE Lambda (λ) | 0.95 |
| Clip ε | 0.2 |
| Entropy Coef | 0.05 |
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

## Reward System (as implemented in `player_env.py`)

| Event | Reward |
|-------|--------|
| Every step (base penalty) | **−0.2** |
| Oscillating move (A→B→A) | **−0.5** |
| Pellet eaten | **+5.0** |
| Super-pellet eaten | **+10.0** |
| Ghost eaten (in powered mode) | **+30.0** |
| Level completed | **+200.0** + remaining_steps bonus |
| Pac-Man died | **−30.0** |
| BFS shaping (distance to nearest pellet) | **0.3 × Δpotential** |

---

## Results Summary

| Metric | Value | At Update |
|--------|-------|-----------|
| Best raw avg reward | 147.0 | 4 |
| Best smoothed avg reward | 84.2 | 17 |
| Best avg pellet % | 44.6% | 57 |
| Best max pellet % | 99.5% | 16 |
| Final avg reward | 86.0 | 84 |
| Final avg pellet % | 35.1% | 84 |
| Final max pellet % | 63.1% | 84 |

---

## Reward Trend Diagnostics

Reward-vs-pellet correlation (smoothed): **0.13**
*(Log has no per-epoch reward field — add `Averge Epoch Rwd` to the logger to unlock instantaneous-vs-smoothed volatility diagnostics.)*
*(Log has no `Avg Maze Area` field yet — add it to check whether random maze-size variance is driving the oscillation.)*
Wide-window residual ratio: **1.34**

### Reward Breakdown Summary

| Component | Avg per Episode |
|-----------|----------------|
| Largest positive | Pellet (+2608.1) |
| Largest penalty | Oscillation (-1054.4) |

Component correlations with total reward:

| Component | Correlation |
|-----------|-------------|
| Complete | nan |
| BFS | -0.52 |
| Super | 0.43 |
| Ghost | 0.41 |
| Osc | 0.23 |
| Death | -0.19 |
| Pellet | 0.12 |
| Step | 0.07 |


Time spent in each trend regime (by update count):

| Regime | % of run |
|--------|----------|
| Improving | 20% |
| Plateau | 19% |
| Declining | 25% |

### Segments

| Updates | Regime | Avg slope (Δreward/upd) |
|---------|--------|---------------------------|
| 4–24 | declining | -0.7422 |
| 25–40 | plateau | -0.0215 |
| 41–57 | improving | +0.6450 |

### What this run suggests

- Reward is still trending up for a meaningful share of the run (~20% improving). Worth letting this configuration keep training before changing anything.
- Reward and pellet completion are weakly correlated (corr=0.13). Reward is moving for reasons other than pellet progress — check how much of it comes from the BFS shaping, oscillation penalty, or per-step cost; the agent may be optimizing those instead.
- The oscillation mostly survives a much wider smoothing window (~134% of the standard-smoothed curve's variance remains). That argues against pure window-sampling noise — this looks like a real, slower-cycle pattern in training itself (e.g. an LR/entropy interaction, or periodic instability), not just averaging artifacts.
- **Reward breakdown:** the largest positive driver is **Pellet** (avg +2608.1/ep). The largest penalty is **Oscillation** (avg -1054.4/ep).
- Pellet reward (corr=0.12) dominates over BFS shaping (corr=-0.52) — the shaping term is well-calibrated.

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


| `08_smoothing_window_check.png` | Standard vs. wide smoothing — tests whether the oscillation is window noise |
| `09_reward_breakdown.png` | Per-component reward contribution over time |
| `10_positive_composition.png` | Stacked positive rewards (pellet, super, ghost, complete, BFS) |
| `11_penalty_composition.png` | Stacked penalties (step, oscillation, death) |
| `12_component_importance.png` | Each component as % of total |reward| |


![Overview](00_overview.png)

---

## Notes / Observations

<!-- Add manual notes about this run here -->
- Training data covers updates 1–84 (570 episodes).
- Log window size: last 100 completed episodes per update (smoothed breakdown).
