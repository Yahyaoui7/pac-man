# Training Report 002 — PPO Stage-1

Generated: 2026-08-21 18:39  
Log file: `training_log.txt`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | 1 → 39 (39 logged) |
| Total Episodes | 342 |
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
| Best raw avg reward | 140.4 | 24 |
| Best smoothed avg reward | 82.6 | 32 |
| Best avg pellet % | 38.1% | 2 |
| Best max pellet % | 89.5% | 14 |
| Final avg reward | 66.3 | 39 |
| Final avg pellet % | 33.8% | 39 |
| Final max pellet % | 77.8% | 39 |

---

## Reward Trend Diagnostics

Reward-vs-pellet correlation (smoothed): **0.79**
*(Log has no per-epoch reward field — add `Averge Epoch Rwd` to the logger to unlock instantaneous-vs-smoothed volatility diagnostics.)*
*(Log has no `Avg Maze Area` field yet — add it to check whether random maze-size variance is driving the oscillation.)*
Wide-window residual ratio: **0.47**

### Reward Breakdown Summary

| Component | Avg per Episode |
|-----------|----------------|
| Largest positive | Pellet (+4272.5) |
| Largest penalty | Oscillation (-2619.5) |

Component correlations with total reward:

| Component | Correlation |
|-----------|-------------|
| Ghost | -0.87 |
| Osc | 0.80 |
| Pellet | 0.75 |
| Step | 0.27 |
| Super | -0.01 |
| Complete | nan |
| BFS | 0.76 |
| Death | 0.66 |


Time spent in each trend regime (by update count):

| Regime | % of run |
|--------|----------|
| Improving | 54% |
| Plateau | 46% |
| Declining | 0% |

### Segments

| Updates | Regime | Avg slope (Δreward/upd) |
|---------|--------|---------------------------|
| 1–5 | plateau | +2.5918 |
| 6–13 | improving | +3.0967 |
| 14–26 | plateau | +2.9732 |
| 27–39 | improving | +3.5484 |

### What this run suggests

- Reward is still trending up for a meaningful share of the run (~54% improving). Worth letting this configuration keep training before changing anything.
- Reward and pellet completion move together closely (corr=0.79). Reward increases are coming from actually eating more pellets — the shaping is aligned with the real objective.
- The oscillation mostly survives a much wider smoothing window (~47% of the standard-smoothed curve's variance remains). That argues against pure window-sampling noise — this looks like a real, slower-cycle pattern in training itself (e.g. an LR/entropy interaction, or periodic instability), not just averaging artifacts.
- **Reward breakdown:** the largest positive driver is **Pellet** (avg +4272.5/ep). The largest penalty is **Oscillation** (avg -2619.5/ep).
- BFS shaping (corr=0.76) correlates with total reward more strongly than raw pellet reward (corr=0.75). The agent may be optimizing the distance heuristic instead of actual pellets — consider lowering `bfs_shaping_coef`.

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
- Training data covers updates 1–39 (342 episodes).
- Log window size: last 100 completed episodes per update (smoothed breakdown).
