# Training Report 010 — PPO Stage-1

Generated: 2026-08-06 14:21  
Log file: `training_log.txt`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | 1 → 684 (684 logged) |
| Total Episodes | 1408 |
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
| Best raw avg reward | 141.2 | 660 |
| Best smoothed avg reward | 134.7 | 665 |
| Best avg pellet % | 89.1% | 2 |
| Best max pellet % | 100.0% | 6 |
| Final avg reward | 128.3 | 684 |
| Final avg pellet % | 83.7% | 684 |
| Final max pellet % | 90.1% | 684 |

---

## Reward Trend Diagnostics

Reward-vs-pellet correlation (smoothed): **0.59**
Episode-to-window reward volatility (std): **85.4**
Reward-vs-maze-size correlation (smoothed): **0.03**
Wide-window residual ratio: **0.62**

### Reward Breakdown Summary

| Component | Avg per Episode |
|-----------|----------------|
| Largest positive | Pellet (+151.8) |
| Largest penalty | Oscillation (-113.0) |

Component correlations with total reward:

| Component | Correlation |
|-----------|-------------|
| Pellet | 0.34 |
| Super | -0.21 |
| Osc | 0.06 |
| Step | 0.02 |
| Ghost | nan |
| Complete | 0.28 |
| Death | nan |
| BFS | -0.27 |


Time spent in each trend regime (by update count):

| Regime | % of run |
|--------|----------|
| Improving | 35% |
| Plateau | 33% |
| Declining | 22% |

### Segments

| Updates | Regime | Avg slope (Δreward/upd) |
|---------|--------|---------------------------|
| 9–46 | plateau | +0.1474 |
| 47–93 | improving | +0.3352 |
| 109–142 | declining | -0.3348 |
| 143–158 | plateau | +0.0075 |
| 159–197 | improving | +0.2983 |
| 198–258 | plateau | +0.0445 |
| 259–285 | improving | +0.1962 |
| 286–324 | plateau | +0.0336 |
| 338–385 | plateau | -0.0142 |
| 386–422 | declining | -0.4318 |
| 431–468 | improving | +0.5729 |
| 477–556 | declining | -0.3095 |
| 571–656 | improving | +0.4204 |
| 657–683 | plateau | -0.0168 |

### What this run suggests

- Reward is still trending up for a meaningful share of the run (~35% improving). Worth letting this configuration keep training before changing anything.
- Reward and pellet completion are moderately correlated (corr=0.59).
- Per-episode reward is noisy relative to the smoothed trend (episode-to-window std ≈ 85.4). Individual episodes still swing a lot — consider whether the maze size/seed randomization is creating too much variance per update, or whether more PPO epochs/rollout steps would help it settle.
- The oscillation mostly survives a much wider smoothing window (~62% of the standard-smoothed curve's variance remains). That argues against pure window-sampling noise — this looks like a real, slower-cycle pattern in training itself (e.g. an LR/entropy interaction, or periodic instability), not just averaging artifacts.
- Reward doesn't correlate much with average maze size in the window (corr=0.03) — maze-mix doesn't look like the driver of the oscillation here, so it's more likely coming from training dynamics.
- **Reward breakdown:** the largest positive driver is **Pellet** (avg +151.8/ep). The largest penalty is **Oscillation** (avg -113.0/ep).
- Pellet reward (corr=0.34) dominates over BFS shaping (corr=-0.27) — the shaping term is well-calibrated.

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
| `07_reward_vs_maze_size.png` | Reward vs average maze size in the window |
| `08_smoothing_window_check.png` | Standard vs. wide smoothing — tests whether the oscillation is window noise |
| `09_reward_breakdown.png` | Per-component reward contribution over time |
| `10_positive_composition.png` | Stacked positive rewards (pellet, super, ghost, complete, BFS) |
| `11_penalty_composition.png` | Stacked penalties (step, oscillation, death) |
| `12_component_importance.png` | Each component as % of total |reward| |


![Overview](00_overview.png)

---

## Notes / Observations

<!-- Add manual notes about this run here -->
- Training data covers updates 1–684 (1408 episodes).
- Log window size: last 100 completed episodes per update (smoothed breakdown).
