# Training Report 008 — PPO Stage-1

Generated: 2026-08-05 15:49  
Log file: `training_log.txt`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | 1 → 2174 (2174 logged) |
| Total Episodes | 6341 |
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
| Best raw avg reward | 582.8 | 1572 |
| Best smoothed avg reward | 545.8 | 1578 |
| Best avg pellet % | 98.0% | 1040 |
| Best max pellet % | 100.0% | 1 |
| Final avg reward | 228.9 | 2174 |
| Final avg pellet % | 84.8% | 2174 |
| Final max pellet % | 93.6% | 2174 |

---

## Reward Trend Diagnostics

Reward-vs-pellet correlation (smoothed): **0.95**
Episode-to-window reward volatility (std): **186.7**
Reward-vs-maze-size correlation (smoothed): **-0.02**
Wide-window residual ratio: **0.51**

### Reward Breakdown Summary

| Component | Avg per Episode |
|-----------|----------------|
| Largest positive | Complete (+209.1) |
| Largest penalty | Step (-74.4) |

Component correlations with total reward:

| Component | Correlation |
|-----------|-------------|
| Pellet | 0.78 |
| Super | 0.73 |
| Step | 0.69 |
| Osc | 0.64 |
| Ghost | nan |
| Complete | 0.94 |
| Death | nan |
| BFS | 0.63 |


Time spent in each trend regime (by update count):

| Regime | % of run |
|--------|----------|
| Improving | 26% |
| Plateau | 39% |
| Declining | 33% |

### Segments

| Updates | Regime | Avg slope (Δreward/upd) |
|---------|--------|---------------------------|
| 1–30 | declining | -0.8605 |
| 31–134 | plateau | -0.0959 |
| 135–225 | declining | -0.6923 |
| 226–339 | plateau | +0.0798 |
| 340–392 | improving | +0.4728 |
| 393–470 | plateau | +0.1765 |
| 479–572 | plateau | +0.0346 |
| 573–686 | improving | +1.4350 |
| 699–803 | declining | -1.6634 |
| 804–891 | plateau | -0.1421 |
| 895–947 | plateau | -0.0810 |
| 948–1083 | improving | +1.9109 |
| 1084–1119 | plateau | +0.1000 |
| 1120–1305 | declining | -1.4804 |
| 1322–1586 | improving | +1.0614 |
| 1587–1613 | plateau | -0.0775 |
| 1614–1660 | declining | -0.4164 |
| 1661–1814 | plateau | -0.0320 |
| 1815–1882 | declining | -0.7858 |
| 1883–1914 | plateau | -0.2231 |
| 1915–2112 | declining | -0.9915 |
| 2113–2174 | plateau | -0.0444 |

### What this run suggests

- Reward is declining across ~33% of the run. That usually means an unstable update (LR too high, value loss diverging, or a shaping term that's easy to exploit in a way that eventually collapses behavior) rather than something more training will fix.
- Reward and pellet completion move together closely (corr=0.95). Reward increases are coming from actually eating more pellets — the shaping is aligned with the real objective.
- Per-episode reward is noisy relative to the smoothed trend (episode-to-window std ≈ 186.7). Individual episodes still swing a lot — consider whether the maze size/seed randomization is creating too much variance per update, or whether more PPO epochs/rollout steps would help it settle.
- The oscillation mostly survives a much wider smoothing window (~51% of the standard-smoothed curve's variance remains). That argues against pure window-sampling noise — this looks like a real, slower-cycle pattern in training itself (e.g. an LR/entropy interaction, or periodic instability), not just averaging artifacts.
- Reward doesn't correlate much with average maze size in the window (corr=-0.02) — maze-mix doesn't look like the driver of the oscillation here, so it's more likely coming from training dynamics.
- **Reward breakdown:** the largest positive driver is **Complete** (avg +209.1/ep). The largest penalty is **Step** (avg -74.4/ep).
- Pellet reward (corr=0.78) dominates over BFS shaping (corr=0.63) — the shaping term is well-calibrated.

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
- Training data covers updates 1–2174 (6341 episodes).
- Log window size: last 100 completed episodes per update (smoothed breakdown).
