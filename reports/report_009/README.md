# Training Report 009 — PPO Stage-1

Generated: 2026-08-06 07:10  
Log file: `training_log.txt`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | 1 → 1000 (1000 logged) |
| Total Episodes | 3212 |
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
| Best raw avg reward | 620.9 | 257 |
| Best smoothed avg reward | 591.1 | 447 |
| Best avg pellet % | 98.5% | 530 |
| Best max pellet % | 100.0% | 2 |
| Final avg reward | 552.3 | 1000 |
| Final avg pellet % | 97.4% | 1000 |
| Final max pellet % | 100.0% | 1000 |

---

## Reward Trend Diagnostics

Reward-vs-pellet correlation (smoothed): **0.67**
Episode-to-window reward volatility (std): **198.2**
Reward-vs-maze-size correlation (smoothed): **-0.07**
Wide-window residual ratio: **0.98**

### Reward Breakdown Summary

| Component | Avg per Episode |
|-----------|----------------|
| Largest positive | Complete (+361.6) |
| Largest penalty | Step (-67.8) |

Component correlations with total reward:

| Component | Correlation |
|-----------|-------------|
| Osc | 0.37 |
| Step | 0.35 |
| Pellet | 0.11 |
| Super | 0.05 |
| Ghost | nan |
| Complete | 0.58 |
| Death | nan |
| BFS | -0.31 |


Time spent in each trend regime (by update count):

| Regime | % of run |
|--------|----------|
| Improving | 34% |
| Plateau | 26% |
| Declining | 24% |

### Segments

| Updates | Regime | Avg slope (Δreward/upd) |
|---------|--------|---------------------------|
| 1–39 | improving | +4.2213 |
| 43–75 | improving | +1.3190 |
| 85–116 | declining | -1.3600 |
| 132–165 | improving | +1.1597 |
| 166–229 | plateau | -0.1821 |
| 230–254 | improving | +1.1731 |
| 264–294 | declining | -1.5447 |
| 305–337 | improving | +1.2011 |
| 338–354 | plateau | -0.0354 |
| 355–376 | declining | -0.9614 |
| 377–393 | plateau | +0.0312 |
| 394–434 | improving | +0.9509 |
| 450–479 | declining | -1.2261 |
| 493–519 | improving | +0.9427 |
| 520–537 | plateau | +0.0139 |
| 538–575 | declining | -1.3739 |
| 586–614 | improving | +1.3772 |
| 625–651 | declining | -1.1390 |
| 692–717 | declining | -1.4041 |
| 725–757 | improving | +1.8045 |
| 765–797 | declining | -1.5286 |
| 809–833 | improving | +1.0355 |
| 834–881 | plateau | +0.1825 |
| 882–900 | improving | +0.7397 |
| 901–991 | plateau | -0.0661 |

### What this run suggests

- Reward is still trending up for a meaningful share of the run (~34% improving). Worth letting this configuration keep training before changing anything.
- Reward and pellet completion move together closely (corr=0.67). Reward increases are coming from actually eating more pellets — the shaping is aligned with the real objective.
- Per-episode reward is noisy relative to the smoothed trend (episode-to-window std ≈ 198.2). Individual episodes still swing a lot — consider whether the maze size/seed randomization is creating too much variance per update, or whether more PPO epochs/rollout steps would help it settle.
- The oscillation mostly survives a much wider smoothing window (~98% of the standard-smoothed curve's variance remains). That argues against pure window-sampling noise — this looks like a real, slower-cycle pattern in training itself (e.g. an LR/entropy interaction, or periodic instability), not just averaging artifacts.
- Reward doesn't correlate much with average maze size in the window (corr=-0.07) — maze-mix doesn't look like the driver of the oscillation here, so it's more likely coming from training dynamics.
- **Reward breakdown:** the largest positive driver is **Complete** (avg +361.6/ep). The largest penalty is **Step** (avg -67.8/ep).
- Pellet reward (corr=0.11) dominates over BFS shaping (corr=-0.31) — the shaping term is well-calibrated.

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
- Training data covers updates 1–1000 (3212 episodes).
- Log window size: last 100 completed episodes per update (smoothed breakdown).
