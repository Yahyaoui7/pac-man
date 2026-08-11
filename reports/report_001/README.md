# Training Report 001 — PPO Stage-1

Generated: 2026-08-10 19:24  
Log file: `training_log.txt`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | 1 → 33 (33 logged) |
| Total Episodes | 65 |
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
| Best raw avg reward | 1433.3 | 16 |
| Best smoothed avg reward | 1216.3 | 28 |
| Best avg pellet % | 82.0% | 31 |
| Best max pellet % | 97.3% | 21 |
| Final avg reward | 1056.8 | 33 |
| Final avg pellet % | 81.7% | 33 |
| Final max pellet % | 87.3% | 33 |

---

## Reward Trend Diagnostics

Reward-vs-pellet correlation (smoothed): **0.94**
*(Log has no per-epoch reward field — add `Averge Epoch Rwd` to the logger to unlock instantaneous-vs-smoothed volatility diagnostics.)*
*(Log has no `Avg Maze Area` field yet — add it to check whether random maze-size variance is driving the oscillation.)*
Wide-window residual ratio: **0.05**

### Reward Breakdown Summary

| Component | Avg per Episode |
|-----------|----------------|
| Largest positive | Pellet (+1032.3) |
| Largest penalty | Step (-1715.2) |

Component correlations with total reward:

| Component | Correlation |
|-----------|-------------|
| Osc | 0.96 |
| Ghost | -0.91 |
| Super | 0.68 |
| Pellet | 0.49 |
| Complete | nan |
| Death | -0.97 |
| BFS | 0.44 |
| Step | -0.39 |


Time spent in each trend regime (by update count):

| Regime | % of run |
|--------|----------|
| Improving | 0% |
| Plateau | 79% |
| Declining | 0% |

### Segments

| Updates | Regime | Avg slope (Δreward/upd) |
|---------|--------|---------------------------|
| 8–33 | plateau | +30.8799 |

### What this run suggests

- Reward is flat for ~79% of the run. The policy has likely converged to whatever the current reward shape rewards most easily — more updates alone are unlikely to move it further; this is the point to change the reward system or curriculum rather than keep training.
- Reward and pellet completion move together closely (corr=0.94). Reward increases are coming from actually eating more pellets — the shaping is aligned with the real objective.
- A much wider smoothing window removes most of the wiggle (only ~5% of the standard-smoothed curve's variance survives at the wide window). That's a sign the apparent oscillation is largely window-sampling noise — the 20-episode average swinging with which mazes happened to land in it — rather than the policy itself cycling up and down. If maze-size data isn't logged yet, that's the next thing to add to confirm it directly.
- **Reward breakdown:** the largest positive driver is **Pellet** (avg +1032.3/ep). The largest penalty is **Step** (avg -1715.2/ep).
- Pellet reward (corr=0.49) dominates over BFS shaping (corr=0.44) — the shaping term is well-calibrated.
- Death penalty is strongly anti-correlated with total reward (corr=-0.97). The agent is still dying frequently on high-reward attempts — consider whether the penalty magnitude (-30) is too small relative to the completion bonus (200+).

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
- Training data covers updates 1–33 (65 episodes).
- Log window size: last 100 completed episodes per update (smoothed breakdown).
