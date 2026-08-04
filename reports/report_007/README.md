# Training Report 007 — PPO Stage-1

Generated: 2026-08-04 01:15  
Log file: `RL_logs.txt`

---

## Run Configuration

| Parameter | Value |
|-----------|-------|
| PPO Updates | 1 → 1000 (1000 logged) |
| Total Episodes | 711 |
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
| Best raw avg reward | -96.9 | 10 |
| Best smoothed avg reward | -138.6 | 333 |
| Best avg pellet % | 92.9% | 787 |
| Best max pellet % | 100.0% | 4 |
| Final avg reward | -170.6 | 1000 |
| Final avg pellet % | 89.3% | 1000 |
| Final max pellet % | 100.0% | 1000 |

---

## Plots

| File | Description |
|------|-------------|
| `00_overview.png` | Combined 3-panel summary (reward / pellets / value loss) |
| `01_avg_reward.png` | Average episode reward over updates |
| `02_pellet_completion.png` | Pellet completion % (avg & max) |
| `03_value_loss.png` | Value network loss |

![Overview](00_overview.png)

---

## Notes / Observations

<!-- Add manual notes about this run here -->
- Training data covers updates 1–1000 (711 episodes).
- Log window size: last 20 completed episodes per update.
- Oscillation penalty introduced in this run to combat node-to-node back-and-forth behavior.
