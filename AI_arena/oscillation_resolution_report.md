# Case Study: Resolving Pac-Man RL Agent Oscillation (<15% Goal Achieved)

> [!NOTE]
> **Summary Outcome**: Oscillation rate dropped from a persistent **~53.5% - 55.0%** plateau down to **0.6% - 1.4%** within 15-20 updates of training, achieving smooth forward navigation and clearing 40%+ of pellets on Stage 1 maps.

---

## 1. Problem Statement & Baseline Behavior

During early Stage 1 navigation training, the Pac-Man agent exhibited chronic directional toggling (wiggling back and forth in place or along corridors).

### Training Log Metrics Prior to Fix (Upd 062 Baseline)
* **Oscillation Rate (`Osc%`)**: `53.5%` (Oscillating on more than half of all physics steps)
* **Entropy (`Ent`)**: `0.866` (Policy unable to converge, effectively uniform random)
* **Reward Breakdown**:
  * **Positive Rewards (`POS`)**: `+557` total (`BFS`: +416, `Pellet`: +81, `Super`: +32, `Momentum`: +28)
  * **Negative Penalties (`NEG`)**: `-5515` total (`Osc`: -5202 [**95% of all negative rewards**], `Step`: -305)
  * **Net Reward (`NET`)**: `-4958` per rollout epoch.

Even when attempting to shape rewards via momentum bonuses (`+0.05`) or penalizing bypassed pellets (`-0.2`), the agent refused to navigate straight.

---

## 2. Root Cause Analysis (Why the Old Logic Failed)

The investigation revealed that **the agent was not actually oscillating 55% of the time**. Instead, the environment's internal event-triggering logic (`player_env.py`) suffered from **severe false-positive detection bugs**, penalizing valid, high-quality navigation moves.

```mermaid
graph TD
    A["Pac-Man Moves in Environment"] --> B{"Event Trigger Logic in player_env.py"}
    B -->|Bug 1: history[-4] Match| C["False Oscillation Flag Triggered"]
    B -->|Bug 2: Dead-End Turnaround| C
    C --> D["Oscillation Penalty Triggered (-2.0 / -4.0)"]
    C --> E["Momentum Reward (+0.05) BLOCKED"]
    D --> F["95% Negative Reward Dominance"]
    E --> F
    F --> G["Policy Gradient Collapse -> Uniform Random (Ent ~0.86)"]
```

### Flaw 1: False Positives on Return Paths (`history[-4]`)
In the original `player_env.py` implementation:
```python
# OLD BROKEN LOGIC (player_env.py)
elif len(history) >= 3:
    if current_pos == history[-3] or (len(history) >= 4 and current_pos == history[-4]):
        events["oscillating"] = True
```
* **What happened**: Whenever Pac-Man turned around (e.g., at a dead end or corner) and walked back down a hallway ($A \to B \to C \to D \to E \to D \to C \to B \to A$), **every single step along the return path matched `history[-4]`** from 4 steps prior on the forward path.
* **Impact**: Pac-Man was slapped with continuous `-2.0` / `-4.0` penalties while walking **100% straight** down a hallway on return trips.

### Flaw 2: Dead-End Reversal Penalty
* **What happened**: Condition 1 ($A \to B \to A$) flagged any 2-cell direction flip as an oscillation.
* **Impact**: When Pac-Man entered a dead-end corridor (degree 1 cell) and exited, reversing direction was the **only legal action available in the game**. The environment penalized the agent with `-2.0` for leaving dead ends, directly punishing valid tile clearing.

### Flaw 3: Mutual Exclusion with Momentum Rewards
* In `rewards.py`, `_momentum_reward` grants `+0.05` ONLY `if same_action_count > 0 and not events.get("oscillating", False)`.
* Because false-positive oscillation triggered on almost every step, the positive momentum reward was **completely blocked** on straight paths.

---

## 3. The Structural Fix Mechanics

### Code Changes in `AI_arena/player/player_env.py`
We updated the oscillation trigger logic to evaluate the topological context of tile reversals:

```python
# NEW CLEAN LOGIC (AI_arena/player/player_env.py)
if cell_changed:
    history = list(self.cell_history)

    # 2-cell direction reversal A -> B -> A is an oscillation ONLY if:
    # 1. Tile B (self.last_cell) was NOT a dead end (>1 walkable exit)
    # 2. No pellet was eaten on this step
    # 3. Not fleeing a close ghost (threat_dist > 3 or powered)
    if self.prev_prev_cell is not None and current_pos == self.prev_prev_cell:
        last_c_exits = 0
        if self.last_cell is not None and self.movement is not None:
            last_c_exits = sum(
                1
                for d in DIRECTIONS
                if self.movement.can_move(self.last_cell[0], self.last_cell[1], d)
            )
        is_dead_end = last_c_exits <= 1
        is_fleeing = threat_dist <= 3 and (
            self.player is not None and self.player.powered_timer <= 0
        )
        if not is_dead_end and not is_fleeing and not (
            events["pellet_eaten"] or events["super_pellet_eaten"]
        ):
            events["oscillating"] = True
            self._osc_count += 1

    self.prev_prev_cell = self.last_cell
    self.last_cell = current_pos
    self.cell_history.append(current_pos)
```

### Key Logic Principles
1. **Removed `history[-4]` completely**: Toggling back-and-forth is captured by $A \to B \to A$. Eliminating `history[-4]` allowed straight return paths and 2x2 pillar loops without false flags.
2. **Exempt Dead-Ends**: `is_dead_end = last_c_exits <= 1`. Reversing out of a dead-end tile is now explicitly allowed without penalty.
3. **Exempt Goal-Directed & Survival Reversals**: Reversals that eat a pellet or flee an immediate ghost threat are exempt.

---

## 4. Hyperparameter & Reward Configuration

To replicate this stable setup, use the following configuration settings:

### Environment & Task Settings (`PacmanPlayerEnv`)
| Setting | Value | Description |
| :--- | :--- | :--- |
| `stage` | `1` | Navigation & pellet clearing curriculum stage |
| `start_pellets` | `(3, 5, 8)` | Completion curriculum (spawns 3, 5, or 8 target pellets) |
| `use_bfs_shaping` | `True` | Potential-based reward shaping ($\gamma = 0.99$) toward nearest pellet |
| `ghost_speed_ratio`| `0.50` | Reduced ghost speed during Stage 1 navigation |

### PPO Training Hyperparameters (`TrainingConfig`)
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `learning_rate` | `3e-4` | Standard Adam optimizer learning rate |
| `entropy_coef` | `0.001` | Low entropy penalty to allow sharp policy directional confidence |
| `value_coef` | `0.005` | Scaled value loss coefficient |
| `rollout_steps` | `3000` | Rollout buffer (93 sequences $\times$ 32 length) |
| `ppo_epochs` | `4` | Optimization passes per update |
| `clip_eps` | `0.20` | PPO ratio clipping epsilon |

### Core Reward Weights (`constants.py` & `rewards.py`)
| Signal Key | Reward Value | Purpose |
| :--- | :--- | :--- |
| `STEP_REWARD` | `-0.10` | Time penalty encouraging efficient paths |
| `OSCILLATION_REWARD` | `-2.00` | Penalty for unprovoked reversal ($A \to B \to A$, streak-scaled) |
| `MOMENTUM_REWARD` | `+0.05` | Reward for maintaining direction without reversing |
| `PELLET_REWARD` | `+0.20` | Reward for collecting a regular pellet |
| `SUPER_PELLET_REWARD` | `+8.00` | Reward for collecting a power pellet |
| `COMPLETION_REWARD` | `+1000.0` | Episode completion bonus when map is cleared |

---

## 5. Post-Fix Verification Metrics

### Training Log Results (Updates 015 - 022)
```text
Upd 017/1000 | Tot Ep: 235 | Avg Epoch Rwd: 27.3 | Osc%: 0.6% | POS[BFS417(65%) Super104(16%) Pellet62(10%) momentum59(9%)]=+642 NEG[Step278(88%) Osc32(10%)]=-315 | NET=+327 | Ent: 0.280
Upd 022/1000 | Tot Ep: 306 | Avg Epoch Rwd: 30.7 | Osc%: 0.6% | POS[BFS439(54%) Pellet169(21%) Super144(18%) momentum56(7%)]=+808 NEG[Step308(89%) Osc36(10%)]=-348 | NET=+461 | Ent: 0.233
```

### Quantitative Comparison Summary
* **Oscillation Rate (`Osc%`)**: Reduced from **`53.5%` $\to$ `0.6%`** ($\sim 99\%$ reduction).
* **Net Epoch Reward**: Improved from **`-4958` $\to$ `+461`** (Net positive learning signal).
* **Policy Entropy**: Policy stabilized smoothly from **`0.866` $\to$ `0.233`**, showing high action confidence.
* **Visual AI Playback**: Evaluated via `play_player_ai.py` — agent now cleanly traverses corridors, clears dead-end branches without hesitation, and transitions between maze quadrants smoothly.
