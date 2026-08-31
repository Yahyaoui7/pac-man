# 📖 AI Arena: Data Collection, Representation & Model Architecture

This document provides a comprehensive overview of how data is collected, formatted, and processed by the neural network models for both Pac-Man and the Ghosts in the AI Arena.

---

## 1. How Data is Collected (The Interaction Loop)

The data is collected by running headless game simulations using `ghost_collector.py`:

```
                    ┌───────────────────────────────┐
                    │       PacmanPlayerEnv         │
                    │ (Maze, Pellets, Player, Ghosts│
                    └───────────────┬───────────────┘
                                    │
       ┌────────────────────────────┴───────────────────────────┐
       ▼                                                        ▼
1. Extract Game State                                    2. Compute Labels
   • Player position & direction                            • GhostExpert runs BFS from player
   • 4 Ghost positions, directions & edible flags           • If dangerous: min BFS dist (CHASE)
   • Pellets and Maze walls                                 • If edible: max BFS dist (FLEE)
       │                                                        │
       ▼                                                        ▼
3. Format Observation                                    4. Store in JSONL
   • grid: [6, 25, 50]                                      • grid, extra_features,
   • extra_features: [45]                                     valid_actions, labels
       │                                                        │
       └────────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
                         5. Advance Simulation
                            • PacmanExpert computes optimal move
                            • env.step(player_action) moves entities
                            • Repeats for next snapshot
```

### Why `PacmanExpert` is used during Ghost collection:
Ghosts need a moving, intelligent target to react to. If Pac-Man stood still or died immediately, ghosts would stay near their spawns and the dataset would have no mid-game chase data. `PacmanExpert` keeps the game alive for ~300 steps per episode.

---

## 2. How the Player Calculates Direction Scores (`PacmanExpert`)

When Pac-Man must move, `PacmanExpert.choose_action()` performs a **7-step tree lookahead search**:

1. **Immediate Score**:
   - **Momentum**: `+0.2` if keeping current direction.
   - **Pellet**: `+18.0` for regular pellet, `+35.0` for power pellet.
2. **Lookahead Search (Depth 1 to 7)**:
   - **Ghost Threat**: If a dangerous ghost can arrive at cell $\le \text{depth} + 1$: **`-100,000.0`** (immediate death pruning).
   - **Safe Distance**: If safe, adds `+1.5 * safe_distance` (encourages spacing).
   - **Dead-End Avoidance**: **`-25.0`** penalty if corridor has only $\le 1$ exit while ghosts are nearby.
   - **Hunting Edible Ghosts**: `+7.0 / (dist + 1)` if ghost is frightened and reachable before timer ends.
   - **Leaf Heuristic (at depth 7)**: `-0.6 * BFS_dist_to_nearest_pellet` to guide movement toward food clusters.
3. **Action Selection**: $\text{Action} = \text{argmax}(\text{Scores})$.

---

## 3. How Ghost Target Labels are Decided (`GhostExpert`)

At each snapshot, BFS shortest-path distances from Pac-Man to all maze tiles are calculated (`player_dists`):

- **Dangerous Ghost (Hunting)**:
  $$\text{Score}(\text{direction}) = -\text{BFS\_Distance}(\text{next\_cell}, \text{player})$$
  $\rightarrow$ Picks the move that **minimizes** distance to Pac-Man (**CHASE**).
- **Edible Ghost (Frightened)**:
  $$\text{Score}(\text{direction}) = +\text{BFS\_Distance}(\text{next\_cell}, \text{player})$$
  $\rightarrow$ Picks the move that **maximizes** distance from Pac-Man (**FLEE**).

---

## 4. Feature Encoding: What the Data Looks Like

Each line in `CNN_DATA.jsonl` contains:

### A. `grid` Tensor: Shape `[6, 25, 50]` (Spatial Data)

| Channel | Name | Encoding |
|:---:|:---|:---|
| **0** | `maze_bitmask` | `maze[y][x] / 15.0` (wall topology, range $[0, 1]$) |
| **1** | `normal_pellet` | `1.0` if standard pellet present, else `0.0` |
| **2** | `power_pellet` | `1.0` if energizer pellet present, else `0.0` |
| **3** | `player` | $3 \times 3$ Gaussian-like patch (center=`1.0`, sides=`0.5`, diags=`0.25`) |
| **4** | `ghosts_signed` | $3 \times 3$ patch per ghost: **$+1.0$** (dangerous), **$-1.0$** (edible) |
| **5** | `walkable` | `1.0` for open tiles, `0.0` for solid walls & zero-padded margins |

---

### B. `extra_features` Vector: Shape `[45]` (Scalars & Vectors)

```text
[0..3]   : Player direction (One-hot: UP, DOWN, LEFT, RIGHT)
[4..7]   : Player last action (One-hot: UP, DOWN, LEFT, RIGHT)
[8]      : Player powered active flag (1.0 if power pellet active)
[9..12]  : Ghost edible flags (Blinky, Pinky, Inky, Clyde)
[13..15] : Normalized maze width, height, and area
[16..27] : Per ghost (4 ghosts x 3 values):
           - dx to player normalized: (px - gx) / max_dim
           - dy to player normalized: (py - gy) / max_dim
           - BFS distance to player:  (dist + 1) / max_dim
[28..43] : Per ghost direction (4 ghosts x 4 one-hot directions)
[44]     : Distance to nearest power pellet
```

---

## 5. Concrete Snapshot Sequence Example

Here is how the data evolves across 3 consecutive steps in the same episode:

```text
─── STEP 0 (Game Start) ──────────────────────────────────────────────────────────
Maze: Pacman at center (10, 12), Blinky at corner (0, 0)
• grid[ch3]: Heatmap centered at (10, 12)
• grid[ch4]: Positive heatmap (+1.0) at (0, 0)
• extra_features: blinky_dx = +0.48, blinky_dy = +0.40, blinky_edible = 0.0
• label: Blinky = DOWN (0)  [Chasing Pacman via shortest BFS corridor]

─── STEP 1 (Pacman moves RIGHT, Blinky moves DOWN) ──────────────────────────────
Maze: Pacman at (10, 13), Blinky at (1, 0)
• grid[ch3]: Heatmap shifted to (10, 13)
• grid[ch4]: Heatmap shifted to (1, 0)
• extra_features: blinky_dx = +0.52, blinky_dy = +0.36, player_dir = RIGHT
• label: Blinky = DOWN (0)  [Continuing chase]

─── STEP 2 (Pacman eats Power Pellet) ───────────────────────────────────────────
Maze: Pacman at (10, 14), Blinky at (2, 0) is now EDIBLE
• grid[ch4]: Ghost patch inverted to NEGATIVE (-1.0) at (2, 0)
• extra_features: player_powered = 1.0, blinky_edible = 1.0
• label: Blinky = UP (1)  [Ghost switches to FLEE mode, reversing away from Pacman]
```

---

## 6. Neural Network Architecture: Layers & Purpose

The model (`GhostCNN`) uses a dual-tower backbone (`PacmanCNNBackbone`):

```
       grid: [B, 6, 25, 50]                       extra_features: [B, 45]
                │                                            │
       ┌────────┴────────┐                          ┌────────┴────────┐
       │  SPATIAL TOWER  │                          │  SCALAR TOWER   │
       │  (CNN + ResNet) │                          │  (LayerNorm+MLP)│
       └────────┬────────┘                          └────────┬────────┘
             [B, 128]                                     [B, 128]
                │                                            │
                └──────────────────┬─────────────────────────┘
                                   ▼
                         FUSION: Concat [B, 256]
                                   │
                                   ▼
                         GRU RECURRENT MEMORY (384)
                                   │
                                   ▼
                         LayerNorm & Dropout
                                   │
                                   ▼
                         Latent Vector: [B, 256]
                                   │
                                   ▼
                       CLASSIFICATION HEAD: Linear(256 -> 16)
                                   │
                                   ▼
                       Reshape: [B, 4 ghosts, 4 actions]
```

### Layer-by-Layer Breakdown & Rationale

| Component | Layer Type | Shape | Why We Need It |
|:---|:---|:---|:---|
| **Spatial Input** | Input Tensor | `[B, 6, 25, 50]` | Full 2D game state: walls, pellets, player & ghost coordinates. |
| **CNN Convolutions** | `Conv2d(6->64)` + `ReLU` | `[B, 64, 25, 50]` | Extracts local spatial features (walls, adjacent corridors, nearby items). |
| **ResBlocks** | `Conv2d + Conv2d + Skip` | `[B, 64, 25, 50]` | Prevents vanishing gradients; allows deep representation of maze structure. |
| **SEBlock** | Squeeze-and-Excitation | `[B, 64, 25, 50]` | Channel-attention: dynamically weighs channels (e.g. boosts ghost channel when near). |
| **Spatial Compress** | `Linear(10400 -> 256 -> 128)` | `[B, 128]` | Compresses large 2D visual maps into a compact 128-dim spatial embedding. |
| **Scalar LayerNorm**| `LayerNorm(45)` | `[B, 45]` | Normalizes raw scalar distances and flags to zero-mean unit-variance. |
| **Scalar MLP** | `Linear(45 -> 128 -> 128)` | `[B, 128]` | Processes non-spatial relations (relative $dx, dy$, BFS distances, edible timers). |
| **Fusion Layer** | `Concat + Linear(256 -> 384)` | `[B, 384]` | Blends spatial map understanding (50%) and exact distance features (50%). |
| **GRU Memory** | `nn.GRU(2 layers, 384 hidden)` | `[B, 384]` | Temporal sequence processor; tracks movement velocity and trajectory context. |
| **GRU LayerNorm** | `LayerNorm(384)` | `[B, 384]` | Stabilizes recurrent activations across backpropagation. |
| **Output Head** | `Linear(256 -> 16)` | `[B, 16]` | Maps features to unnormalized probabilities for 4 ghosts $\times$ 4 actions. |
| **Action Reshape** | `.view(-1, 4, 4)` | `[B, 4, 4]` | Outputs logits for each ghost: `[Blinky, Pinky, Inky, Clyde] x [UP, DOWN, LEFT, RIGHT]`. |

---

## 7. How the Loss is Computed (`masked_cross_entropy`)

Ghosts must never move into walls. During training, invalid moves are masked out with $-\infty$:

$$\text{Masked Logits} = \text{logits}.\text{masked\_fill}(\sim\text{valid\_actions}, -\infty)$$
$$\mathcal{L} = \text{CrossEntropyLoss}(\text{Masked Logits}, \text{labels})$$

This forces the network to learn only valid, optimal navigation decisions.










## use each ghoss target player to use bfs with heuristic
-the first one target dirctly  just bfs
-and second target use  bfs with heuristic but one step forword from player
-and three one two step forword from playerx
-and four ghost four step forword from player
