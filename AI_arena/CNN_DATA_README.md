# CNN Dataset Format

`CNN_DATA.jsonl` contains training examples for a CNN that predicts the next
movement of all four ghosts. Each line is one independent JSON object and one
snapshot of the game world.

The collector produces this structure:

```python
{
    "grid":           grid,           # Model input:    [6, 25, 50]
    "maze_width":     maze_width,     # Original unpadded width
    "maze_height":    maze_height,    # Original unpadded height
    "extra_features": extra_features, # Model input:    [45]
    "valid_actions":  valid_actions,  # Output mask:    [4, 4]
    "labels":         labels,         # Training targets:[4]
    "episode_id":     episode_id,     # Episode index
    "episode_step":   episode_step,   # Step within episode
}
```

The direction index is used consistently by `valid_actions` and `labels`:

| Index | Direction |
|---:|---|
| 0 | UP |
| 1 | DOWN |
| 2 | LEFT |
| 3 | RIGHT |

The ghost order is also fixed:

| Index | Ghost |
|---:|---|
| 0 | Blinky |
| 1 | Pinky |
| 2 | Inky |
| 3 | Clyde |

---

## `grid`: spatial model input

`grid` has the fixed shape `[6, 25, 50]` — 6 channels, height 25, width 50.
It is produced by `ObservationFormatter.format_observation()` in
`AI_arena/data/formatter.py`. Smaller mazes are placed at the top-left corner
and zero-padded. Channel 5 (`walkable`) distinguishes active maze cells from
padding.

`maze_width` and `maze_height` store the original unpadded dimensions. They are
metadata only and are not passed to the model.

| Channel | Name | How it is encoded |
|---:|---|---|
| 0 | `maze_bitmask` | `maze[y][x] / 15.0` — raw wall bitmask normalized to [0, 1] |
| 1 | `normal_pellet` | `1.0` if a normal pellet occupies the cell, else `0.0` |
| 2 | `power_pellet` | `1.0` if a power pellet occupies the cell, else `0.0` |
| 3 | `player` | 3×3 heat map centered on the player (center=1.0, orthogonal=0.5, diagonal=0.25) |
| 4 | `ghosts_signed` | 3×3 heat map per ghost: **positive** (+1.0) when dangerous, **negative** (−1.0) when edible; overlapping patches use max/min |
| 5 | `walkable` | `1.0` for every non-wall cell inside the active maze, `0.0` for walls and padding |

> **Note on channel 0 — maze bitmask**
> Each cell value is a 4-bit bitmask encoding which sides are open (passable):
> `bit 0 = NORTH`, `bit 1 = EAST`, `bit 2 = SOUTH`, `bit 3 = WEST`.
> A solid wall cell has value `15` (all bits set → all sides blocked).
> Dividing by 15.0 normalizes the range to [0, 1].

> **Note on channel 4 — signed ghost heat map**
> A non-edible (dangerous) ghost contributes **positive** values to the
> 3×3 patch around its position. An edible (frightened) ghost contributes
> **negative** values. This lets the model distinguish threat from opportunity
> in a single channel.

---

## `extra_features`: non-spatial model input

`extra_features` is a flat vector of **45 floats** produced by
`ObservationFormatter.format_observation()`. The table below lists every entry
in order:

| Index | Name | Description |
|---:|---|---|
| 0 | `player_dir_up` | 1.0 if player is moving UP |
| 1 | `player_dir_down` | 1.0 if player is moving DOWN |
| 2 | `player_dir_left` | 1.0 if player is moving LEFT |
| 3 | `player_dir_right` | 1.0 if player is moving RIGHT |
| 4 | `last_action_up` | 1.0 if the last action taken was UP |
| 5 | `last_action_down` | 1.0 if the last action taken was DOWN |
| 6 | `last_action_left` | 1.0 if the last action taken was LEFT |
| 7 | `last_action_right` | 1.0 if the last action taken was RIGHT |
| 8 | `player_powered` | 1.0 if any ghost is currently edible (player ate a power pellet) |
| 9 | `blinky_edible` | 1.0 if Blinky is edible |
| 10 | `pinky_edible` | 1.0 if Pinky is edible |
| 11 | `inky_edible` | 1.0 if Inky is edible |
| 12 | `clyde_edible` | 1.0 if Clyde is edible |
| 13 | `maze_width_norm` | `maze_width / 50.0` |
| 14 | `maze_height_norm` | `maze_height / 25.0` |
| 15 | `maze_area_norm` | `(maze_width × maze_height − 1) / 1000.0` |
| 16 | `blinky_dx` | `(player_x − blinky_x) / max_dim` |
| 17 | `blinky_dy` | `(player_y − blinky_y) / max_dim` |
| 18 | `blinky_bfs_dist` | BFS distance from player to Blinky, `(dist + 1) / max_dim` |
| 19 | `pinky_dx` | `(player_x − pinky_x) / max_dim` |
| 20 | `pinky_dy` | `(player_y − pinky_y) / max_dim` |
| 21 | `pinky_bfs_dist` | BFS distance from player to Pinky |
| 22 | `inky_dx` | `(player_x − inky_x) / max_dim` |
| 23 | `inky_dy` | `(player_y − inky_y) / max_dim` |
| 24 | `inky_bfs_dist` | BFS distance from player to Inky |
| 25 | `clyde_dx` | `(player_x − clyde_x) / max_dim` |
| 26 | `clyde_dy` | `(player_y − clyde_y) / max_dim` |
| 27 | `clyde_bfs_dist` | BFS distance from player to Clyde |
| 28 | `blinky_dir_up` | 1.0 if Blinky is moving UP |
| 29 | `blinky_dir_down` | 1.0 if Blinky is moving DOWN |
| 30 | `blinky_dir_left` | 1.0 if Blinky is moving LEFT |
| 31 | `blinky_dir_right` | 1.0 if Blinky is moving RIGHT |
| 32 | `pinky_dir_up` | 1.0 if Pinky is moving UP |
| 33 | `pinky_dir_down` | 1.0 if Pinky is moving DOWN |
| 34 | `pinky_dir_left` | 1.0 if Pinky is moving LEFT |
| 35 | `pinky_dir_right` | 1.0 if Pinky is moving RIGHT |
| 36 | `inky_dir_up` | 1.0 if Inky is moving UP |
| 37 | `inky_dir_down` | 1.0 if Inky is moving DOWN |
| 38 | `inky_dir_left` | 1.0 if Inky is moving LEFT |
| 39 | `inky_dir_right` | 1.0 if Inky is moving RIGHT |
| 40 | `clyde_dir_up` | 1.0 if Clyde is moving UP |
| 41 | `clyde_dir_down` | 1.0 if Clyde is moving DOWN |
| 42 | `clyde_dir_left` | 1.0 if Clyde is moving LEFT |
| 43 | `clyde_dir_right` | 1.0 if Clyde is moving RIGHT |
| 44 | `nearest_power_pellet_dist` | BFS distance from player to nearest power pellet, `(dist + 1) / max_dim`; `0.0` if none remain |

> `max_dim = max(maze_width, maze_height)` — used to normalize all distances.

---

## `valid_actions`: hard output mask

`valid_actions` has shape `[4, 4]`: one row per ghost and one column per
direction. `True` means the ghost can move in that direction; `False` means
the direction is blocked by a wall.

```python
valid_actions = [
    [True,  False, True,  False],  # Blinky can move UP or LEFT
    [False, True,  False, True ],  # Pinky  can move DOWN or RIGHT
    [True,  True,  False, False],  # Inky   can move UP or DOWN
    [False, False, True,  True ],  # Clyde  can move LEFT or RIGHT
]
```

Apply this mask to logits as a **hard mask** — the model must never select a
movement through a wall. Apply the same mask during training, validation, and
inference.

---

## `labels`: correct training actions

`labels` contains one correct direction index for every ghost, produced by the
BFS-based ghost expert:

- Ghost **not edible** → label = first step of the BFS shortest path **toward** the player.
- Ghost **edible** (frightened) → label = direction that **maximises** BFS distance from the player.

```python
labels = [0, 3, 1, 2]
```

| Ghost | Label | Direction |
|---|---:|---|
| Blinky | 0 | UP |
| Pinky | 3 | RIGHT |
| Inky | 1 | DOWN |
| Clyde | 2 | LEFT |

Every label must point to an allowed entry in the corresponding action mask —
`valid_actions[ghost_index][label]` must be `True`. Labels are used only during
training; they do not exist when the trained model controls ghosts in the game.

---

## Complete simplified example

```python
sample = {
    "grid":           # shape [6, 25, 50] — 6 channels described above
    "extra_features": [0, 0, 0, 1,   # player moving RIGHT
                       0, 0, 0, 1,   # last action was RIGHT
                       1,            # player is powered
                       1, 1, 0, 0,   # Blinky & Pinky edible
                       0.4, 1.0, 9.999,   # maze dims
                       # ... 29 more spatial/direction values ...
                      ],
    "valid_actions":  [
        [True,  False, True,  False],
        [False, True,  False, True ],
        [True,  True,  False, False],
        [False, False, True,  True ],
    ],
    "labels":         [0, 3, 1, 2],
    "episode_id":     0,
    "episode_step":   42,
    "maze_width":     20,
    "maze_height":    25,
}
```

---

## How to use a batch in PyTorch

The model returns logits with shape `[batch_size, 4, 4]`.

```python
logits = model(grid, extra_features)                         # [B, 4, 4]
masked_logits = logits.masked_fill(~valid_actions, -1e9)    # [B, 4, 4]

loss = criterion(
    masked_logits.reshape(-1, 4),   # [B*4, 4]
    labels.reshape(-1),             # [B*4]
)
```

At inference time, omit `labels` and select the best permitted direction:

```python
logits = model(grid, extra_features)
masked_logits = logits.masked_fill(~valid_actions, -1e9)
predicted_actions = masked_logits.argmax(dim=-1)  # [batch_size, 4]
```

---

## Training in batches

The collector pads all mazes to the same fixed dimensions, so the DataLoader
can stack multiple samples directly:

```text
grid:           [batch_size, 6, 25, 50]
extra_features: [batch_size, 45]
valid_actions:  [batch_size, 4, 4]
labels:         [batch_size, 4]
```

### Validate each sample

```python
assert len(grid) == 6
assert 1 <= maze_width  <= 50
assert 1 <= maze_height <= 25
assert len(extra_features) == 45
assert len(valid_actions) == 4
assert all(len(mask) == 4 for mask in valid_actions)
assert len(labels) == 4

for ghost_index, label in enumerate(labels):
    assert 0 <= label < 4
    assert valid_actions[ghost_index][label] == 1
```

### Convert JSON data to tensors

| Value | Tensor type | Shape |
|---|---|---|
| `grid` | `torch.float32` | `[1, 6, 25, 50]` |
| `extra_features` | `torch.float32` | `[1, 45]` |
| `valid_actions` | `torch.bool` | `[1, 4, 4]` |
| `labels` | `torch.long` | `[1, 4]` |

```python
record = json.loads(line)

grid         = torch.tensor(record["grid"],           dtype=torch.float32).unsqueeze(0)
extra        = torch.tensor(record["extra_features"], dtype=torch.float32).unsqueeze(0)
valid_actions= torch.tensor(record["valid_actions"],  dtype=torch.bool).unsqueeze(0)
labels       = torch.tensor(record["labels"],         dtype=torch.long).unsqueeze(0)

logits        = model(grid, extra)
masked_logits = logits.masked_fill(~valid_actions, -1e9)
loss          = criterion(masked_logits.reshape(-1, 4), labels.reshape(-1))
```

### Example training loop

```python
from AI_arena.data.dataset import create_cnn_dataloader

train_loader = create_cnn_dataloader(
    "AI_arena/data/CNN_DATA.jsonl",
    batch_size=32,
    shuffle=True,
)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for grid, extra_features, valid_actions, labels in train_loader:
        optimizer.zero_grad()
        logits        = model(grid, extra_features)
        masked_logits = logits.masked_fill(~valid_actions, -1e9)
        loss          = criterion(
            masked_logits.reshape(-1, 4),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()
```

---

## Model architecture (actual implementation)

The project uses `GhostCNN` in [`AI_arena/models/cnn_ghost.py`](AI_arena/models/cnn_ghost.py),
built on top of `PacmanCNNBackbone` in [`AI_arena/models/cnn_backbone.py`](AI_arena/models/cnn_backbone.py).

```
Grid input:  [B, 6, 25, 50]
Scalar input:[B, 45]
    |
PacmanCNNBackbone:
  ├── Spatial tower:  CNN (ResBlocks + SEBlocks) → 128-dim
  ├── Scalar tower:   LayerNorm → MLP → 128-dim
  ├── Fusion:         concat(256) → MLP → GRU-hidden
  └── GRU (2-layer) + LayerNorm → Linear → ReLU → Dropout → 256-dim
    |
GhostCNN head:
  Linear(256 → 16) → reshape → [B, 4 ghosts, 4 actions]
```

Train with:

```bash
uv run python -m AI_arena.ghosts.ghost_training --epochs 30
```

By default, training reads `AI_arena/data/CNN_DATA.jsonl` and saves the best
weights to `AI_arena/models/ghost_ai.pt`.

---

## Collecting the dataset

Use the ghost data collector to generate `CNN_DATA.jsonl`:

```bash
uv run python -m AI_arena.ghosts.ghost_collector --samples 50000
```

The collector:
1. Runs `PacmanPlayerEnv` episodes with the `PacmanExpert` controlling the player.
2. At each step, calls `ObservationFormatter.format_observation()` to build `grid` and `extra_features`.
3. Calls `GhostExpert` to compute the BFS-optimal label for each ghost.
4. Writes one JSONL record per step to `CNN_DATA.jsonl`.
