# CNN Dataset Format

`CNN_DATA.jsonl` contains training examples for a CNN that predicts the next
movement of all four ghosts. Each line is one independent JSON object and one
snapshot of the game world.

The collector produces this structure:

```python
{
    "grid": grid,                     # Model input: [12, 50, 25]
    "maze_width": maze_width,         # Original unpadded width
    "maze_height": maze_height,       # Original unpadded height
    "extra_features": extra_features, # Model input: [9]
    "valid_actions": valid_actions,   # Output mask: [4, 4]
    "labels": labels,                 # Training targets: [4]
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

## `grid`: spatial model input

`grid` has the fixed shape `[12, 50, 25]`. It is similar to a 12-channel image:
every channel describes one kind of information at every maze cell. Smaller
mazes are placed at the top-left and zero-padded. The `valid_cell` channel
distinguishes maze cells from padding.

`maze_width` and `maze_height` store the original unpadded dimensions. The
dataset validator checks that they fit inside the fixed tensor and that every
grid value outside those dimensions is zero. They are metadata and are not
passed to the model.

| Channel | Name | Meaning of value `1` |
|---:|---|---|
| 0 | `wall_up` | The cell has a wall on its upper side |
| 1 | `wall_down` | The cell has a wall on its lower side |
| 2 | `wall_left` | The cell has a wall on its left side |
| 3 | `wall_right` | The cell has a wall on its right side |
| 4 | `normal_pellet` | A normal pellet is in the cell |
| 5 | `super_pellet` | A power pellet is in the cell |
| 6 | `player` | The player is in the cell |
| 7 | `blinky` | Blinky is in the cell |
| 8 | `pinky` | Pinky is in the cell |
| 9 | `inky` | Inky is in the cell |
| 10 | `clyde` | Clyde is in the cell |
| 11 | `valid_cell` | The cell is walkable rather than a blocked pattern cell |

For example, this player channel describes a player at `(x=1, y=0)` in a
3-by-3 maze:

```python
grid[6] = [
    [0, 1, 0],
    [0, 0, 0],
    [0, 0, 0],
]
```

The grid and its channels are passed to the CNN after conversion to a floating
point tensor.

## `extra_features`: non-spatial model input

`extra_features` contains nine non-spatial values:

```text
[
    player_up,
    player_down,
    player_left,
    player_right,
    player_powered,
    blinky_frightened,
    pinky_frightened,
    inky_frightened,
    clyde_frightened,
]
```

The first four entries encode the player's direction. The remaining entries
encode the global powered state and the frightened state of each ghost.

These values are model inputs. A typical model processes `grid` with
convolutional layers, processes or concatenates `extra_features` after the
convolutional layers, and outputs four action scores for each ghost.

## `valid_actions`: hard output mask

`valid_actions` has shape `[4, 4]`: one row per ghost and one column per
direction. A `1` means that the ghost can move in that direction; a `0` means
that the direction is blocked.

```python
valid_actions = [
    [1, 0, 1, 0],  # Blinky can move UP or LEFT
    [0, 1, 0, 1],  # Pinky can move DOWN or RIGHT
    [1, 1, 0, 0],  # Inky can move UP or DOWN
    [0, 0, 1, 1],  # Clyde can move LEFT or RIGHT
]
```

This is not a normal model input. Apply it as a hard mask to the four output
scores so that the model can never select a movement through a wall. The mask
must be applied both during training and during inference.

## `labels`: correct training actions

`labels` contains one correct direction index for every ghost. These answers
are produced from the first direction in the BFS path and are used to calculate
the supervised training loss.

```python
labels = [0, 3, 1, 2]
```

This means:

| Ghost | Label | Correct movement |
|---|---:|---|
| Blinky | 0 | UP |
| Pinky | 3 | RIGHT |
| Inky | 1 | DOWN |
| Clyde | 2 | LEFT |

Every label should point to an allowed entry in the corresponding action mask.
For example, Blinky's label is `0`, so `valid_actions[0][0]` must equal `1`.
Labels are used only during training; they do not exist when the trained model
is controlling ghosts in the game.

## Complete simplified example

The real `grid` contains all 12 matrices. It is abbreviated below to keep the
example readable:

```python
sample = {
    "grid": "12 matrices, each with shape [50, 25]",
    "extra_features": [0, 0, 0, 1, 1, 1, 1, 0, 0],
    "valid_actions": [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ],
    "labels": [0, 3, 1, 2],
}
```

The sample says the player is moving right and is powered; Blinky and Pinky
are frightened. The teacher wants Blinky to move up, Pinky right, Inky down,
and Clyde left; all four target actions are permitted by their masks.

## How to use a batch in PyTorch

The model should return logits with shape `[batch_size, 4, 4]`.

```python
logits = model(grid, extra_features)
masked_logits = logits.masked_fill(valid_actions == 0, -1e9)

loss = criterion(
    masked_logits.reshape(-1, 4),
    labels.reshape(-1),
)
```

At inference time, omit `labels` and select the best permitted direction:

```python
logits = model(grid, extra_features)
masked_logits = logits.masked_fill(valid_actions == 0, -1e9)
predicted_actions = masked_logits.argmax(dim=-1)
```

## Training in batches

The collector pads all mazes to the same dimensions, so the data loader can
stack multiple samples directly:

```text
grid:           [batch_size, 12, 50, 25]
extra_features: [batch_size, 9]
valid_actions:  [batch_size, 4, 4]
labels:         [batch_size, 4]
```

Adaptive pooling may still be used, but is not required to handle varying
input dimensions.

### Validate each sample

Check every record before training:

```python
assert len(grid) == 12
assert 1 <= maze_width <= 25
assert 1 <= maze_height <= 50
assert len(extra_features) == 9
assert len(valid_actions) == 4
assert all(len(mask) == 4 for mask in valid_actions)
assert len(labels) == 4

for ghost_index, label in enumerate(labels):
    assert 0 <= label < 4
    assert valid_actions[ghost_index][label] == 1
```

Invalid records should be fixed or excluded instead of silently accepted.

### Convert JSON data to tensors

Read one JSONL record and convert its lists to tensors:

| Value | Tensor type | Shape passed to training |
|---|---|---|
| `grid` | `torch.float32` | `[1, 12, 50, 25]` |
| `extra_features` | `torch.float32` | `[1, 9]` |
| `valid_actions` | `torch.bool` | `[1, 4, 4]` |
| `labels` | `torch.long` | `[1, 4]` |

Pass only `grid` and `extra_features` into the model. Apply `valid_actions` to
the output and use `labels` to calculate training loss:

```python
criterion = torch.nn.CrossEntropyLoss()

logits = model(grid, extra_features)
masked_logits = logits.masked_fill(~valid_actions, -1e9)

loss = criterion(
    masked_logits.reshape(-1, 4),
    labels.reshape(-1),
)
```

Use the same mask during training, validation, testing, and gameplay. It
guarantees that the selected movement is not blocked by a wall.

## Passing JSONL Data to the Model

Use `CNN_DATA.jsonl` as the training-data source, but do not pass JSON objects
directly to the CNN. The dataset loader must read one line with `json.loads()`
and convert each field to a PyTorch tensor:

- `grid` becomes `torch.float32`.
- `extra_features` becomes `torch.float32`.
- `valid_actions` becomes `torch.bool`.
- `labels` becomes `torch.long`.

Pass only `grid` and `extra_features` into the model. Use `valid_actions` to
mask blocked directions and use `labels` to calculate the training loss:

```python
record = json.loads(line)

grid = torch.tensor(record["grid"], dtype=torch.float32).unsqueeze(0)
extra = torch.tensor(
    record["extra_features"], dtype=torch.float32
).unsqueeze(0)
valid_actions = torch.tensor(
    record["valid_actions"], dtype=torch.bool
).unsqueeze(0)
labels = torch.tensor(
    record["labels"], dtype=torch.long
).unsqueeze(0)

logits = model(grid, extra)
masked_logits = logits.masked_fill(~valid_actions, -1e9)

loss = criterion(
    masked_logits.reshape(-1, 4),
    labels.reshape(-1),
)
```

For one sample, the tensor shapes are:

```text
grid:           [1, 12, 50, 25]
extra:          [1, 9]
model output:   [1, 4, 4]
valid_actions:  [1, 4, 4]
labels:         [1, 4]
```

### Example training loop

Generate and save the dataset before training. The data loader should shuffle
the training records and return them gradually; do not load or pass the entire
dataset to the model at once. Increase or decrease the batch size based on
available memory.

```python
from AI_arena.cnn_dataset import create_cnn_dataloader


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

        # Only numeric grid and extra-feature tensors enter the CNN.
        logits = model(grid, extra_features)

        # Prevent the loss from selecting directions blocked by walls.
        masked_logits = logits.masked_fill(~valid_actions, -1e9)
        loss = criterion(
            masked_logits.reshape(-1, 4),
            labels.reshape(-1),
        )

        loss.backward()
        optimizer.step()
```

After the last sample, the next epoch reads the shuffled training dataset
again. Validation and test records must remain separate and must not be used
by `optimizer.step()`.

## Recommended CNN architecture

The following architecture is a practical starting point for the fixed input
shape `[batch_size, 12, 50, 25]`:

```text
Grid input: 12 x 50 x 25
    |
Conv2d: 12 -> 32 filters, 3x3 kernel, padding 1
ReLU
MaxPool2d: 2x2
    |
Conv2d: 32 -> 64 filters, 3x3 kernel, padding 1
ReLU
MaxPool2d: 2x2
    |
Conv2d: 64 -> 128 filters, 3x3 kernel, padding 1
ReLU
AdaptiveAvgPool2d: 4x4
    |
Flatten and concatenate the 9 extra features
    |
Linear: 2057 -> 256
ReLU
Dropout: 0.3
    |
Linear: 256 -> 128
ReLU
    |
Linear: 128 -> 16
    |
Reshape: [batch_size, 4 ghosts, 4 actions]
```

The first fully connected layer has 2,057 inputs:

```text
(128 channels * 4 * 4) + 9 extra features = 2057
```

An example PyTorch implementation is:

```python
import torch
from torch import nn


class GhostCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 4 + 9, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
        )

    def forward(self, grid, extra_features):
        spatial_features = self.cnn(grid)
        spatial_features = torch.flatten(spatial_features, start_dim=1)
        combined_features = torch.cat(
            (spatial_features, extra_features), dim=1
        )
        logits = self.head(combined_features)
        return logits.view(-1, 4, 4)
```

The filter counts `32`, `64`, and `128` are initial recommendations. They can
be adjusted later by comparing training and validation performance.

The project implementation is in `AI_arena/cnn_model.py`. Train it with:

```bash
uv run python -m AI_arena.cnn_training
```

By default, training reads `AI_arena/data/CNN_DATA.jsonl`, runs for 20 epochs,
and saves the weights to `AI_arena/models/ghost_ai.pt`. Use `--help` to see
options for changing the dataset, output path, epochs, batch size, and learning
rate.

## Next Steps

1. Generate a larger CNN dataset instead of the current single sample.
2. Create a PyTorch JSONL dataset loader.
3. Validate every record's dimensions, labels, and valid-action masks.
4. Convert fields to tensors with the documented types.
5. Build a CNN that:
   - Accepts 12 spatial channels.
   - Accepts 9 extra features.
   - Produces four direction logits for each of the four ghosts.
6. Select a batch size that fits the available CPU or GPU memory.
7. Mask blocked actions before calculating cross-entropy loss.
8. Add validation and testing, and save the best model.
9. Integrate model predictions into ghost gameplay.
