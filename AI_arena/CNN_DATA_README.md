# CNN Dataset Format

`CNN_DATA.jsonl` contains training examples for a CNN that predicts the next
movement of all four ghosts. Each line is one independent JSON object and one
snapshot of the game world.

The collector produces this structure:

```python
{
    "grid": grid,                     # Model input: [12, H, W]
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

`grid` has shape `[12, H, W]`, where `H` and `W` are the maze height and
width. It is similar to a 12-channel image: every channel describes one kind
of information at every maze cell.

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
    "grid": "12 matrices, each with shape [H, W]",
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

## Training one sample at a time

The collector preserves the original `[12, H, W]` maze dimensions. Use a data
loader with `batch_size=1`; it adds the leading batch dimension without
requiring padding:

```text
grid:           [1, 12, H, W]
extra_features: [1, 9]
valid_actions:  [1, 4, 4]
labels:         [1, 4]
```

The CNN should use adaptive pooling before its fully connected layers so it
can accept different maze heights and widths.

### Validate each sample

Check every record before training:

```python
assert len(grid) == 12
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
| `grid` | `torch.float32` | `[1, 12, H, W]` |
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
