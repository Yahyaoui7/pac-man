# Pac-Man Supervised Player

## Commands

Collect expert data and train in one command:

```bash
uv run python -u -m AI_arena.player.player_training \
    --collect-samples 10000 \
    --stage 2 \
    --epochs 20 \
    --batch-size 64
```

Or run collection and training separately:

```bash
make collect-player
make train-player
```

Then test the trained player visually:

```bash
uv run python -m AI_arena.player.play_player_ai
```

The separate commands are equivalent to:

```bash
uv run python -m AI_arena.player.player_collector \
    --samples 10000 \
    --stage 2 \
    --horizon 7

uv run python -m AI_arena.player.player_training \
    --epochs 20 \
    --batch-size 64 \
    --log-interval 10
```

Press `Ctrl+C` during supervised training to save the current model and its
optimizer checkpoint safely. Continue later with:

```bash
uv run python -m AI_arena.player.player_training --resume --epochs 10
```

Training prints progress every 10 batches. Add `--log-interval 1` to print
after every batch.

Use `--fresh` (the default behavior) to ignore an existing checkpoint and
start again from random weights.

# Pac-Man Supervised Learning / Imitation Learning Plan

This section describes a new Pac-Man player trained only from expert
state/action examples. It is separate from the existing PPO player and from
the existing supervised ghost dataset.

## Important current-project status

- `AI_arena/data/formatter.py` already creates a 12-channel observation for
  the live game and the player environment.
- `AI_arena/data_collector/` currently creates labels for **four ghosts**. It
  does not create Pac-Man expert labels.
- `AI_arena/ghosts/ghost_training.py` is supervised ghost training.
- `AI_arena/player/player_training.py` now trains `PlayerImitationCNN` with
  masked cross-entropy; it no longer runs PPO.
- `AI_arena/player/expert.py` and `player_collector.py` generate Pac-Man labels
  from risk-aware expert play.

Do not reuse ghost labels as Pac-Man labels. The new Pac-Man record should
contain one `label` and one `[4]` `valid_actions` mask, rather than the ghost
dataset's four labels and `[4, 4]` mask.

## 1. Canonical 12-channel input

The project already uses the following zero-based channel order. Keep it to
avoid breaking the live formatter, ghost model, PPO checkpoints, and existing
datasets:

| Index | Channel | Value `1` means |
|---:|---|---|
| 0 | `wall_up` | wall above this cell |
| 1 | `wall_down` | wall below this cell |
| 2 | `wall_left` | wall left of this cell |
| 3 | `wall_right` | wall right of this cell |
| 4 | `normal_pellet` | normal pellet in this cell |
| 5 | `super_pellet` | power pellet in this cell |
| 6 | `player` | Pac-Man occupies this cell |
| 7 | `blinky` | Blinky occupies this cell |
| 8 | `pinky` | Pinky occupies this cell |
| 9 | `inky` | Inky occupies this cell |
| 10 | `clyde` | Clyde occupies this cell |
| 11 | `valid_cell` | real walkable cell, not padding/blocked pattern |

The tensor shape is `[12, 50, 25]`. A smaller maze is zero-padded. Maze
coordinates must always use `grid[y][x]`; entity positions in this project are
usually exposed as `(grid_x, grid_y)`, so convert them carefully.

The four wall channels must come from the maze bit flags (`NORTH`, `SOUTH`,
`WEST`, `EAST`), not from whether the neighboring array entry looks valid.
This matters for one-way/special topology and keeps the input identical to
`MovementSystem.can_move()`.

Tunnels are not currently represented by `MovementSystem.can_move()`: it
rejects moves outside the rectangular grid. If wrap-around tunnels are added,
both movement/pathfinding and the observation need the same explicit tunnel
edges. A normal CNN cannot infer that the left and right boundary cells are
connected merely from padding.

## 2. Are the 12 channels enough?

They are enough for static geometry, pellets, and current entity positions,
but they are not a complete Markov state. Two visually identical 12-channel
grids can require different actions when ghost directions, frightened timers,
or Pac-Man's current direction differ.

Keep exactly 12 spatial channels and pass dynamic non-spatial values through a
separate feature vector. The existing `PacmanCNNBackbone` already supports
this pattern by concatenating `extra_features` after the convolutions.

Recommended extra inputs, normalized to roughly `[0, 1]` or `[-1, 1]`:

- Pac-Man direction: 4-value one-hot.
- Each ghost direction: 4 values per ghost (16 total). Add an explicit
  stationary/unknown flag if that state occurs in gameplay.
- Each ghost frightened/edible flag: 4 values.
- Each ghost frightened timer divided by its maximum: 4 values. Do not derive
  the timer only from `is_edible`.
- Pac-Man valid action mask: 4 values. Also use it as a hard output mask.
- Remaining normal and power pellets divided by their starting counts: 2
  values.
- Optionally, lives and level/stage if they change expert behavior.

Score is usually a consequence of past events and should not affect the best
movement, so collect it as metadata first rather than model input. Absolute
pixel positions should not be inputs; cell positions already exist spatially.
Derived BFS distances are useful for debugging the expert but should initially
remain metadata, otherwise the learned model partly depends on a path planner
at inference time.

Previous frames are optional. First train a single-state baseline. If temporal
information is later needed, use a short sequence model or stack embeddings;
do not replace the agreed 12-channel schema without versioning the dataset and
model.

## 3. Pac-Man expert: risk-aware action search

Plain BFS is ideal for computing unweighted maze distances, but "nearest
pellet by BFS" is not a sufficient expert because it ignores moving ghosts,
dead ends, power timing, and future choices. A* returns the same shortest path
more efficiently when there is one target, but changing BFS to A* does not
solve the decision-quality problem.

Use a two-level expert:

1. Precompute graph information with BFS: distances to pellets and power
   pellets, distances to junctions, dead-end depth, escape routes, and distance
   from every ghost.
2. At each Pac-Man decision, evaluate every legal first action with a short
   time-based rollout (for example 6-12 cell moves). Simulate likely ghost
   movement during the rollout and choose the action with the best worst-case
   or expected score.

A practical leaf/trajectory score is:

```text
score =
    + pellet_reward
    + power_pellet_reward_when_threatened
    + frightened_ghost_reward_if_reachable_before_timer_expires
    + distance_from_dangerous_ghosts
    + number_of_escape_routes
    - death_or_collision_penalty
    - near_future_collision_penalty
    - dead_end_depth_when_threatened
    - unnecessary_reverse_or_oscillation
    - distance_to_next_safe_pellet
```

Death must dominate all positive terms. First reject actions that cause an
immediate collision or an unavoidable collision within the rollout horizon.
Among surviving actions, prefer food progress. Only chase a frightened ghost
when its shortest intercept time, including a safety margin, is less than its
remaining frightened time. Treat a power pellet as valuable when dangerous
ghosts are close; avoid consuming it wastefully when the area is already safe.

For ghosts, predict more than their current cells:

- Use their current directions and disallow an immediate reverse when the game
  rules disallow it.
- For deterministic BFS ghosts, simulate their real controller exactly.
- If a ghost can choose several equal moves, branch over those moves for a
  conservative expert, or evaluate all choices and penalize the worst case.
- Build a danger-time map: for each cell, estimate the earliest time a
  dangerous ghost can reach it. A planned Pac-Man step at time `t` is unsafe
  when ghost arrival time is `<= t + safety_margin`.
- Detect both entities entering the same cell and edge swaps
  (`Pac-Man A->B`, ghost `B->A`) as collisions.
- Frightened ghosts contribute opportunity rather than danger until the timer
  is close to expiring; the rollout must switch them back to dangerous at the
  correct future step.

This is still supervised/imitation learning: the search is only the teacher
that produces labels. The neural model learns to imitate its decisions.

## 4. Labels and ambiguous expert decisions

Use the project action order everywhere:

```text
0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
```

Apply the Pac-Man valid-action mask before selecting the expert action and
before both training and inference. Assert that `valid_actions[label] == 1`.

Several actions can be equally good. A random tie-break creates contradictory
labels for identical states. Use a deterministic tie-break (continue straight,
then a fixed action order), and preferably store all four expert action scores
as `teacher_scores`. The first baseline can train with one hard `label`; later,
near-equal scores can become soft targets or sample weights.

## 5. Collection policy

Collect one sample at a real decision point: when Pac-Man is centered in a
cell and can select a direction. Do not collect every rendered/pixel frame.
Corridor states with only one legal action add little value; either skip most
of them or keep a small fraction. Always prioritize intersections, forced
turns, nearby dangerous ghosts, dead ends, power-pellet choices, frightened
ghost intercepts, and tunnel entrances.

Generate full episodes controlled by the expert, not unrelated random board
snapshots. Randomize maze seed/size, spawn positions where valid, ghost timing,
and power states, but preserve states that can actually occur under game
physics.

To prevent repeated data:

- Hash the discrete state fields plus the expert label and drop exact
  duplicates, or cap each hash at a small count.
- Downsample straight corridors and the majority action class.
- Keep rare danger/power/dead-end cases with higher probability.
- Report action counts and scenario counts before training.
- Split train/validation/test by complete `episode_id` or maze seed, never by
  individual adjacent frames. Otherwise nearly identical states leak into
  validation.

The first dataset should come from expert play. After the baseline works, use
DAgger-style collection: let the learned policy play, ask the expert to label
the states the policy actually visits, then add those samples and retrain. This
is imitation learning, not reward-based reinforcement learning, and it reduces
the compounding-error problem.

## 6. Pac-Man JSONL record

Use a separate file such as
`AI_arena/data/PACMAN_IMITATION_DATA.jsonl`. One line should be:

```json
{
  "schema_version": 1,
  "grid": "float/binary array [12,50,25]",
  "extra_features": "normalized float array [N]",
  "valid_actions": [true, false, true, true],
  "label": 3,
  "teacher_scores": [-20.0, -1000000.0, 2.5, 8.0],
  "episode_id": 17,
  "episode_step": 42,
  "maze_seed": 1234,
  "maze_width": 10,
  "maze_height": 10,
  "outcome": "metadata filled after episode ends"
}
```

JSONL is convenient for initial inspection and streaming. For a large dataset,
convert validated records to tensor shards (`.pt`) or another chunked binary
format so training does not repeatedly parse huge nested JSON arrays. Save a
manifest containing schema version, channel names/order, feature names/order,
normalization constants, sample counts, and collection/expert configuration.

## 7. Supervised CNN training

Create a player classifier with the existing `PacmanCNNBackbone` and one
`Linear(128, 4)` output. It should return only four action logits; an
actor-critic value output and PPO loss are not needed.

```python
logits = model(grid, extra_features)                 # [batch, 4]
masked_logits = logits.masked_fill(~valid_actions, -1e9)
loss = torch.nn.functional.cross_entropy(masked_logits, label)
```

Train with Adam/AdamW, shuffled episode-safe training data, early stopping, and
the held-out validation episodes. Track masked top-1 accuracy, illegal-action
rate (must be zero after masking), per-action recall, intersection accuracy,
danger-state accuracy, death/survival rate, pellet completion, and score during
closed-loop games. Offline accuracy alone is not enough because one wrong turn
changes all later states.

If action classes remain imbalanced after collection, use balanced sampling or
class weights. Do not hide a weak expert/model by measuring the many trivial
one-action corridor states as the main accuracy number.

## 8. Test inside the game

Add a supervised checkpoint path and controller rather than overwriting the
PPO checkpoint. Reuse `ObservationFormatter` so training and live gameplay see
the same tensors. At every centered-cell decision:

1. Build `grid`, `extra_features`, and the valid Pac-Man action mask.
2. Run the classifier in `torch.inference_mode()`.
3. Mask invalid logits with `-1e9`.
4. Use greedy `argmax` for reproducible evaluation.
5. Set `player.next_direction` to the chosen direction.

Run many headless episodes on unseen maze seeds and compare the learned player
against the expert and simple baselines (random legal action and nearest-pellet
BFS). Then use `AI_arena/player/play_player_ai.py` for visual debugging. Log
the model choice, valid actions, expert choice, confidence, ghost distances,
and eventual collision so failures can be added through DAgger collection.

## Recommended implementation order

1. Fix and test one canonical observation schema. In particular, expose real
   frightened timers; the current live formatter only receives `is_edible`.
2. Implement `PacmanExpert` with legal-action filtering, BFS distance maps,
   danger-time maps, dead-end detection, and deterministic tie-breaking.
3. Unit-test tiny hand-built mazes: blocked moves, nearest pellet, approaching
   ghost, dead end, power pellet, edible ghost, timer expiry, and edge-swap
   collision.
4. Implement a real-episode Pac-Man collector and the separate versioned JSONL
   schema.
5. Add strict dataset validation and episode/seed-based train/validation/test
   splits.
6. Add `PlayerImitationCNN` and masked cross-entropy training.
7. Add a separate supervised inference checkpoint/controller mode.
8. Evaluate on unseen seeds, inspect failures, then perform DAgger-style
   expert relabeling of policy-visited states.

The best immediate next step is **the expert and its unit tests**, not training.
Without a reliable teacher, a larger dataset or CNN only learns unreliable
labels faster.
