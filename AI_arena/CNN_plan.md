# Pac-Man Ghost AI Plan

## 1. Main Idea

Use one central model to control all four ghosts.

```text
Full maze state
      ↓
CNN model
      ↓
4 action outputs
      ↓
Ghost 1: UP
Ghost 2: LEFT
Ghost 3: RIGHT
Ghost 4: DOWN
```

The model gives the next move, not the complete path.

After every movement, it reads the new game state and chooses again. All these movements together create each ghost's path.

---

## 2. Project Structure

The project can be organized like this:

```text
pacman_ai/
├── game/
│   ├── maze.py          # Maze generation & logic
│   ├── player.py        # Pac-Man logic
│   └── ghost.py         # Ghost movement logic
├── data/
│   ├── generate_data.py # Creates training samples
│   └── training_data.npz
├── ai/
│   ├── model.py         # CNN architecture
│   ├── dataset.py       # Loads data for training
│   ├── train.py         # Training loop
│   └── predict.py       # Uses trained model
├── models/
│   └── ghost_ai.pt      # Saved model weights
└── main.py              # Run the game
```

---

# Phase 1: Supervised Learning

## 3. Create a Teacher

Supervised learning needs correct answers called **labels**.

| Step                  | Teacher Type                                              | Why                                    |
| --------------------- | --------------------------------------------------------- | -------------------------------------- |
| **Step 1** (1 ghost)  | **Simple BFS or A\***                                     | One ghost has nobody to cooperate with |
| **Step 2** (4 ghosts) | **Cooperative pathfinder** (or independent BFS per ghost) | Now ghosts can work together           |


---

## 4. Input Data

Represent the maze as a multi-channel image. Each channel is a binary grid (0 or 1).
| Channel | Information      | `1` means...        | `0` means...        |
| ------- | ---------------- | ------------------- | ------------------- |
| 0       | Walls            | Wall exists         | Empty space         |
| 1       | Normal pellets   | Pellet exists       | No pellet           |
| 2       | Power pellets    | Power pellet exists | No power pellet     |
| 3       | Player position  | Player is here      | Player is not here  |
| 4       | Ghost 1 position | Ghost 1 is here     | Ghost 1 is not here |
| 5       | Ghost 2 position | Ghost 2 is here     | Ghost 2 is not here |
| 6       | Ghost 3 position | Ghost 3 is here     | Ghost 3 is not here |
| 7       | Ghost 4 position | Ghost 4 is here     | Ghost 4 is not here |


The input shape is:

```python
state_grid.shape = (8, maze_height, maze_width)
```

For example, for a `5 × 5` maze:

```python
state_grid.shape = (8, 5, 5)
```

This means:

```text
9 channels × 5 rows × 5 columns
```


### Channel 0: Walls

```text
1 1 1 1 1
1 0 0 0 1
1 0 0 0 1
1 0 0 0 1
1 1 1 1 1
```

* `1` = wall
* `0` = no wall

### Channel 1: Normal Pellets

```text
0 0 0 0 0
0 1 1 1 0
0 1 0 1 0
0 1 1 1 0
0 0 0 0 0
```

* `1` = a normal pellet exists
* `0` = no normal pellet

### Channel 2: Power Pellets

```text
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
0 0 0 1 0
0 0 0 0 0
```

* `1` = a power pellet exists
* `0` = no power pellet

### Channel 3: Player Position

```text
0 0 0 0 0
0 0 0 0 0
0 0 1 0 0
0 0 0 0 0
0 0 0 0 0
```

The value `1` shows the player’s position.

### Channel 4: Ghost 1 Position

```text
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
```

### Channel 5: Ghost 2 Position

```text
0 0 0 0 0
0 0 0 1 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
```

### Channel 6: Ghost 3 Position

```text
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
```

### Channel 7: Ghost 4 Position

```text
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 1 0
0 0 0 0 0
```

Each channel describes one type of information. Together, the nine channels describe the complete game state for the CNN.



### How the Model Understands Channel Values

The CNN does not automatically know that `1` means a wall and `0` means no wall.

We define this rule when we create the input data:

```python
wall_channel[row][col] = 1  # There is a wall
wall_channel[row][col] = 0  # There is no wall
```

The model learns the meaning during training.

For every game state, it receives:

```text
Maze channels
      +
Correct actions from BFS
```

Example:

```text
Wall in front of the ghost
        ↓
BFS chooses LEFT
        ↓
CNN learns that moving forward is not valid
```

After seeing many training examples, the model learns that:

```text
1 in the wall channel → blocked cell
0 in the wall channel → no wall
```

The model does not understand the word **wall**. It only learns patterns between input numbers and correct actions.

It is important to always use the same rules:

* Channel 1 must always represent walls.
* `1` must always mean a wall.
* `0` must always mean no wall.
* Training and prediction must use the same channel order.

If the meaning changes, the model will receive incorrect information and may choose bad movements.

## 5. Extra Information

Some information is easier to send as a normal vector.

```python
extra_data = [
    # Player direction
    0, 0, 0, 1,

    # Ghost edible states
    0, 0, 0, 0,
]
```

The player direction can use one-hot encoding:

| Encoding       | Meaning |
| -------------- | ------- |
| `[1, 0, 0, 0]` | UP      |
| `[0, 1, 0, 0]` | DOWN    |
| `[0, 0, 1, 0]` | LEFT    |
| `[0, 0, 0, 1]` | RIGHT   |

Total extra_data length: 8

---

## 6. Labels

Each ghost has four possible actions:

```python
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
```

Example label:

```python
label_actions = [3, 0, 1, 2]
```

This means:

| Ghost | Action |
|---|---|
| Ghost 1 | RIGHT |
| Ghost 2 | UP |
| Ghost 3 | DOWN |
| Ghost 4 | LEFT |

One training sample looks like this:

```python
sample = {
    "grid": state_grid,              # Shape: (8, H, W)
    "extra": extra_data,             # Direction and edible states
    "valid_actions": action_masks,   # Shape: (4, 4)
    "labels": [3, 0, 1, 2],          # Correct actions
}
```

---

## 7. Valid-Action Mask

The model must know which movements are possible.

```python
valid_actions = [
    [1, 0, 1, 1],  # Ghost 1: UP ok, DOWN blocked, LEFT ok, RIGHT ok
    [0, 1, 1, 0],  # Ghost 2: UP blocked, DOWN ok, LEFT ok, RIGHT blocked
    [1, 1, 0, 1],  # Ghost 3
    [1, 0, 0, 1],  # Ghost 4
]
```

For every ghost, the order is:

```text
[UP, DOWN, LEFT, RIGHT]
```

- `1` = valid movement
- `0` = movement blocked by a wall

The model should never choose an action marked as `0`.

---

## 8. Collect Training Data

✅ Correct approach: Generate independent random states. Each sample is a brand new scenario.
```python
for i in range(50000):  # 50,000 independent samples
    # 1. Pick random VALID positions
    player_pos = random_walkable_cell(maze)
    ghost1_pos = random_walkable_cell(maze)
    ghost2_pos = random_walkable_cell(maze)
    ghost3_pos = random_walkable_cell(maze)
    ghost4_pos = random_walkable_cell(maze)

    # 2. Build the state
    state = build_grid(maze, player_pos, [g1, g2, g3, g4])

    # 3. Ask the teacher for correct actions
    labels = teacher.choose_actions(state)

    # 4. Get valid-action masks
    masks = get_valid_actions([g1, g2, g3, g4], maze)

    # 5. Save
    save_sample(state, extra, masks, labels)
```
``Why this is better: ``
The model sees every possible situation immediately.
It learns: "No matter where I am, move toward the player."
No need to simulate movement between samples.
#### Dataset shapes for 50,000 samples:

## 9. CNN Structure

```text
Maze channels
      ↓
Convolution layers
      ↓
Feature vector
      ↓
Extra information
      ↓
Fully connected layers
      ↓
Four action heads
```

The output shape is:

```python
output.shape = (4, 4)
```

Example output:

```python
[
    [1.2, 0.4, 2.1, 5.3],  # Ghost 1
    [4.7, 1.0, 0.2, 2.4],  # Ghost 2
    [0.1, 3.8, 2.2, 1.7],  # Ghost 3
    [2.5, 1.1, 4.6, 0.3],  # Ghost 4
]
```

The highest value is selected for each ghost:

| Ghost | Selected Action |
|---|---|
| Ghost 1 | RIGHT |
| Ghost 2 | UP |
| Ghost 3 | DOWN |
| Ghost 4 | LEFT |

---

## 10. Train the Supervised Model

`Loss function:` Cross-entropy loss for each ghost, summed together.

```text
total_loss = loss(ghost1_pred, ghost1_label) +
             loss(ghost2_pred, ghost2_label) +
             loss(ghost3_pred, ghost3_label) +
             loss(ghost4_pred, ghost4_label)
```

Training process:

```text
State
  ↓
CNN prediction
  ↓
Compare prediction with BFS labels
  ↓
Calculate loss
  ↓
Update model
```

#### Evaluation metrics:
- Action accuracy (matches teacher?)
- Invalid move rate (should be 0%)
- Catch time (how fast to catch player?)
- Corridor overlap (are ghosts spreading out?)
- Generalization (performance on unseen mazes)

---


# Phase 2: Reinforcement Learning

## 11. Start From the Supervised Model

Do not create a new model.

Load the trained supervised model:

```python
rl_model.load_state_dict(
    supervised_model.state_dict()
)
```

The model already knows basic chasing.

Reinforcement learning helps it discover better cooperation.

---

## 12. Reinforcement Learning Data Format

At every reinforcement learning step, store:

```python
transition = {
    "state": state,
    "actions": [3, 0, 1, 2],
    "reward": 5.0,
    "next_state": next_state,
    "done": False,
}
```

The important reinforcement-learning data is:

```text
State
Action
Reward
Next state
Game finished or not
```

---

## 13. Rewards

Start with a simple team reward.

| Event | Reward |
|---|---:|
| A ghost catches the player | `+100` |
| Ghosts reduce their distance to the player | `+1` |
| Ghosts block different exits | `+5` |
| Two ghosts try to enter the same cell | `-5` |
| A ghost chooses an invalid movement | `-10` |
| The player escapes for another step | `-0.01` |

All ghosts can receive the same team reward when one ghost catches the player.

This encourages cooperation.

---

## 14. Reinforcement Learning Loop

```python
state = game.reset()

while not game.is_finished:
    actions = model.choose_actions(state)

    next_state, reward, done = game.step(actions)

    model.learn(
        state=state,
        actions=actions,
        reward=reward,
        next_state=next_state,
        done=done,
    )

    state = next_state
```

---

## 15. Recommended Reinforcement Learning Algorithm

For four ghosts, use a model with four action outputs:

```text
Ghost 1 action
Ghost 2 action
Ghost 3 action
Ghost 4 action
```

An algorithm such as **PPO** is suitable because it can manage several action outputs.

A simple DQN with one combined action would have:

```text
4 × 4 × 4 × 4 = 256 possible action combinations
```

This can be harder to train.

---

# Development Order

## Step 1

One fixed maze, one ghost, and supervised learning.

## Step 2

One fixed maze, four ghosts, and supervised learning.

## Step 3

Different mazes, four ghosts, and supervised learning.

## Step 4

Load the supervised model into reinforcement learning.

## Step 5

Train the model using team rewards.

## Step 6

Test the model on maze seeds that were not used during training.

## Step 7

Connect the trained model to the real Pac-Man game.

---

# Final Architecture

```text
Game
  ↓
Create maze channels and extra data
  ↓
CNN reads the full situation
  ↓
Four action heads choose four movements
  ↓
Apply valid-action masks
  ↓
Move all ghosts
  ↓
Check collisions and calculate rewards
  ↓
Read the new state
  ↓
Repeat
```
