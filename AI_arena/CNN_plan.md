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
pac_man/
├── game/
├── data/
│   └── training_data.npy
├── ai/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── predict.py
└── pac_man.py
```

### Main Roles

| File | Role |
|---|---|
| `model.py` | Defines the CNN architecture |
| `dataset.py` | Loads maze channels, extra information, and labels |
| `train.py` | Trains the model and saves its weights |
| `predict.py` | Uses the trained model to choose directions |
| `training_data.npy` | Stores the supervised-learning samples |
| `pac_man.py` | Starts and runs the game |

A more detailed version can look like this:

```text
pac_man/
├── game/
│   ├── game.py
│   ├── maze.py
│   ├── ghost.py
│   └── player.py
├── data/
│   ├── training_data.npz
│   └── generate_data.py
├── ai/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── predict.py
│   └── rewards.py
├── models/
│   └── ghost_ai.pt
└── pac_man.py
```

---

# Phase 1: Supervised Learning

## 3. Create a Teacher

Supervised learning needs correct answers called **labels**.

Use a cooperative BFS or A* system as a teacher:

```text
Game state
    ↓
Cooperative BFS/A*
    ↓
Good action for every ghost
    ↓
Save state + actions
```

The BFS or A* system is used only to create training data.

After training, the neural network controls the ghosts.

---

## 4. Input Data

Represent the maze using separate channels.

| Channel   | Information      |
| --------- | ---------------- |
| Channel 0 | Valid maze cells |
| Channel 1 | Walls            |
| Channel 2 | Normal pellets   |
| Channel 3 | Power pellets    |
| Channel 4 | Player position  |
| Channel 5 | Ghost 1 position |
| Channel 6 | Ghost 2 position |
| Channel 7 | Ghost 3 position |
| Channel 8 | Ghost 4 position |

The input shape is:

```python
state_grid.shape = (9, maze_height, maze_width)
```

For example, for a `5 × 5` maze:

```python
state_grid.shape = (9, 5, 5)
```

This means:

```text
9 channels × 5 rows × 5 columns
```

### Channel 0: Valid Maze Cells

```text
0 0 0 0 0
0 1 1 1 0
0 1 1 1 0
0 1 1 1 0
0 0 0 0 0
```

* `1` = the cell is walkable
* `0` = the cell is not walkable

### Channel 1: Walls

```text
1 1 1 1 1
1 0 0 0 1
1 0 0 0 1
1 0 0 0 1
1 1 1 1 1
```

* `1` = wall
* `0` = no wall

### Channel 2: Normal Pellets

```text
0 0 0 0 0
0 1 1 1 0
0 1 0 1 0
0 1 1 1 0
0 0 0 0 0
```

* `1` = a normal pellet exists
* `0` = no normal pellet

### Channel 3: Power Pellets

```text
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
0 0 0 1 0
0 0 0 0 0
```

* `1` = a power pellet exists
* `0` = no power pellet

### Channel 4: Player Position

```text
0 0 0 0 0
0 0 0 0 0
0 0 1 0 0
0 0 0 0 0
0 0 0 0 0
```

The value `1` shows the player’s position.

### Channel 5: Ghost 1 Position

```text
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
```

### Channel 6: Ghost 2 Position

```text
0 0 0 0 0
0 0 0 1 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
```

### Channel 7: Ghost 3 Position

```text
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0
0 1 0 0 0
0 0 0 0 0
```

### Channel 8: Ghost 4 Position

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

```python
UP    = [1, 0, 0, 0]
DOWN  = [0, 1, 0, 0]
LEFT  = [0, 0, 1, 0]
RIGHT = [0, 0, 0, 1]
```

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
    "grid": state_grid,              # Shape: (7, H, W)
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
    [1, 0, 1, 1],  # Ghost 1
    [0, 1, 1, 0],  # Ghost 2
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

At every game step:

```python
state = game.get_ai_state()

expert_actions = cooperative_pathfinder.choose_actions(
    state
)

save_sample(
    state=state,
    labels=expert_actions,
)

game.step(expert_actions)
```

Generate data using:

- Different player positions
- Different ghost positions
- Different player directions
- Different maze seeds
- Situations where ghosts are close together
- Situations where ghosts need different paths

Save large arrays using `.npz` or `.pt`, not normal JSON.

Example:

```text
training_data.npz
```

---

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

Use cross-entropy loss for every ghost.

```text
Total loss =
Ghost 1 loss
+ Ghost 2 loss
+ Ghost 3 loss
+ Ghost 4 loss
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

Evaluate the model using:

- Action accuracy
- Invalid movement rate
- Average time needed to catch the player
- Number of ghosts using the same corridor
- Performance on unseen mazes

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
