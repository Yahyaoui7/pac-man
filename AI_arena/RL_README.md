# Moving the Ghost AI from Supervised Learning to Reinforcement Learning

This project currently trains the ghost CNN with supervised examples. The
teacher action is stored in `labels`, and `cnn_training.py` optimizes masked
cross-entropy. Reinforcement learning (RL) removes the teacher labels: the
ghosts choose actions in an environment and learn from rewards returned by the
game.

## Files and their roles

| File | RL action |
|---|---|
| `AI_arena/cnn_model.py` | Keep the convolutional model; add a value head if using PPO/actor-critic. |
| `AI_arena/pacman_ghost_env.py` | Complete the RL environment. This is the main migration file. |
| `AI_arena/cnn_training.py` | Keep for supervised training; do not use its label loss for RL. |
| `AI_arena/cnn_dataset.py` | Keep only if you still want the old supervised dataset. |
| `AI_arena/cnn_controller.py` | Keep for inference/evaluation after loading an RL checkpoint. |
| `AI_arena/data_collector/` | No longer required for RL data collection. |
| `AI_arena/rl_training.py` | Create this new file for PPO/DQN training. |

## 1. Finish the environment first

Implement the TODO sections in `pacman_ghost_env.py`.

### `_create_entities()`

Create one Pac-Man and four distinct ghosts on walkable cells. Pac-Man should
start at the centre, or use `find_player_spawn()` when the centre is blocked.
Ghosts should use valid cells nearest the four corners. Do not use a death
reset to initialize Pac-Man if that reset clears state you need; construction
already initializes `powered_mode` to `None`.

### `_get_observation()`

Return three tensors:

```text
grid          [1, 12, 50, 25]
features      [1, FEATURE_COUNT]
valid_actions [1, 4, 4]
```

Use the same channel meaning as `CNN_DATA_README.md`: walls, pellets,
Pac-Man, four ghosts, and valid cells. The action order must remain:

```text
0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
```

Apply `valid_actions` as a hard mask before sampling or selecting an action.
Never allow a policy to move through a wall.

Important: the repository currently has inconsistent feature sizes. The data
README describes 9 features, while `cnn_dataset.py` defines
`EXTRA_FEATURE_COUNT = 37` and the environment comments expect 37. Choose one
layout, update the model and all validators to match it, and use that same
layout everywhere. A simple first version is the 9 features documented in
`CNN_DATA_README.md`.

### `_move_player()`

Give Pac-Man a deterministic or random legal policy. Pac-Man is part of the
environment, not an RL-controlled action. Keep its current direction when it
is legal; otherwise choose one of its legal directions using `self.rng`.

### `_update_entities()`

Advance Pac-Man and every ghost through `MovementSystem`. This method must be
headless: do not call Pygame drawing, keyboard, or sound code.

### `_check_events()`

Implement collision and pellet checks. At minimum report:

```python
{
    "pacman_died": bool,
    "ghost_was_eaten": bool,
    "level_completed": bool,
}
```

When Pac-Man is powered, colliding ghosts should be eaten and later respawn.
When Pac-Man is not powered, a ghost collision should end the episode.

## 2. Define the RL action and reward

The action is one direction for each ghost:

```python
actions = [blinky_action, pinky_action, inky_action, clyde_action]
```

The environment already expects four actions in `step(actions)`.

Start with a shared team reward:

```text
-0.001 every step                  (encourage progress)
+10 when Pac-Man is caught
-2 when a ghost is eaten
-10 when Pac-Man completes the level
```

You can later add small shaping rewards, such as a positive value when the
ghost team gets closer to Pac-Man, but keep shaping small so catching Pac-Man
remains the objective.

## 3. Choose an RL algorithm

PPO with a shared policy is the recommended first implementation. The four
ghosts have the same action space and can share one CNN. The policy should
produce logits with shape `[batch, 4, 4]`; a value head should produce one
state value per environment state.

For each ghost, mask invalid logits before sampling:

```python
masked_logits = logits.masked_fill(~valid_actions, -1e9)
distribution = torch.distributions.Categorical(logits=masked_logits)
actions = distribution.sample()
log_probability = distribution.log_prob(actions)
```

Store the observation, action, log probability, reward, value, and done flag
for every step. Compute advantages with Generalized Advantage Estimation (GAE),
then optimize the clipped PPO objective plus value loss and entropy bonus.

## 4. Create `AI_arena/rl_training.py`

The training script should roughly follow this loop:

```python
env = PacmanGhostEnv(seed=42)
policy = GhostActorCritic().to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

for update in range(NUM_UPDATES):
    observation = env.reset()
    rollout = []

    for step in range(ROLLOUT_LENGTH):
        grid, features, valid_actions = observation
        logits, value = policy(grid, features)
        masked_logits = logits.masked_fill(~valid_actions, -1e9)
        dist = torch.distributions.Categorical(logits=masked_logits)
        actions = dist.sample()

        next_observation, reward, done, info = env.step(actions[0].tolist())
        rollout.append((observation, actions, dist.log_prob(actions),
                        value, reward, done))
        observation = env.reset() if done else next_observation

    # Compute returns/advantages, then perform PPO minibatch updates here.
```

The exact PPO implementation can be added after the environment passes a
random-action smoke test. Do not start by training; an incomplete environment
will produce meaningless rewards.

## 5. Model changes

The current `GhostCNN` can be reused as the policy backbone. For PPO, add:

```python
self.actor = nn.Linear(hidden_size, 4 * 4)
self.critic = nn.Linear(hidden_size, 1)
```

Reshape actor output to `[batch, 4, 4]`. The actor outputs action logits for
the four ghosts; the critic estimates the shared value of the state.

If using independent DQN instead, output four Q-values per ghost and store
replay transitions. PPO is usually simpler here because `env.step()` returns a
single cooperative reward for all ghosts.

## 6. What to stop using

RL does not need:

- JSONL `labels`
- BFS teacher actions from `data_collector/`
- `masked_cross_entropy()` in `cnn_training.py`
- supervised train/validation splits

The `valid_actions` mask is still required. It constrains the policy but is not
a training label.

## 7. Validation checklist

Before trusting a checkpoint:

1. `env.reset()` returns tensors with stable shapes.
2. Every sampled action is legal after masking.
3. Ghosts never spawn on walls or on the same cell.
4. `env.step()` changes positions and eventually terminates or truncates.
5. A random-policy smoke test runs for many episodes without exceptions.
6. Training reward is compared against the random-policy baseline.
7. Save the RL policy separately, for example `models/ghost_rl.pt`.

The existing `cnn_training.py` and `CNN_DATA_README.md` describe the old
supervised pipeline. This document describes the new RL path and is the guide
for completing `pacman_ghost_env.py` and adding the RL trainer.
