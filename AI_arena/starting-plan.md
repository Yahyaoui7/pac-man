# AI Training Preparation Roadmap

## 1. Refactor the Game Engine

The current game is designed for gameplay. Before introducing AI, the engine must be adapted to support automated training.

### Objectives

* Separate the game simulation from rendering.
* Allow the game to run without displaying graphics.
* Expose the complete game state at every step.
* Allow external agents (AI models) to control Pacman and the ghosts.
* Support automatic reset and episode generation.
* Allow configurable maze size, seed, and initial entity positions.

---

## 2. Dataset Generation

Before reinforcement learning, build a dataset using the existing BFS-based behaviors.

The objective is to automatically generate a large number of game states by:

* Generating random mazes.
* Randomizing Pacman and ghost spawn positions.
* Running scripted behaviors (chasing and escaping).
* Recording the observations and the corresponding expert actions.

This dataset will be used for experimentation and as a possible starting point for model pre-training.

### State Representations

Two observation formats will be investigated.

### Representation 1 — CNN

Represent the game state as a spatial image composed of multiple channels.

Possible channels include:

* Maze walls
* Pellets
* Power pellets
* Pacman position
* Ghost positions
* Additional game entities if required

The objective is to allow the model to learn spatial relationships directly from the maze.

---

### Representation 2 — MLP

Represent the same state using structured numerical features.

Possible features include:

* Entity positions
* Entity directions
* Remaining pellets
* Power mode information
* Local movement availability
* Distances to important objectives
* Additional engineered features

The objective is to compare handcrafted state representations with learned spatial representations.

---

## 3. Reinforcement Learning Environment

Transform the game into an environment suitable for Reinforcement Learning.

The environment should provide:

* Observation
* Action execution
* Reward calculation
* Episode termination
* Environment reset

The reward system should remain configurable so it can evolve during experimentation.

---

## 4. Progressive Training Strategy

Training will begin with simplified objectives before introducing the complete game mechanics.

### Pacman Training

Stage 1

* Learn basic movement.
* Collect pellets.
* Optimize navigation.

---

Stage 2

* Collect pellets while avoiding static ghosts.

---

Stage 3

* Collect pellets while avoiding ghosts controlled by scripted algorithms.

---

Stage 4

* Compete against learned ghost policies.

---

### Ghost Training

Stage 1

* Learn maze navigation.
* Reach moving targets.

---

Stage 2

* Learn to chase Pacman controlled by scripted behavior.

---

Stage 3

* Learn frightened (escape) behavior when Pacman has a power pellet.

---

Stage 4

* Learn against trained Pacman agents.

---

## Initial Experiment

The first objective is **not** to build the strongest possible AI.

The initial experiment aims to answer the following questions:

* Can the agents successfully learn the basic game mechanics?
* How does a CNN-based representation compare with an MLP-based representation?
* Which representation learns faster?
* Which representation produces better gameplay?
* Which representation provides the best balance between performance and model size?
