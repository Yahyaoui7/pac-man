# Adversarial Multi-Agent Training (Pac-Man vs Ghosts)

This module contains the infrastructure for training both Pac-Man and the Ghosts simultaneously using Reinforcement Learning (MARL).

## Current Architecture

### 1. The Environment (`adversarial_env.py`)
This environment overrides the standard `PacmanPlayerEnv`. Instead of controlling the ghosts with hardcoded BFS logic, it accepts neural network actions for both Pac-Man and all 4 ghosts simultaneously.

### 2. The Training Loop (`adversarial_training.py`)
To prevent the models from crashing mathematically by constantly changing the rules on each other, we use **Alternating Training**:
* **Phase 1:** Freeze the Ghosts' weights. Train Pac-Man using PPO so he learns how to beat the current ghost strategy.
* **Phase 2:** Freeze Pac-Man's weights. Train the Ghosts using PPO so they learn how to counter Pac-Man's new tricks.

## 🚀 UPCOMING FEATURE: Custom Ghost Reward System

Currently, the ghosts use a **Zero-Sum** reward system. They get penalized whenever Pac-Man succeeds (e.g., if Pac-Man gets +10 points, the ghosts get -5 points). 

**The Plan:** We will build a dedicated `ghost_rewards.py` to give the ghosts their own complex reward math to teach them advanced team strategies.

### Proposed Ghost Rewards (`ghost_rewards.py`)

1. **The Pincer Attack Reward (+5 points)**
   * If two ghosts are approaching Pac-Man from completely different directions (e.g., trapping him in a hallway), they will get a bonus.
   * *Goal:* Teach them to cut off escape routes instead of just blindly chasing him in a single-file line.

2. **The Spacing Penalty (-2 points)**
   * If three or more ghosts are clustered in the exact same 3x3 grid, they will lose points.
   * *Goal:* Prevent all 4 ghosts from stacking on top of each other. Teach them to spread out and cover the map.

3. **The Chase Reward (+0.5 points)**
   * If a ghost's action moves it physically closer to Pac-Man, it gets a micro-reward. 
   * *Goal:* Encourage aggressive hunting behavior.

4. **The Ultimate Goal (+50 points)**
   * A massive shared team bonus if they successfully catch and kill Pac-Man.
   * *Goal:* The final objective of the game.

5. **The Fear Penalty (-10 points)**
   * A penalty if a ghost is eaten by a powered-up Pac-Man.
   * *Goal:* Teach the ghosts to scatter and hide when the power pellet is active.

By implementing this custom reward calculator, the ghosts will evolve from mindless chasers into a highly coordinated team of hunters!
