"""Local multiprocessing rollout collector for single-machine parallel training."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.multiprocessing as mp

from AI_arena.models.cnn_player import PlayerActorCritic
from AI_arena.player.player_env import PacmanPlayerEnv


def _init_pygame():
    """Each process needs its own pygame init."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_surface():
        pygame.display.set_mode((1, 1))


def collect_worker(
    seed: int,
    stage: int,
    steps: int,
    state_dict: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """
    Collect a rollout chunk in an isolated process.
    Returns tensors ready for concatenation on the master.
    """
    _init_pygame()

    device = torch.device("cpu")
    env = PacmanPlayerEnv(seed=seed, stage=stage, device="cpu")

    policy = PlayerActorCritic()
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()

    grids, features, valids = [], [], []
    actions, log_probs, rewards = [], [], []
    dones, values, hiddens = [], [], []

    obs = env.reset()
    hidden: torch.Tensor | None = None

    for _ in range(steps):
        grid, feats, valid = obs

        with torch.no_grad():
            logits, value, hidden = policy(grid.to(device), feats.to(device), hidden)
            masked = logits.masked_fill(~valid.to(device), -1e4)
            dist = torch.distributions.Categorical(logits=masked)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        next_obs, reward, done, info = env.step(action.item())

        grids.append(grid)
        features.append(feats)
        valids.append(valid)
        actions.append(action.cpu())
        log_probs.append(log_prob.cpu())
        rewards.append(torch.tensor([reward], dtype=torch.float32))
        dones.append(torch.tensor([done], dtype=torch.float32))
        values.append(value.squeeze(-1).cpu())
        hiddens.append(
            hidden.squeeze(0).cpu() if hidden is not None else torch.zeros(1, 128)
        )

        if done:
            obs = env.reset()
            hidden = None
        else:
            obs = next_obs

    return {
        "grids": torch.cat(grids, dim=0),
        "features": torch.cat(features, dim=0),
        "valids": torch.cat(valids, dim=0),
        "actions": torch.cat(actions, dim=0),
        "log_probs": torch.cat(log_probs, dim=0),
        "rewards": torch.cat(rewards, dim=0),
        "dones": torch.cat(dones, dim=0),
        "values": torch.cat(values, dim=0),
        "hiddens": torch.cat(hiddens, dim=0),
    }
