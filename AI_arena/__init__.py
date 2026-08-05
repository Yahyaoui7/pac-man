"""AI Arena package for Pac-Man and Ghost neural network models, datasets, environments, and training."""

# from AI_arena.data.constants import (
#     ACTION_COUNT,
#     CNN_CHANNEL_COUNT,
#     CNN_HEIGHT,
#     CNN_WIDTH,
#     EPISODE_LENGTH,
#     EXTRA_FEATURE_COUNT,
#     GHOST_COUNT,
# )
# from AI_arena.data.dataset import CNNJSONLDataset, create_cnn_dataloader
from AI_arena.ghosts.ghost_controller import CNNGhostController
from AI_arena.ghosts.ghost_env import PacmanGhostEnv
from AI_arena.ghosts.ghost_training import train as train_ghost_cnn
from AI_arena.models.cnn_backbone import PacmanCNNBackbone
from AI_arena.models.cnn_ghost import GhostCNN
from AI_arena.models.cnn_player import PlayerActorCritic, PlayerImitationCNN
from AI_arena.player.player_controller import CNNPlayerController
from AI_arena.player.player_env import PacmanPlayerEnv

__all__ = [
    "CNN_CHANNEL_COUNT",
    "EXTRA_FEATURE_COUNT",
    "GHOST_COUNT",
    "ACTION_COUNT",
    "CNN_HEIGHT",
    "CNN_WIDTH",
    "EPISODE_LENGTH",
    "CNNJSONLDataset",
    "create_cnn_dataloader",
    "PacmanCNNBackbone",
    "GhostCNN",
    "PlayerActorCritic",
    "PlayerImitationCNN",
    "PacmanPlayerEnv",
    "CNNPlayerController",
    "PacmanGhostEnv",
    "CNNGhostController",
    "train_ghost_cnn",
]
