"""Dataset loader for Pac-Man expert demonstrations stored as JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    CNN_HEIGHT,
    CNN_WIDTH,
)
from AI_arena.player.data.observation import PLAYER_EXTRA_FEATURE_COUNT

PlayerSample = tuple[Tensor, Tensor, Tensor, Tensor]


class PlayerImitationDataset(Dataset[PlayerSample]):
    def __init__(self, path: str | Path, indices: list[int] | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Player imitation dataset not found: {self.path}")
        self.offsets: list[int] = []
        self.episode_ids: list[int] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                self._validate(record, len(self.offsets))
                self.offsets.append(offset)
                self.episode_ids.append(int(record["episode_id"]))
        if indices is not None:
            self.offsets = [self.offsets[index] for index in indices]
            self.episode_ids = [self.episode_ids[index] for index in indices]
        if not self.offsets:
            raise ValueError(f"Player imitation dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> PlayerSample:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            record = json.loads(stream.readline())
        return (
            torch.tensor(record["grid"], dtype=torch.float32),
            torch.tensor(record["extra_features"], dtype=torch.float32),
            torch.tensor(record["valid_actions"], dtype=torch.bool),
            torch.tensor(record["label"], dtype=torch.long),
        )

    @staticmethod
    def _validate(record: Any, index: int) -> None:
        required = {
            "schema_version", "grid", "extra_features", "valid_actions",
            "label", "teacher_scores", "episode_id", "episode_step",
        }
        if not isinstance(record, dict) or required.difference(record):
            raise ValueError(f"Invalid player imitation record {index}: missing fields")
        grid = record["grid"]
        if len(grid) != CNN_CHANNEL_COUNT or any(len(c) != CNN_HEIGHT for c in grid):
            raise ValueError(f"Invalid grid shape in record {index}")
        if any(any(len(row) != CNN_WIDTH for row in channel) for channel in grid):
            raise ValueError(f"Invalid grid width in record {index}")
        if len(record["extra_features"]) != PLAYER_EXTRA_FEATURE_COUNT:
            raise ValueError(f"Invalid extra feature count in record {index}")
        if len(record["valid_actions"]) != ACTION_COUNT:
            raise ValueError(f"Invalid action mask in record {index}")
        label = int(record["label"])
        if not 0 <= label < ACTION_COUNT or not record["valid_actions"][label]:
            raise ValueError(f"Illegal label in record {index}")
