"""PyTorch dataset utilities for CNN training records stored as JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from AI_arena.data.constants import (
    ACTION_COUNT,
    CNN_CHANNEL_COUNT,
    CNN_HEIGHT,
    CNN_WIDTH,
    EXTRA_FEATURE_COUNT,
    GHOST_COUNT,
)

CNNSample = tuple[Tensor, Tensor, Tensor, Tensor]


class CNNJSONLDataset(Dataset[CNNSample]):
    """Read CNN samples lazily while keeping only line offsets in memory."""

    def __init__(self, path: str | Path, validate: bool = True) -> None:
        self.path = Path(path)
        self.validate = validate
        self._offsets = self._index_records()

        if not self._offsets:
            raise ValueError(f"CNN dataset is empty: {self.path}")

    def _index_records(self) -> list[int]:
        if not self.path.is_file():
            raise FileNotFoundError(f"CNN dataset does not exist: {self.path}")

        file_size = self.path.stat().st_size
        print(f"Indexing dataset ({file_size / 1e6:.0f} MB) ...", flush=True)
        offsets: list[int] = []
        with self.path.open("rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        print(f"Indexed {len(offsets):,} records. Starting training...", flush=True)
        return offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> CNNSample:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("CNN dataset index out of range")

        with self.path.open("rb") as fh:
            fh.seek(self._offsets[index])
            raw_line = fh.readline()

        try:
            record = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"Invalid JSON in {self.path} at record {index}"
            ) from exc

        if self.validate:
            self._validate_record(record, index)

        return (
            torch.tensor(record["grid"], dtype=torch.float32),
            torch.tensor(record["extra_features"], dtype=torch.float32),
            torch.tensor(record["valid_actions"], dtype=torch.bool),
            torch.tensor(record["labels"], dtype=torch.long),
        )

    def _validate_record(self, record: Any, index: int) -> None:
        prefix = f"Invalid CNN record {index} in {self.path}:"
        required = {
            "grid",
            "maze_width",
            "maze_height",
            "extra_features",
            "valid_actions",
            "labels",
            "episode_id",
            "episode_step",
        }

        if not isinstance(record, dict):
            raise ValueError(f"{prefix} expected a JSON object")

        missing = required.difference(record)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{prefix} missing field(s): {names}")


def create_cnn_dataloader(
    path: str | Path,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    validate: bool = True,
    num_workers: int = 0,
) -> DataLoader[CNNSample]:
    """Create a DataLoader for fixed-size, directly batchable CNN samples."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    dataset = CNNJSONLDataset(path, validate=validate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield raw records, primarily for checks that do not require PyTorch."""

    with Path(path).open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number} in {path}"
                )
            yield record
