"""PyTorch dataset utilities for CNN training records stored as JSONL."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

CNN_CHANNEL_COUNT = 12
EXTRA_FEATURE_COUNT = 37
GHOST_COUNT = 4
ACTION_COUNT = 4
CNN_HEIGHT = 50
CNN_WIDTH = 25
EPISODE_LENGTH = 10

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

        offsets: list[int] = []
        with self.path.open("rb") as fh:
            while True:
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
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

        grid = record["grid"]
        if not isinstance(grid, list) or len(grid) != CNN_CHANNEL_COUNT:
            raise ValueError(
                f"{prefix} grid must contain {CNN_CHANNEL_COUNT} channels"
            )

        if not grid or not all(isinstance(channel, list) for channel in grid):
            raise ValueError(f"{prefix} every grid channel must be a matrix")
        if any(
            len(channel) != CNN_HEIGHT
            or any(
                not isinstance(row, list) or len(row) != CNN_WIDTH
                for row in channel
            )
            for channel in grid
        ):
            raise ValueError(
                f"{prefix} grid must have shape "
                f"[{CNN_CHANNEL_COUNT}, {CNN_HEIGHT}, {CNN_WIDTH}]"
            )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value not in (0, 1)
            for channel in grid
            for row in channel
            for value in row
        ):
            raise ValueError(f"{prefix} grid values must be binary")

        maze_width = record["maze_width"]
        maze_height = record["maze_height"]
        if (
            not isinstance(maze_width, int)
            or isinstance(maze_width, bool)
            or not 1 <= maze_width <= CNN_WIDTH
        ):
            raise ValueError(
                f"{prefix} maze_width must be an integer from 1 to {CNN_WIDTH}"
            )
        if (
            not isinstance(maze_height, int)
            or isinstance(maze_height, bool)
            or not 1 <= maze_height <= CNN_HEIGHT
        ):
            raise ValueError(
                f"{prefix} maze_height must be an integer "
                f"from 1 to {CNN_HEIGHT}"
            )
        if any(
            value != 0
            for channel in grid
            for y, row in enumerate(channel)
            for x, value in enumerate(row)
            if y >= maze_height or x >= maze_width
        ):
            raise ValueError(
                f"{prefix} grid values outside the maze dimensions "
                "must be zero"
            )

        extra = record["extra_features"]
        if not isinstance(extra, list) or len(extra) != EXTRA_FEATURE_COUNT:
            raise ValueError(
                f"{prefix} extra_features must contain "
                f"{EXTRA_FEATURE_COUNT} values"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in extra
        ):
            raise ValueError(f"{prefix} extra_features must be finite numbers")

        masks = record["valid_actions"]
        if (
            not isinstance(masks, list)
            or len(masks) != GHOST_COUNT
            or any(
                not isinstance(mask, list) or len(mask) != ACTION_COUNT
                for mask in masks
            )
        ):
            raise ValueError(
                f"{prefix} valid_actions must have shape "
                f"[{GHOST_COUNT}, {ACTION_COUNT}]"
            )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value not in (0, 1)
            for mask in masks
            for value in mask
        ):
            raise ValueError(f"{prefix} valid_actions values must be binary")
        if any(not any(mask) for mask in masks):
            raise ValueError(f"{prefix} every ghost must have a valid action")

        labels = record["labels"]
        if not isinstance(labels, list) or len(labels) != GHOST_COUNT:
            raise ValueError(f"{prefix} labels must contain 4 values")

        for ghost_index, label in enumerate(labels):
            if (
                not isinstance(label, int)
                or isinstance(label, bool)
                or not 0 <= label < ACTION_COUNT
            ):
                raise ValueError(
                    f"{prefix} label {ghost_index} must be an integer "
                    f"from 0 to {ACTION_COUNT - 1}"
                )
            if masks[ghost_index][label] != 1:
                raise ValueError(
                    f"{prefix} label {ghost_index} selects a blocked action"
                )

        episode_id = record["episode_id"]
        episode_step = record["episode_step"]
        if (
            not isinstance(episode_id, int)
            or isinstance(episode_id, bool)
            or episode_id < 0
        ):
            raise ValueError(f"{prefix} episode_id must be non-negative")
        if (
            not isinstance(episode_step, int)
            or isinstance(episode_step, bool)
            or not 0 <= episode_step < EPISODE_LENGTH
        ):
            raise ValueError(
                f"{prefix} episode_step must be from 0 to "
                f"{EPISODE_LENGTH - 1}"
            )


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


def main() -> None:
    """Load, validate, and display the CNN dataset without training a model."""

    dataset_path = Path(__file__).parent / "data" / "CNN_DATA.jsonl"
    dataset = CNNJSONLDataset(dataset_path, validate=True)

    print(f"Dataset: {dataset_path}")
    print(f"Number of samples: {len(dataset)}")

    loader = create_cnn_dataloader(
        dataset_path,
        batch_size=min(32, len(dataset)),
        shuffle=True,
        validate=True,
    )

    for batch_index, batch in enumerate(loader, start=1):
        grid, extra_features, valid_actions, labels = batch
        print(f"Batch {batch_index}")
        print(f"  grid: {tuple(grid.shape)}")
        print(f"  extra_features: {tuple(extra_features.shape)}")
        print(f"  valid_actions: {tuple(valid_actions.shape)}")
        print(f"  labels: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()
