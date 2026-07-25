"""Train and save the ghost CNN using the JSONL dataset."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from AI_arena.cnn_dataset import (
    ACTION_COUNT,
    EPISODE_LENGTH,
    CNNJSONLDataset,
)
from AI_arena.cnn_model import GhostCNN

DEFAULT_DATASET_PATH = Path(__file__).parent / "data" / "CNN_DATA.jsonl"
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "ghost_ai.pt"


def masked_cross_entropy(
    logits: Tensor,
    valid_actions: Tensor,
    labels: Tensor,
) -> Tensor:
    """Calculate action loss after excluding movements blocked by walls."""

    masked_logits = logits.masked_fill(~valid_actions, float("-inf"))
    return nn.functional.cross_entropy(
        masked_logits.reshape(-1, ACTION_COUNT),
        labels.reshape(-1),
    )


def train(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    seed: int = 42,
    patience: int = 5,
) -> GhostCNN:
    """Train a GhostCNN and save its state dictionary."""

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CNNJSONLDataset(dataset_path, validate=True)
    episode_count = len(dataset) // EPISODE_LENGTH
    if episode_count < 2 or len(dataset) % EPISODE_LENGTH:
        raise ValueError(
            f"dataset must contain complete {EPISODE_LENGTH}-sample episodes"
        )
    validation_episodes = max(1, round(episode_count * validation_fraction))
    validation_samples = validation_episodes * EPISODE_LENGTH
    split_index = len(dataset) - validation_samples
    train_loader = DataLoader(
        Subset(dataset, range(split_index)),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        Subset(dataset, range(split_index, len(dataset))),
        batch_size=batch_size,
        shuffle=False,
    )
    model = GhostCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_accuracy = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        correct = 0
        prediction_count = 0

        for grid, extra_features, valid_actions, labels in train_loader:
            grid = grid.to(device)
            extra_features = extra_features.to(device)
            valid_actions = valid_actions.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(grid, extra_features)
            loss = masked_cross_entropy(logits, valid_actions, labels)
            loss.backward()
            optimizer.step()

            current_batch_size = grid.shape[0]
            total_loss += loss.item() * current_batch_size
            sample_count += current_batch_size
            predictions = logits.masked_fill(
                ~valid_actions,
                float("-inf"),
            ).argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            prediction_count += labels.numel()

        average_loss = total_loss / sample_count
        train_accuracy = correct / prediction_count
        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            device,
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"Epoch {epoch:03d}/{epochs:03d} - "
            f"train loss: {average_loss:.6f} acc: {train_accuracy:.2%} - "
            f"val loss: {validation_loss:.6f} "
            f"acc: {validation_accuracy:.2%}"
        )
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping after {epoch} epochs; validation accuracy "
                f"has not improved for {patience} epochs."
            )
            break

    model.load_state_dict(best_weights)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    print(
        f"Saved best model weights to {destination} "
        f"(epoch {best_epoch}, validation accuracy: "
        f"{best_validation_accuracy:.2%})"
    )
    return model


def evaluate(
    model: GhostCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Return masked validation loss and per-ghost action accuracy."""

    model.eval()
    total_loss = 0.0
    sample_count = 0
    correct = 0
    prediction_count = 0
    with torch.inference_mode():
        for grid, extra_features, valid_actions, labels in loader:
            grid = grid.to(device)
            extra_features = extra_features.to(device)
            valid_actions = valid_actions.to(device)
            labels = labels.to(device)
            logits = model(grid, extra_features)
            loss = masked_cross_entropy(logits, valid_actions, labels)
            current_batch_size = grid.shape[0]
            total_loss += loss.item() * current_batch_size
            sample_count += current_batch_size
            predictions = logits.masked_fill(
                ~valid_actions,
                float("-inf"),
            ).argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            prediction_count += labels.numel()
    return total_loss / sample_count, correct / prediction_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(
        dataset_path=args.dataset,
        model_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
