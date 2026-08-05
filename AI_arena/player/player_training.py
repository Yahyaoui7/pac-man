"""Supervised imitation training for the Pac-Man player."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from AI_arena.models.cnn_player import PlayerImitationCNN
from AI_arena.player.imitation_dataset import PlayerImitationDataset
from AI_arena.player.observation import PLAYER_EXTRA_FEATURE_COUNT
from AI_arena.player.player_collector import (
    DEFAULT_DATASET_PATH,
    collect_demonstrations,
)

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "player_sl.pt"


def _checkpoint_path(model_path: str | Path) -> Path:
    destination = Path(model_path)
    return destination.with_name(f"{destination.stem}_checkpoint.pt")


def _best_model_path(model_path: str | Path) -> Path:
    destination = Path(model_path)
    return destination.with_name(f"{destination.stem}_best.pt")


def _save_training_state(
    path: Path,
    model: PlayerImitationCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_accuracy: float,
    stale_epochs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "stale_epochs": stale_epochs,
        },
        path,
    )


def _episode_split(
    dataset: PlayerImitationDataset,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    episodes = sorted(set(dataset.episode_ids))
    if len(episodes) < 2:
        raise ValueError("Dataset needs at least two complete episodes")
    random.Random(seed).shuffle(episodes)
    validation_count = max(1, round(len(episodes) * validation_fraction))
    validation_episodes = set(episodes[:validation_count])
    train_indices = [
        i
        for i, episode in enumerate(dataset.episode_ids)
        if episode not in validation_episodes
    ]
    validation_indices = [
        i
        for i, episode in enumerate(dataset.episode_ids)
        if episode in validation_episodes
    ]
    return train_indices, validation_indices


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.inference_mode():
        for grid, features, valid, labels in loader:
            grid = grid.to(device)
            features = features.to(device)
            valid = valid.to(device)
            labels = labels.to(device)
            logits = model(grid, features).masked_fill(~valid, -1e9)
            loss = nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.numel()
    return total_loss / count, correct / count


def train_player_supervised(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.15,
    patience: int = 5,
    seed: int = 42,
    resume: bool = False,
    log_interval: int = 10,
) -> PlayerImitationCNN:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if patience < 1:
        raise ValueError("patience must be at least 1")
    if log_interval < 1:
        raise ValueError("log_interval must be at least 1")
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading and validating dataset: {dataset_path}", flush=True)
    complete = PlayerImitationDataset(dataset_path)
    print(
        f"Loaded {len(complete)} samples from "
        f"{len(set(complete.episode_ids))} episodes",
        flush=True,
    )
    train_indices, validation_indices = _episode_split(
        complete, validation_fraction, seed
    )
    # Reuse the validated dataset. Constructing two more dataset instances
    # would parse the entire (potentially multi-gigabyte) JSONL file twice.
    train_data = Subset(complete, train_indices)
    validation_data = Subset(complete, validation_indices)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    model = PlayerImitationCNN(PLAYER_EXTRA_FEATURE_COUNT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_weights = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    stale_epochs = 0
    completed_epoch = 0
    checkpoint = _checkpoint_path(model_path)
    best_destination = _best_model_path(model_path)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    if resume:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot resume: training checkpoint not found: {checkpoint}"
            )
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        completed_epoch = int(saved["epoch"])
        best_accuracy = float(saved.get("best_accuracy", -1.0))
        previous_stale_epochs = int(saved.get("stale_epochs", 0))
        # A resume command starts a new training session. Carrying an exhausted
        # early-stopping counter into it can stop the requested run after one
        # epoch, even when the user explicitly requested more training.
        stale_epochs = 0
        if best_destination.is_file():
            best_weights = torch.load(
                best_destination,
                map_location=device,
                weights_only=True,
            )
        else:
            best_weights = copy.deepcopy(model.state_dict())
        print(
            f"Resumed supervised training from epoch {completed_epoch} "
            f"(reset early-stopping counter from {previous_stale_epochs})"
        )

    print(
        f"Supervised Pac-Man training on {device}: "
        f"{len(train_data)} train / {len(validation_data)} validation",
        flush=True,
    )
    first_epoch = completed_epoch + 1
    final_epoch = completed_epoch + epochs
    print(
        "Press Ctrl+C to stop safely and save the current training state.",
        flush=True,
    )
    try:
        for epoch in range(first_epoch, final_epoch + 1):
            model.train()
            loss_sum = 0.0
            correct = 0
            count = 0
            batch_total = len(train_loader)
            for batch_index, (grid, features, valid, labels) in enumerate(
                train_loader,
                start=1,
            ):
                grid = grid.to(device)
                features = features.to(device)
                valid = valid.to(device)
                labels = labels.to(device)
                logits = model(grid, features).masked_fill(~valid, -1e9)
                loss = nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item() * labels.numel()
                correct += (logits.argmax(dim=1) == labels).sum().item()
                count += labels.numel()
                if (
                    batch_index % log_interval == 0
                    or batch_index == batch_total
                ):
                    print(
                        f"Epoch {epoch:03d} | batch "
                        f"{batch_index:04d}/{batch_total:04d} | "
                        f"loss={loss_sum/count:.5f} | "
                        f"acc={correct/count:.2%}",
                        flush=True,
                    )
            val_loss, val_accuracy = _evaluate(
                model, validation_loader, device
            )
            completed_epoch = epoch
            print(
                f"Epoch {epoch:03d}: train loss={loss_sum/count:.5f} "
                f"acc={correct/count:.2%} | val loss={val_loss:.5f} "
                f"acc={val_accuracy:.2%}"
            )
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, best_destination)
                stale_epochs = 0
            else:
                stale_epochs += 1
            _save_training_state(
                checkpoint,
                model,
                optimizer,
                completed_epoch,
                best_accuracy,
                stale_epochs,
            )
            # Keep the live-game checkpoint usable after every epoch.
            torch.save(model.state_dict(), Path(model_path))
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs")
                break
    except KeyboardInterrupt:
        destination = Path(model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), destination)
        _save_training_state(
            checkpoint,
            model,
            optimizer,
            completed_epoch,
            best_accuracy,
            stale_epochs,
        )
        print(
            "\nTraining stopped safely. Current model saved to "
            f"{destination}; "
            f"resume state saved to {checkpoint}."
        )
        return model

    model.load_state_dict(best_weights)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination)
    print(
        f"Saved supervised player to {destination} "
        f"(validation accuracy {best_accuracy:.2%})"
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--collect-samples", type=int, default=0)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print progress every N training batches",
    )
    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--resume",
        action="store_true",
        help="Continue from player_sl_checkpoint.pt",
    )
    start_group.add_argument(
        "--fresh",
        action="store_true",
        help="Start from new random weights (the default)",
    )
    args = parser.parse_args()
    if args.collect_samples:
        collect_demonstrations(
            args.collect_samples,
            args.dataset,
            stage=args.stage,
            seed=args.seed,
            horizon=args.horizon,
        )
    train_player_supervised(
        args.dataset,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
