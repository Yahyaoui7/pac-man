"""
MLP Classifier Trainer
======================

A minimal PyTorch implementation for training a Multi-Layer Perceptron (MLP)
to classify feature vectors (e.g., from MLPFormatter) into discrete classes.

Modules
-------
- MLP : A configurable feedforward neural network for classification.
- Training : A reusable training wrapper with validation support.

Typical Usage
-------------
>>> trainer = Training(
...     input_size=21,
...     hidden_sizes=[32, 16],
...     output_size=4,
...     epochs=200,
...     lr=0.001,
... )
>>> trainer.fit(train_loader, test_loader, validation_interval=40)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional

# =============================================================================
# MODEL
# =============================================================================


class MLP(nn.Module):
    """
    Multi-Layer Perceptron that maps an input feature vector to class logits.

    Architecture
    ------------
    For ``hidden_sizes = [h1, h2, ..., hk]`` the network is::

        Linear(input_size -> h1) → ReLU → Linear(h1 -> h2) → ReLU → ... → Linear(hk -> output_size)

    Every hidden layer is followed by a ReLU activation. The output layer has
    **no activation** — it returns raw logits (for CrossEntropyLoss).

    Parameters
    ----------
    input_size : int
        Size of the input feature vector.
    hidden_sizes : List[int]
        Number of neurons in each hidden layer, in order.
    output_size : int
        Number of output classes (e.g., 4 for directions).

    Attributes
    ----------
    net : nn.Sequential
        The composed feedforward stack.
    """

    def __init__(
        self, input_size: int, hidden_sizes: List[int], output_size: int
    ):
        super().__init__()

        layers: List[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_size))  # Output layer

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_size)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch_size, output_size)``.
        """
        return self.net(x)


# =============================================================================
# TRAINING WRAPPER
# =============================================================================


class Training:
    """
    Reusable training engine for ``MLP`` (or any ``nn.Module``).

    Responsibilities
    ----------------
    1. Instantiate the model on the correct device (CPU / CUDA).
    2. Set up a loss function and an optimizer.
    3. Run the training loop with optional periodic validation.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input vectors.
    hidden_sizes : List[int]
        Hidden-layer widths. **Not mutated** — a copy is made internally.
    output_size : int
        Number of classes.
    epochs : int
        Number of complete passes over the training dataset.
    lr : float
        Learning rate passed to the optimizer. Controls step size during
        gradient descent.
    criterion : nn.Module, optional
        Loss function instance. Defaults to ``nn.CrossEntropyLoss()`` if not provided.
    optimizer_class : type, optional
        **Uninstantiated** optimizer class (not an instance). Defaults to
        ``torch.optim.Adam``.
    device : torch.device, optional
        Device to run computations on. Defaults to CUDA if available,
        otherwise CPU.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        epochs: int,
        lr: float,
        criterion: Optional[nn.Module] = None,
        optimizer_class: Optional[type] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Device selection
        # ------------------------------------------------------------------
        self.device: torch.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        """Computation device (CPU or CUDA)."""

        self.epochs: int = epochs
        """Total number of training epochs."""

        self.lr: float = lr
        """Optimizer learning rate."""

        # ------------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------------
        self.model: MLP = MLP(input_size, hidden_sizes, output_size).to(
            self.device
        )
        """The neural network being trained."""

        # ------------------------------------------------------------------
        # Loss function
        # ------------------------------------------------------------------
        self.criterion: nn.Module = criterion or nn.CrossEntropyLoss()
        """Loss function (default: CrossEntropyLoss)."""

        # ------------------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------------------
        opt_class: type = optimizer_class or torch.optim.Adam
        self.optimizer: torch.optim.Optimizer = opt_class(
            self.model.parameters(), lr=self.lr
        )
        """Parameter optimizer."""

    # ------------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        validation_interval: int = 40,
    ) -> None:
        """
        Execute the full training (and optional validation) loop.
        """
        for epoch in range(1, self.epochs + 1):
            # --------------------------------------------------------------
            # TRAINING PHASE
            # --------------------------------------------------------------
            self.model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            # --------------------------------------------------------------
            # VALIDATION PHASE (periodic)
            # --------------------------------------------------------------
            if test_loader is not None and epoch % validation_interval == 0:
                self.model.eval()

                total_loss: float = 0.0
                total_correct: int = 0
                total_samples: int = 0

                with torch.no_grad():
                    for x, y in test_loader:
                        x = x.to(self.device)
                        y = y.to(self.device)

                        pred = self.model(x)
                        batch_loss = self.criterion(pred, y).item()

                        _, predicted = torch.max(pred, 1)
                        total_correct += (predicted == y).sum().item()

                        total_loss += batch_loss * x.size(0)
                        total_samples += x.size(0)

                avg_test_loss: float = total_loss / total_samples
                accuracy: float = total_correct / total_samples * 100.0
                print(
                    f"Epoch [{epoch:3d}/{self.epochs}]  "
                    f"Test Loss: {avg_test_loss:.6f}  "
                    f"Accuracy: {accuracy:.2f}%"
                )


# =============================================================================
# DEMO / STAND-ALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration: Train an MLP on collected data from MLPFormatter.

    Task
    ----
    Predict one of 4 directions based on feature vectors.
    """

    import json
    import os

    # ------------------------------------------------------------------
    # Load Data from JSONL
    # ------------------------------------------------------------------
    data_path = os.path.join(
        os.path.dirname(__file__), "data", "MLP_DATA.jsonl"
    )

    features = []
    labels = []
    with open(data_path, "r") as f:
        for line in f:
            obj = json.loads(line)
            features.append(obj["features"])
            labels.append(obj["label"])

    # Convert to tensors
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    INPUT_SIZE: int = X.shape[1]
    NUM_CLASSES: int = 4  # 4 directions (UP, DOWN, LEFT, RIGHT)
    HIDDEN: List[int] = [128, 64, 32]
    BATCH_SIZE: int = 128
    EPOCHS: int = 1000
    LR: float = 0.0001

    # ------------------------------------------------------------------
    # Data Loaders
    # ------------------------------------------------------------------
    dataset_size = len(X)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = torch.utils.data.random_split(
        TensorDataset(X, y), [train_size, test_size], generator=generator
    )

    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
    )

    # ------------------------------------------------------------------
    # Initialize trainer and run
    # ------------------------------------------------------------------
    trainer: Training = Training(
        input_size=INPUT_SIZE,
        hidden_sizes=HIDDEN,
        output_size=NUM_CLASSES,
        epochs=EPOCHS,
        lr=LR,
    )

    model_path = os.path.join(os.path.dirname(__file__), "mlp_model.pth")

    if os.path.exists(model_path):
        trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device, weights_only=True))
        print(f"Loaded existing model from {model_path}")
    else:
        print("No existing model found, starting from scratch.")

    trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        validation_interval=40)

    # ------------------------------------------------------------------
    # Save the trained model
    # ------------------------------------------------------------------
    torch.save(trainer.model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
