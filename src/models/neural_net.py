"""PyTorch deep learning model for keystroke behavioral classification."""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for PyTorch reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KeystrokeMLP(nn.Module):
    """Feed-forward neural network for keystroke dynamics subject classification."""

    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via Softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = RANDOM_SEED,
) -> Tuple[KeystrokeMLP, Dict[str, float]]:
    """Train the PyTorch KeystrokeMLP classifier with validation monitoring.

    Args:
        X_train: Scaled training features.
        y_train: Encoded training labels.
        X_val: Scaled validation/test features.
        y_val: Encoded validation/test labels.
        num_classes: Total number of subject classes.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        learning_rate: Adam optimizer learning rate.
        seed: Random seed.

    Returns:
        Tuple of (trained_model, final_evaluation_metrics).
    """
    set_seed(seed)
    input_dim = X_train.shape[1]
    model = KeystrokeMLP(input_dim=input_dim, num_classes=num_classes)

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.long)
    val_tensor_x = torch.tensor(X_val, dtype=torch.float32)
    val_tensor_y = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_weights = None
    patience = 8
    patience_counter = 0

    logger.info(f"Training PyTorch MLP (input_dim={input_dim}, classes={num_classes}, epochs={epochs})...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        train_loss = total_loss / len(X_train)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor_x)
            val_loss = criterion(val_outputs, val_tensor_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Final evaluation on validation set
    metrics = evaluate_neural_network(model, X_val, y_val)
    logger.info(
        f"PyTorch Neural Network -> Accuracy: {metrics['accuracy']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}"
    )

    return model, metrics


def evaluate_neural_network(
    model: KeystrokeMLP,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate trained PyTorch KeystrokeMLP on test features."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits = model(x_tensor)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, average="macro", zero_division=0))
    rec = float(recall_score(y_test, preds, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }
