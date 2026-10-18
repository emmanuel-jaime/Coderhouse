"""Paso de entrenamiento y persistencia de métricas/curvas."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Compatible con servidores y entornos sin monitor.
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        # 1. Reiniciar acumulación de gradientes.
        optimizer.zero_grad(set_to_none=True)
        # 2. Forward pass: calcular logits y pérdida.
        logits = model(features)
        loss = criterion(logits, labels)
        # 3. Backpropagation: calcular gradientes vía autograd.
        loss.backward()
        # 4. Actualizar pesos con Adam.
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("El DataLoader de entrenamiento está vacío")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def save_history_csv(history: list[dict], filepath: Path) -> None:
    with filepath.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
        )
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict], filepath: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Entrenamiento")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validación")
    axes[0].set(xlabel="Época", ylabel="Cross-entropy", title="Pérdida")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Entrenamiento")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], label="Validación")
    axes[1].set(xlabel="Época", ylabel="Accuracy", title="Exactitud", ylim=(0, 1.02))
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Iris · Pipeline PyTorch", fontsize=14)
    fig.tight_layout()
    fig.savefig(filepath, dpi=160)
    plt.close(fig)
