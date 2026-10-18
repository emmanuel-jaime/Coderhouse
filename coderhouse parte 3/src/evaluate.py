"""Evaluación sin cálculo de gradientes ni actualizaciones de pesos."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        raise ValueError("El DataLoader de evaluación está vacío")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
