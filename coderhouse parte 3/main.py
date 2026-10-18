"""Entrada principal: entrenar, validar y evaluar una vez sobre el test reservado.

Ejemplo: python main.py --epochs 50 --learning-rate 0.001 --batch-size 16
"""

import argparse
import json
from pathlib import Path
import platform

import joblib
import torch
from torch import nn

from src.data import make_dataloaders
from src.evaluate import evaluate
from src.model import IrisMLP
from src.train import plot_history, save_history_csv, train_one_epoch
from src.utils import get_device, set_seed

PROJECT_ROOT = Path(__file__).resolve().parent


def run_experiment(
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    seed: int = 42,
    output_dir: str | Path = "outputs",
    requested_device: str = "auto",
) -> dict:
    """Realiza un experimento reproducible; devuelve y guarda sus resultados."""
    if epochs <= 0:
        raise ValueError("epochs debe ser mayor que cero")
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser mayor que cero")
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero")

    set_seed(seed)
    device = get_device(requested_device)
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)

    data = make_dataloaders(batch_size=batch_size, seed=seed)
    model = IrisMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Dispositivo: {device} | PyTorch: {torch.__version__}")
    print(f"Muestras: {data.sizes} | Semilla: {seed}")

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    checkpoint_path = destination / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, data.val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": best_val_loss},
                checkpoint_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Época {epoch:3d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | train_acc={train_metrics['accuracy']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.3f}"
            )

    # El test NO participa de la elección del modelo ni de los hiperparámetros.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, data.test_loader, criterion, device)

    save_history_csv(history, destination / "metrics.csv")
    plot_history(history, destination / "curvas.png")
    joblib.dump(data.scaler, destination / "scaler.joblib")

    result = {
        "framework": "PyTorch",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(device),
        "dataset": "Iris (sklearn.datasets.load_iris)",
        "model": "MLP: Linear(4,16) -> ReLU -> Linear(16,3)",
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "split": data.sizes,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "first_train_loss": history[0]["train_loss"],
        "last_train_loss": history[-1]["train_loss"],
        "first_val_loss": history[0]["val_loss"],
        "last_val_loss": history[-1]["val_loss"],
        "features": data.feature_names,
        "classes": data.target_names,
    }
    (destination / "experiment.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Mejor época: {best_epoch} | test_loss={test_metrics['loss']:.4f} | test_accuracy={test_metrics['accuracy']:.3f}")
    print(f"Resultados guardados en: {destination}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpoint PyTorch: Iris MLP")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        requested_device=args.device,
        output_dir=args.output_dir,
    )
