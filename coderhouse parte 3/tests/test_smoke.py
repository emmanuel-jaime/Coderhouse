"""Pruebas funcionales del pipeline, sin depender de GPU o internet."""

import math

import pytest
import torch
from torch import nn

from main import run_experiment
from src.data import make_dataloaders
from src.evaluate import evaluate
from src.model import IrisMLP
from src.utils import get_device


def test_dataset_is_stratified_and_scaled_without_leakage():
    data = make_dataloaders(batch_size=16, seed=42)
    assert data.sizes == {"train": 105, "val": 22, "test": 23}
    train_x = data.train_loader.dataset.tensors[0].numpy()
    assert train_x.shape == (105, 4)
    assert all(abs(v) < 1e-5 for v in train_x.mean(axis=0))
    assert int(data.scaler.n_samples_seen_) == 105
    for loader in (data.train_loader, data.val_loader, data.test_loader):
        class_counts = torch.bincount(loader.dataset.tensors[1], minlength=3)
        assert (class_counts > 0).all()  # Split estratificado: todas las clases presentes.
        assert int(class_counts.max() - class_counts.min()) <= 2


def test_model_output_and_eval_do_not_modify_weights():
    data = make_dataloaders(batch_size=16)
    model = IrisMLP()
    x, _ = next(iter(data.train_loader))
    assert model(x).shape == (16, 3)
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    metrics = evaluate(model, data.val_loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert not model.training


def test_full_training_generates_outputs(tmp_path):
    result = run_experiment(epochs=3, output_dir=tmp_path, requested_device="cpu")
    assert result["best_epoch"] in (1, 2, 3)
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert math.isfinite(result["test_loss"])
    for name in ["best_model.pt", "scaler.joblib", "metrics.csv", "curvas.png", "experiment.json"]:
        assert (tmp_path / name).is_file(), f"Falta {name}"
    with (tmp_path / "metrics.csv").open(encoding="utf-8") as f:
        assert len(f.readlines()) == 4  # cabecera + 3 épocas


def test_reproducibility_on_cpu(tmp_path):
    first = run_experiment(epochs=3, output_dir=tmp_path / "a", requested_device="cpu")
    second = run_experiment(epochs=3, output_dir=tmp_path / "b", requested_device="cpu")
    assert first["best_val_loss"] == second["best_val_loss"]
    assert first["test_accuracy"] == second["test_accuracy"]
    assert (tmp_path / "a" / "metrics.csv").read_bytes() == (tmp_path / "b" / "metrics.csv").read_bytes()


def test_invalid_settings():
    assert str(get_device("cpu")) == "cpu"
    with pytest.raises(ValueError):
        make_dataloaders(batch_size=0)
    with pytest.raises(ValueError):
        run_experiment(epochs=0)
    with pytest.raises(ValueError):
        run_experiment(learning_rate=0)
    with pytest.raises(ValueError):
        get_device("tpu")
