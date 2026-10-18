"""Carga Iris y separa datos evitando fuga de información (data leakage)."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    sizes: dict[str, int]
    feature_names: list[str]
    target_names: list[str]


def _to_loader(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    # num_workers=0 evita problemas de multiprocessing en Windows y notebooks.
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0
    )


def make_dataloaders(batch_size: int = 16, seed: int = 42) -> DataBundle:
    """Split estratificado train/val/test (70/15/15) y escalado solo con train."""
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero")

    iris = load_iris()
    X = np.asarray(iris.data, dtype=np.float32)
    y = np.asarray(iris.target, dtype=np.int64)

    # Etapa 1: reservar el 30% antes de transformar características.
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    # Etapa 2: dividir el holdout en 15% validación y 15% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.50, random_state=seed, stratify=y_holdout
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return DataBundle(
        train_loader=_to_loader(X_train, y_train, batch_size=batch_size, shuffle=True, seed=seed),
        val_loader=_to_loader(X_val, y_val, batch_size=batch_size, shuffle=False, seed=seed),
        test_loader=_to_loader(X_test, y_test, batch_size=batch_size, shuffle=False, seed=seed),
        scaler=scaler,
        sizes={"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        feature_names=list(iris.feature_names),
        target_names=list(iris.target_names),
    )
