"""Configuración de dispositivos y semillas reproducibles."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija semillas y solicita algoritmos deterministas cuando existen.

    Nota: versiones de librerías y tipos de hardware distintos pueden producir
    pequeñas diferencias incluso si la semilla se mantiene constante.
    """
    if seed < 0:
        raise ValueError("seed debe ser un entero no negativo")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(requested: str = "auto") -> torch.device:
    """Elige CUDA, luego Apple MPS y por último CPU; valida opciones manuales."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Dispositivo desconocido: {requested}")

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if requested == "auto":
        if has_cuda:
            return torch.device("cuda")
        if has_mps:
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not has_cuda:
        raise RuntimeError("CUDA no disponible: comprueba la instalación de PyTorch y sus drivers")
    if requested == "mps" and not has_mps:
        raise RuntimeError("MPS no disponible: requiere macOS/Apple Silicon compatible")
    return torch.device(requested)
