"""Clasificador base MLP para Iris (4 variables, 3 clases)."""

from torch import nn


class IrisMLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, num_classes: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # Se devuelven logits; CrossEntropyLoss aplica internamente log-softmax.
        return self.network(x)
