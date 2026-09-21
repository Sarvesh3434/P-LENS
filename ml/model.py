"""[ML] 2-layer GRU sequence classifier.

Why GRU: windows are a short temporal sequence of aggregate traffic; GRU captures
order with fewer parameters than LSTM, which matters on a CPU demo with ~thousands
of windows. Hidden 64 / 2 layers / dropout 0.2 is enough capacity for 9 features
without immediately memorising the training split.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StressGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2, n_classes: int = 3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h = self.gru(x)
        return self.head(out[:, -1, :])
