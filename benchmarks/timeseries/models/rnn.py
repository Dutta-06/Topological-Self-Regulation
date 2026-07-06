"""
LSTM / GRU baselines, shared encoder with task-specific heads.

Forecasting: multivariate multi-horizon target (pred_len steps, all
channels), read from the final hidden state — matches data/etth.py and
data/electricity.py's (B, pred_len, C) target shape.

Classification: logits over num_classes, read from the final hidden state.
"""

import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        cell_type: str = "lstm",
        dropout: float = 0.1,
    ):
        super().__init__()
        cell_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = cell_cls(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> last hidden state (B, hidden_size)."""
        out, _ = self.rnn(x)
        return out[:, -1, :]


class RNNForecaster(nn.Module):
    def __init__(self, input_size: int, pred_len: int, hidden_size: int = 64, num_layers: int = 2,
                 cell_type: str = "lstm", dropout: float = 0.1):
        super().__init__()
        self.encoder = RNNEncoder(input_size, hidden_size, num_layers, cell_type, dropout)
        self.pred_len = pred_len
        self.input_size = input_size
        self.head = nn.Linear(hidden_size, pred_len * input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.encoder(x))
        return out.view(-1, self.pred_len, self.input_size)


class RNNClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 64,
                 num_layers: int = 2, cell_type: str = "lstm", dropout: float = 0.1):
        super().__init__()
        self.encoder = RNNEncoder(input_size, hidden_size, num_layers, cell_type, dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
