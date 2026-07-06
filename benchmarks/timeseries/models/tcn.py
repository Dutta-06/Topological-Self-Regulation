"""
Temporal Convolutional Network (Bai, Kolter & Koltun 2018).

Stack of dilated causal conv blocks, dilation doubling per level so receptive
field grows exponentially with depth. Each block: two weight-normalized causal
convs + ReLU + dropout, with a residual connection (1x1 conv when channel
counts differ). Causality enforced by left-padding by (kernel_size-1)*dilation
and cropping the same amount off the end ("chomp").
"""

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        ))
        self.chomp1 = _Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        ))
        self.chomp2 = _Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_channels: int = 64,
        num_levels: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = input_size
        for level in range(num_levels):
            dilation = 2 ** level
            layers.append(_TemporalBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))
            in_ch = hidden_channels
        self.network = nn.Sequential(*layers)
        self.hidden_channels = hidden_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> last-step representation (B, hidden_channels)."""
        h = self.network(x.transpose(1, 2))  # (B, hidden_channels, L)
        return h[:, :, -1]


class TCNForecaster(nn.Module):
    def __init__(self, input_size: int, pred_len: int, hidden_channels: int = 64, num_levels: int = 4,
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = TCNEncoder(input_size, hidden_channels, num_levels, kernel_size, dropout)
        self.pred_len = pred_len
        self.input_size = input_size
        self.head = nn.Linear(hidden_channels, pred_len * input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.encoder(x))
        return out.view(-1, self.pred_len, self.input_size)


class TCNClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_channels: int = 64,
                 num_levels: int = 4, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = TCNEncoder(input_size, hidden_channels, num_levels, kernel_size, dropout)
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
