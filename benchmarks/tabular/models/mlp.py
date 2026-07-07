"""
Plain MLP baseline — the "fixed architecture, no tricks" reference point.

Categorical columns are embedded (small fixed embedding dim) and concatenated
with the standardized numeric features into one flat input vector; a stack
of Linear -> BatchNorm1d -> ReLU -> Dropout blocks follows.
"""

from typing import List

import torch
import torch.nn as nn

from .common import CatEmbedding


class MLP(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: List[int],
        num_out: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        cat_emb_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_embedding = CatEmbedding(cat_cardinalities, cat_emb_dim)
        d_in = num_numeric + self.cat_embedding.output_dim

        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(d_in, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            d_in = hidden_size
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, num_out)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_num, self.cat_embedding(x_cat)], dim=-1)
        return self.head(self.trunk(x))
