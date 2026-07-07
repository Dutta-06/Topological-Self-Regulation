"""
ResNet-MLP baseline (Gorishniy et al. 2021, "Revisiting Deep Learning Models
for Tabular Data") — a plain MLP with residual blocks instead of a flat
stack, shown in that paper to be a surprisingly strong, hard-to-beat tabular
baseline.

Block: Linear(d, d_hidden) -> BatchNorm1d -> ReLU -> Dropout ->
       Linear(d_hidden, d) -> Dropout, added back to the block's input.
"""

from typing import List

import torch
import torch.nn as nn

from .common import CatEmbedding


class ResNetBlock(nn.Module):
    def __init__(self, d: int, d_hidden: int, dropout: float):
        super().__init__()
        self.norm = nn.BatchNorm1d(d)
        self.linear1 = nn.Linear(d, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.linear1(h))
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)
        return x + h


class ResNetMLP(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: List[int],
        num_out: int,
        d_main: int = 128,
        d_hidden: int = 256,
        num_blocks: int = 4,
        cat_emb_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cat_embedding = CatEmbedding(cat_cardinalities, cat_emb_dim)
        d_in = num_numeric + self.cat_embedding.output_dim
        self.input_proj = nn.Linear(d_in, d_main)
        self.blocks = nn.ModuleList([ResNetBlock(d_main, d_hidden, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(d_main), nn.ReLU(), nn.Linear(d_main, num_out))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_num, self.cat_embedding(x_cat)], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
