"""
FT-Transformer (Gorishniy et al. 2021, "Revisiting Deep Learning Models for
Tabular Data") — tokenizes every feature (numeric via a per-feature learned
linear map, categorical via embedding lookup), prepends a [CLS] token, and
runs a standard pre-LN Transformer encoder over the resulting feature
sequence. The CLS token's final representation feeds the prediction head.

Scales with d_token/num_layers/n_heads independent of feature count (aside
from the linear growth in sequence length), so unlike the forecasting
PatchTST head, param count here is governed by the presets, not by the
dataset — this is the model pushed hardest for the multi-million-param
Pareto sweep on Higgs/Covertype (configs/higgs.yaml, configs/covertype.yaml).
"""

from typing import List

import torch
import torch.nn as nn

from .common import FeatureTokenizer


class FTTransformer(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: List[int],
        num_out: int,
        d_token: int = 64,
        n_heads: int = 8,
        num_layers: int = 3,
        d_ffn_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_numeric, cat_cardinalities, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * d_ffn_factor,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, num_out))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(tokens)
        return self.head(encoded[:, 0, :])
