"""
Shared building blocks for tabular baselines.

Every dataset loader in benchmarks/tabular/data yields (x_num, x_cat, y):
    x_num: (B, num_numeric) float32 — standardized continuous features
    x_cat: (B, num_categorical) long — label-encoded categorical features
           (an empty (B, 0) tensor for datasets with no categoricals)
    y: (B,) long (classification) or (B,) float (regression)

Two ways models consume (x_num, x_cat):
  - "flat" (MLP, ResNet-MLP, TabNet): embed each categorical column and
    concatenate with x_num into one dense vector.
  - "tokenized" (FT-Transformer, SAINT): tokenize every feature (numeric and
    categorical alike) into its own d_token vector, forming a (B, F, d_token)
    sequence a Transformer can attend over.
"""

from typing import List

import torch
import torch.nn as nn


class CatEmbedding(nn.Module):
    """Embeds each categorical column and concatenates into one flat vector."""

    def __init__(self, cardinalities: List[int], emb_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cardinalities])
        self.output_dim = len(cardinalities) * emb_dim

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        if not self.embeddings:
            return x_cat.new_zeros(x_cat.shape[0], 0, dtype=torch.float32)
        return torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)], dim=-1)


class FeatureTokenizer(nn.Module):
    """Tokenizes numeric + categorical features into a (B, F, d_token) sequence.

    Numeric tokenizer follows the FT-Transformer convention (Gorishniy et al.
    2021): a per-feature learned linear map, x_i -> x_i * w_i + b_i, rather
    than a single shared linear layer — each feature gets its own scale/shift
    into token space before attention ever mixes features together.
    """

    def __init__(self, num_numeric: int, cardinalities: List[int], d_token: int):
        super().__init__()
        self.num_numeric = num_numeric
        self.d_token = d_token
        self.num_weight = nn.Parameter(torch.empty(num_numeric, d_token))
        self.num_bias = nn.Parameter(torch.empty(num_numeric, d_token))
        nn.init.normal_(self.num_weight, std=0.02)
        nn.init.zeros_(self.num_bias)
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, d_token) for card in cardinalities])
        self.num_tokens = num_numeric + len(cardinalities)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = []
        if self.num_numeric > 0:
            # (B, F, 1) * (1, F, d) + (1, F, d) -> (B, F, d)
            num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0) + self.num_bias.unsqueeze(0)
            tokens.append(num_tokens)
        if self.cat_embeddings:
            cat_tokens = torch.stack([emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
            tokens.append(cat_tokens)
        return torch.cat(tokens, dim=1)


def build_head(d_in: int, d_out: int) -> nn.Module:
    """Standard prediction head: norm -> activation -> linear, shared across models."""
    return nn.Sequential(nn.LayerNorm(d_in), nn.ReLU(), nn.Linear(d_in, d_out))
