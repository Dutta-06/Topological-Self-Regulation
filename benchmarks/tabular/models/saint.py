"""
SAINT (Somepalli et al. 2021, "SAINT: Improved Neural Networks for Tabular
Data via Row Attention and Contrastive Pre-Training") — same feature
tokenizer + [CLS] token as FT-Transformer, but each block alternates two
attention passes:
  1. self-attention over the feature-token sequence (as in FT-Transformer)
  2. "intersample" attention: flatten each sample's tokens into one vector,
     then attend *across the batch* (samples attend to other samples) —
     implemented by reshaping (B, F, d) -> (1, B, F*d) so nn.MultiheadAttention
     treats the batch dimension as its sequence.

Intersample attention's embed dim is F*d_token (F = num tokens including
CLS), which won't always divide evenly by the configured n_heads for an
arbitrary dataset's feature count — `_largest_divisor_leq` picks the largest
head count <= n_heads that divides evenly, rather than hardcoding a fixed
head count that could silently be wrong for a different dataset.

The intersample block skips the FFN sublayer (attention + residual only):
its embed dim (F*d_token) is already large before any FFN widening, so a
d_ffn_factor-scaled FFN there would make param count blow up quadratically
with feature count for essentially no accuracy benefit — the feature_attn
sublayer already provides FFN capacity. Even without it, SAINT is
inherently costlier per d_token than FT-Transformer once feature count
grows (Covertype's 54 features vs. Higgs's 28), which is why its presets
use a noticeably smaller d_token than the other transformer baseline.
"""

from typing import List

import torch
import torch.nn as nn

from .common import FeatureTokenizer


def _largest_divisor_leq(n: int, upper: int) -> int:
    for h in range(min(upper, n), 0, -1):
        if n % h == 0:
            return h
    return 1


class PreLNAttentionBlock(nn.Module):
    """Pre-LN transformer block: MHA (+ optional FFN), both with residuals."""

    def __init__(self, d: int, n_heads: int, d_ffn_factor: int, dropout: float, use_ffn: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.use_ffn = use_ffn
        if use_ffn:
            self.norm2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(
                nn.Linear(d, d * d_ffn_factor), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * d_ffn_factor, d)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        if self.use_ffn:
            x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class SAINTBlock(nn.Module):
    def __init__(self, d_token: int, num_tokens: int, n_heads: int, d_ffn_factor: int, dropout: float):
        super().__init__()
        self.feature_attn = PreLNAttentionBlock(d_token, n_heads, d_ffn_factor, dropout)
        d_flat = d_token * num_tokens
        inter_heads = _largest_divisor_leq(d_flat, n_heads)
        self.intersample_attn = PreLNAttentionBlock(d_flat, inter_heads, d_ffn_factor, dropout, use_ffn=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_attn(x)  # (B, F, d)
        b, f, d = x.shape
        flat = x.reshape(b, f * d).unsqueeze(0)  # (1, B, F*d) — attend across the batch
        flat = self.intersample_attn(flat)
        return flat.squeeze(0).reshape(b, f, d)


class SAINT(nn.Module):
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
        num_tokens = self.tokenizer.num_tokens + 1  # + CLS

        self.blocks = nn.ModuleList(
            [SAINTBlock(d_token, num_tokens, n_heads, d_ffn_factor, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, num_out))

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x_num, x_cat)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(tokens[:, 0, :])
