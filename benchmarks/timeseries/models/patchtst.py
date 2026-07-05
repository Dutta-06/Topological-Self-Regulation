"""
PatchTST (Nie, Nguyen, Sinthong & Kalagnanam, 2023) — lightweight reference.

Core ideas kept faithful to the paper:
  1. Channel independence: every channel is patched and encoded by the SAME
     shared Transformer (no cross-channel mixing inside the encoder).
  2. Patching: each channel's series is split into overlapping patches (length
     patch_len, stride), turning a long sequence into a short one of patch
     tokens — this is what lets a plain Transformer scale to long series.

Forecasting adapts to this repo's single-scalar-target convention (window ->
one future value of the target/last channel): every channel is still encoded
independently through the shared patch embedding + Transformer, and each
channel produces its own scalar head output; only the target channel's output
is used as the prediction (matching the ETT/Electricity loaders' schema).

Classification pools the per-channel encodings before the class head.
"""

import torch
import torch.nn as nn


class _PatchTSTBackbone(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = max(1, (seq_len - patch_len) // stride + 1)

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> per-channel encoding (B, C, num_patches * d_model)."""
        b, L, c = x.shape
        x = x.transpose(1, 2).reshape(b * c, L)             # (B*C, L)
        patches = x.unfold(-1, self.patch_len, self.stride)  # (B*C, num_patches, patch_len)
        h = self.patch_embed(patches) + self.pos_embed       # (B*C, num_patches, d_model)
        h = self.encoder(h)                                  # (B*C, num_patches, d_model)
        h = h.reshape(b, c, self.num_patches * self.d_model)
        return h


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = _PatchTSTBackbone(seq_len, patch_len, stride, d_model, n_heads, num_layers, dropout)
        self.head = nn.Linear(self.backbone.num_patches * d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)          # (B, C, num_patches*d_model)
        pred = self.head(h).squeeze(-1)  # (B, C) — one scalar per channel
        return pred[:, -1]            # target channel is last, matches loader convention


class PatchTSTClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        num_classes: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = _PatchTSTBackbone(seq_len, patch_len, stride, d_model, n_heads, num_layers, dropout)
        self.channel_proj = nn.Linear(self.backbone.num_patches * d_model, d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)                     # (B, C, num_patches*d_model)
        h = self.channel_proj(h)                  # (B, C, d_model)
        h = h.mean(dim=1)                         # channel pooling -> (B, d_model)
        return self.head(h)
