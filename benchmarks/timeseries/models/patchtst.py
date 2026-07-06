"""
PatchTST (Nie, Nguyen, Sinthong & Kalagnanam, 2023) — lightweight reference.

Core ideas kept faithful to the paper:
  1. Channel independence: every channel is patched and encoded by the SAME
     shared Transformer (no cross-channel mixing inside the encoder).
  2. Patching: each channel's series is split into overlapping patches (length
     patch_len, stride), turning a long sequence into a short one of patch
     tokens — this is what lets a plain Transformer scale to long series.

Forecasting: per-channel flatten head (Linear(num_patches*d_model, pred_len)),
matching the paper — every channel independently predicts its own pred_len-step
future; output is (B, pred_len, C) to match the other baselines' target shape.
This head's parameter count scales with num_patches (hence with seq_len),
same as the paper's own reported configs — this is expected, not a bug.

Classification: pools over the PATCH axis (mean) before the class head, so the
head's size depends only on d_model, not on num_patches/seq_len. This matters
for capacity-matched comparisons across classification datasets of different
sequence length — LSTM/GRU/TCN/Mamba are already seq-len-invariant, and a
flatten-based classification head would make PatchTST alone balloon on longer
series with no counterpart in the other baselines.
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
        """x: (B, L, C) -> per-channel per-patch encoding (B, C, num_patches, d_model)."""
        b, L, c = x.shape
        x = x.transpose(1, 2).reshape(b * c, L)             # (B*C, L)
        patches = x.unfold(-1, self.patch_len, self.stride)  # (B*C, num_patches, patch_len)
        h = self.patch_embed(patches) + self.pos_embed       # (B*C, num_patches, d_model)
        h = self.encoder(h)                                  # (B*C, num_patches, d_model)
        h = h.reshape(b, c, self.num_patches, self.d_model)
        return h


class PatchTSTForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = _PatchTSTBackbone(seq_len, patch_len, stride, d_model, n_heads, num_layers, dropout)
        self.pred_len = pred_len
        self.head = nn.Linear(self.backbone.num_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)                       # (B, C, num_patches, d_model)
        h = h.flatten(start_dim=2)                  # (B, C, num_patches*d_model)
        pred = self.head(h)                         # (B, C, pred_len) — per-channel forecast
        return pred.transpose(1, 2)                 # (B, pred_len, C)


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
        # Fixed d_model -> d_model projection (not num_patches*d_model): keeps
        # the classifier's parameter count seq-len-invariant, matching the
        # other baselines. Patch pooling happens in forward() before this.
        self.channel_proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)          # (B, C, num_patches, d_model)
        h = h.mean(dim=2)             # pool over patches -> (B, C, d_model)
        h = self.channel_proj(h)      # (B, C, d_model)
        h = h.mean(dim=1)             # pool over channels -> (B, d_model)
        return self.head(h)
