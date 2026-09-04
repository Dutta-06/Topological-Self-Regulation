"""iTransformer for LTSF, built on the SAME reallocation math as PatchTST.

iTransformer (Liu et al., 2024) inverts the usual transformer-for-timeseries
setup: instead of tokens being timesteps or patches, EACH VARIATE'S WHOLE
lookback window is one token, embedded via `Linear(seq_len, d_model)`.
Attention then mixes information ACROSS VARIATES, not across time, and no
positional encoding is used (variates have no inherent order). Everything
past the embedding — LayerNorm placement, attention block, FFN block — is
IDENTICAL to `bench/patchtst.py`'s `_EncoderBlock`, reused directly rather
than re-implemented.

Reallocation story is therefore identical to PatchTST's, verified the same
way (empirical safety probe, not just condition (N)):
  * d_model is frozen (LayerNorm spans it).
  * attention q/k/v are excluded (`view(B, n_vars, h, d_head)` breaks the
    same way patchTST's did).
  * the FFN hidden width (`fc1`/`fc2` in each block) is the reallocated
    group and carries no normalization on its axis at all.
"""

import torch
import torch.nn as nn

from bench.patchtst import _EncoderBlock


class ITransformer(nn.Module):
    def __init__(self, n_vars: int, seq_len: int, pred_len: int, d_model: int = 128,
                 d_ff: int = 256, n_heads: int = 8, n_blocks: int = 3,
                 dropout: float = 0.0, use_revin: bool = True):
        super().__init__()
        self.n_vars, self.seq_len, self.pred_len = n_vars, seq_len, pred_len
        self.use_revin = use_revin
        self.embed = nn.Linear(seq_len, d_model)   # whole window -> one token, per variate
        self.blocks = nn.ModuleList(
            [_EncoderBlock(d_model, d_ff, n_heads, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, pred_len)   # shared across variate-tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - mean) / std

        z = self.embed(x.transpose(1, 2))          # (B, C, d_model) -- C variate-tokens
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        out = self.head(z).transpose(1, 2)         # (B, pred_len, C)
        if self.use_revin:
            out = out * std + mean
        return out


def build_itransformer(n_vars: int, seq_len: int, pred_len: int, d_model: int = 128,
                        d_ff: int = 256, n_heads: int = 8, n_blocks: int = 3,
                        use_revin: bool = True) -> nn.Module:
    return ITransformer(n_vars, seq_len, pred_len, d_model=d_model, d_ff=d_ff,
                         n_heads=n_heads, n_blocks=n_blocks, use_revin=use_revin)
