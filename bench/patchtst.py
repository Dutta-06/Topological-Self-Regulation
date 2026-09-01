"""PatchTST-style encoder for LTSF, built so TSR-X can reallocate FFN width.

Why this architecture, and why these choices specifically:

  * The TCN testbed failed for a structural reason -- its head was 58-99% of
    all parameters, so the conv body TSR-X could index was 0.8-41% of the
    model and the "discovered" allocation was one knob. Here each block's
    FFN is roughly half the block's parameters and is a genuinely local,
    per-depth quantity, which is the thing the literature says should not
    be uniform across depth.

  * SEPARATE q/k/v projections, not a fused Linear(d, 3d). A fused
    projection is laid out block-major [Q|K|V], so the trailing candidate
    columns CandidateBank appends land entirely inside V. `ParamSlot`
    models channel-major flatten multiplicity, not block-major fusion.
    Splitting them sidesteps the problem instead of patching it.

  * d_model is FROZEN. It is one coupling group spanning every block's
    LayerNorm, q/k/v input, FFN input and the head -- so (a) it violates
    condition (N) (LayerNorm statistics span the candidate channels, so a
    zeroed port is not enough for function preservation) and (b) resizing
    it would confound every block at once. Reallocating it needs the
    masked-LayerNorm construction, which is deliberately out of scope.

  * Attention head width is NOT reallocated. `view(B, L, h, d_head)` needs
    the projection width divisible by the head count, and the coupling
    engine cannot see that constraint -- it models module->module edges and
    has no representation of QK^T or AV. Verified: pruning one index from
    the q group gives "RuntimeError: shape '[2,6,4,8]' is invalid for input
    of size 372". `tsrx.graph.safety.safe_taps` catches this class of bug
    empirically, and `ffn_taps` below returns only the FFN groups.

The FFN hidden axis carries no normalization at all, so condition (N) holds
trivially there -- which is why this needs no new mathematics.
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.fc1 = nn.Linear(d_model, d_ff)      # <- producer of the reallocated group
        self.fc2 = nn.Linear(d_ff, d_model)      # <- its consumer
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        y = self.norm1(x)
        q = self.q(y).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(y).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(y).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        att = torch.softmax(q @ k.transpose(-2, -1) / (self.d_head ** 0.5), dim=-1)
        x = x + self.drop(self.o((att @ v).transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.fc2(F.gelu(self.fc1(self.norm2(x)))))


class PatchTST(nn.Module):
    """Channel-independent patched transformer.

    Each variate is patched and encoded independently through shared
    weights (the PatchTST convention), so parameter count is independent of
    the number of variates and one architecture is comparable across
    datasets. RevIN is applied over the TIME axis per (sample, variate); it
    holds no parameters on any reallocated axis, so it cannot couple
    candidates into real units the way LayerNorm does.
    """

    def __init__(self, n_vars: int, seq_len: int, pred_len: int, patch_len: int = 16,
                 stride: int = 8, d_model: int = 128, d_ff: int = 256, n_heads: int = 8,
                 n_blocks: int = 3, dropout: float = 0.0, use_revin: bool = True):
        super().__init__()
        self.n_vars, self.seq_len, self.pred_len = n_vars, seq_len, pred_len
        self.patch_len, self.stride = patch_len, stride
        self.use_revin = use_revin
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(d_model, d_ff, n_heads, dropout) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - mean) / std

        z = x.permute(0, 2, 1).reshape(B * C, L)                       # channel-independent
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # (B*C, n_patches, patch_len)
        z = self.embed(z) + self.pos
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        out = self.head(z.reshape(z.shape[0], -1))                     # (B*C, pred_len)
        out = out.reshape(B, C, self.pred_len).permute(0, 2, 1)
        if self.use_revin:
            out = out * std + mean
        return out


def ffn_taps(model: nn.Module, bundles: Dict[int, object]) -> List[int]:
    """Taps whose producer is a block's `fc1` — i.e. the FFN hidden widths.

    Deliberately an allowlist rather than a denylist: everything else in a
    transformer is either unsafe to resize (attention head split, d_model's
    LayerNorm) or meaningless (the output head). Prefer
    `tsrx.graph.safety.safe_taps` when you want this derived empirically.
    """
    out = []
    for tap, bd in bundles.items():
        producers = {s.module_name for s in bd.producer_slots}
        if any(n.endswith(".fc1") for n in producers):
            out.append(tap)
    return sorted(out)


def build_patchtst(n_vars: int, seq_len: int, pred_len: int, d_model: int = 128,
                    d_ff: int = 256, n_heads: int = 8, n_blocks: int = 3,
                    patch_len: int = 16, stride: int = 8, use_revin: bool = True) -> nn.Module:
    return PatchTST(n_vars, seq_len, pred_len, patch_len=patch_len, stride=stride,
                    d_model=d_model, d_ff=d_ff, n_heads=n_heads, n_blocks=n_blocks,
                    use_revin=use_revin)
