"""TSMixer (Chen et al., 2023, arXiv:2303.06053) for LTSF — an all-MLP
architecture, the third structural family in this project alongside conv
(TCN) and attention (PatchTST/iTransformer).

Each block alternates two residual MLPs with BatchNorm before each (the
paper's own design — "time-mixing and feature-mixing MLP blocks ... with
residual connections and batch norm" — not a workaround chosen to fit our
engine, though it happens to also sidestep the LayerNorm blocker entirely):

  time-mixing:    BatchNorm(C) -> Linear(L, L)       -- mixes across time
  feature-mixing: BatchNorm(C) -> Linear(C, ff) -> Linear(ff, C)  -- mixes
                  across variates; `ff` is the reallocated group.

Two groups here are task-fixed, not free architectural capacity, and MUST
be excluded even though nothing here violates condition (N):
  * the variate axis C (= n_vars: shrinking it would mean the model no
    longer accepts the dataset's actual channel count)
  * the time-mixing Linear(L, L)'s axis (= seq_len: same problem for the
    input window length)
Both are caught by `tsrx.graph.safety.safe_taps`'s empirical probe, not by
condition (N) — pruning either produces a model whose forward pass no
longer matches the fixed input shape the loader actually produces, which
the probe's "prune one index, then run a forward pass" check catches
directly. This was verified, not assumed — see
tests_tsrx/test_tsmixer.py::test_task_fixed_axes_excluded_by_probe.
"""

import torch
import torch.nn as nn


class _MixerBlock(nn.Module):
    def __init__(self, seq_len: int, n_vars: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.bn_time = nn.BatchNorm1d(n_vars)
        self.fc_time = nn.Linear(seq_len, seq_len)
        self.bn_feat = nn.BatchNorm1d(n_vars)
        self.fc1 = nn.Linear(n_vars, d_ff)     # <- producer of the reallocated group
        self.fc2 = nn.Linear(d_ff, n_vars)     # <- its consumer
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        t = self.bn_time(x.transpose(1, 2))               # (B, C, L), per-channel norm
        t = self.act(self.fc_time(t))
        x = x + self.drop(t.transpose(1, 2))               # time-mixing residual

        f = self.bn_feat(x.transpose(1, 2)).transpose(1, 2)  # (B, L, C)
        f = self.fc2(self.act(self.fc1(f)))
        return x + self.drop(f)                             # feature-mixing residual


class TSMixer(nn.Module):
    def __init__(self, n_vars: int, seq_len: int, pred_len: int, d_ff: int = 256,
                 n_blocks: int = 4, dropout: float = 0.0, use_revin: bool = True):
        super().__init__()
        self.use_revin = use_revin
        self.blocks = nn.ModuleList(
            [_MixerBlock(seq_len, n_vars, d_ff, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(seq_len, pred_len)  # channel-independent output, like DLinear's

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - mean) / std

        z = x
        for blk in self.blocks:
            z = blk(z)
        out = self.head(z.transpose(1, 2)).transpose(1, 2)   # (B, pred_len, C)
        if self.use_revin:
            out = out * std + mean
        return out


def build_tsmixer(n_vars: int, seq_len: int, pred_len: int, d_ff: int = 256,
                   n_blocks: int = 4, use_revin: bool = True) -> nn.Module:
    return TSMixer(n_vars, seq_len, pred_len, d_ff=d_ff, n_blocks=n_blocks, use_revin=use_revin)


def mixer_ffn_taps(model: nn.Module, bundles) -> list:
    """Taps whose producer is a block's `fc1` — the feature-mixing hidden
    width, the only group here that is both condition-(N)-clean AND not
    task-fixed. Prefer `tsrx.graph.safety.safe_taps` for the empirically
    verified set; this is the name-based cross-check."""
    out = []
    for tap, bd in bundles.items():
        producers = {s.module_name for s in bd.producer_slots}
        if any(n.endswith(".fc1") for n in producers):
            out.append(tap)
    return sorted(out)
