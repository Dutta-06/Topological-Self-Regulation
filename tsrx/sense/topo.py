"""u_c readout (Definition 3.2) and the windowed estimator (Assumption 7.1).

u_c is *not* computed — it is READ: after backward(), the candidate slice
of every consumer's weight.grad already contains it (verified this
session, exact to float64 machine precision, for dense and conv
consumers). This module locates that slice per group/candidate and
combines it across a group's (possibly several) consumers per Theorem 5.7,
respecting the reshape multiplicity from bundle.py (row 6).
"""

from collections import deque
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from tsrx.sense.candidates import CandidateBank


def _consumer_candidate_grad(mod: nn.Module, base_size: int, k: int, multiplicity: int) -> torch.Tensor:
    """Return (k, n_next) L2-summed-over-multiplicity gradient magnitude
    for one consumer's port, i.e. per candidate c the squared-norm of its
    (n_next, multiplicity) sub-block of `mod.weight.grad` at axis=1."""
    g = mod.weight.grad
    if g is None:
        return torch.zeros(k, device=mod.weight.device)
    # axis=1 layout: [0:base_size*mult_real??] -- for THIS consumer, the
    # candidate block is always the trailing k*multiplicity columns,
    # regardless of what precedes them (real channels, possibly through
    # their own multiplicity from the SAME reshape).
    n = k * multiplicity
    cand = g[:, -n:]
    # cand: (n_next, k*mult, *spatial_kernel...) -> group by candidate index c
    rest_shape = cand.shape[2:]
    cand = cand.reshape(cand.shape[0], k, multiplicity, *rest_shape)
    # sum of squares over everything except the candidate axis (dim=1)
    sq = cand.pow(2).sum(dim=tuple(d for d in range(cand.dim()) if d != 1))
    return sq  # (k,) sum-of-squares contribution from this consumer


def compute_uc_norms(bank: CandidateBank, tap: int) -> torch.Tensor:
    """||u_c|| per candidate in group `tap`, aggregated over all of the
    group's consumers (Theorem 5.7: concatenated port, so squared norms
    sum across consumers before the final sqrt)."""
    h = bank.handles[tap]
    modules = dict(bank.model.named_modules())
    device = next(bank.model.parameters()).device
    total_sq = torch.zeros(h.k, device=device)
    for slot in h.bundle.consumer_slots:
        mod = modules[slot.module_name]
        total_sq = total_sq + _consumer_candidate_grad(mod, h.base_size, h.k, slot.multiplicity)
    return total_sq.clamp_min(0).sqrt()


class WindowedSignal:
    """Assumption 7.1's sliding-window estimator: hat s_ell = mean over the
    last T steps of max_c |dL/dgamma_c| — here, of ||u_c|| per candidate,
    since there is no gate; the candidate-wise max is taken at read time
    by the caller (growth-timing, Remark 3.5)."""

    def __init__(self, window: int = 100):
        self.window = window
        self._buf: Dict[int, deque] = {}

    def record(self, tap: int, uc_norms: torch.Tensor) -> None:
        self._buf.setdefault(tap, deque(maxlen=self.window)).append(uc_norms.detach().cpu())

    def mean(self, tap: int) -> Optional[torch.Tensor]:
        buf = self._buf.get(tap)
        if not buf:
            return None
        return torch.stack(list(buf)).mean(dim=0)

    def best(self, tap: int) -> float:
        m = self.mean(tap)
        return float(m.max().item()) if m is not None else 0.0

    def is_ready(self, tap: int) -> bool:
        buf = self._buf.get(tap)
        return buf is not None and len(buf) >= self.window

    def clear(self, tap: int) -> None:
        self._buf.pop(tap, None)
