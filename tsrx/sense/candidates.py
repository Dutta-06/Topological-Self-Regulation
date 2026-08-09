"""CandidateBank: attach k dormant candidate units per coupling group,
directly as extra rows/columns of the REAL model's own parameters,
with zeroed ports (Lemma 2.2 / Corollary of the verified `.grad`-on-
zeroed-port identity — see paper/verify_framework.py and this session's
conv/dense checks).

v1 scope (Tier 1): groups whose only norm is BatchNorm/InstanceNorm
(condition (N) holds exactly — verified drift 0.0). Groups touching
LayerNorm/GroupNorm are skipped here and raise on attach; Lemma 5.6's
split-norm handling is Tier 2 work.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from tsrx.graph.bundle import IndexBundle, ParamSlot

_LN_GN_TYPES = (nn.LayerNorm, nn.GroupNorm)


class UnsupportedGroupError(Exception):
    pass


@dataclass
class CandidateHandle:
    tap: int
    bundle: IndexBundle
    k: int
    base_size: int          # real (non-candidate) width of the group
    alive: List[bool]       # length k; False once materialized/discarded/pending-refill


def _modules_by_name(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _kaiming_rows(n: int, fan_in_shape, device, dtype) -> torch.Tensor:
    t = torch.empty(n, *fan_in_shape, device=device, dtype=dtype)
    nn.init.kaiming_uniform_(t.view(n, -1), a=5 ** 0.5) if t.numel() else None
    return t


def _extend_producer_weight(mod: nn.Module, k: int) -> None:
    w = mod.weight
    new_rows = torch.empty(k, *w.shape[1:], device=w.device, dtype=w.dtype)
    if new_rows.numel():
        nn.init.kaiming_uniform_(new_rows.reshape(k, -1), a=5 ** 0.5)
    mod.weight = nn.Parameter(torch.cat([w.data, new_rows], dim=0))
    if getattr(mod, "bias", None) is not None:
        b = mod.bias
        mod.bias = nn.Parameter(torch.cat([b.data, torch.zeros(k, device=b.device, dtype=b.dtype)]))
    _bump_out_attr(mod, k)


def _extend_consumer_weight_zero(mod: nn.Module, k: int, multiplicity: int = 1) -> None:
    # Row 6 (reshape/flatten): each of the k new candidate channels backs
    # `multiplicity` contiguous input slots downstream of a flatten, not 1
    # (Table 1: i -> {iS,...,(i+1)S-1}). Appending k*S zero columns at the
    # end is correct because candidates are appended after the last real
    # channel, and flatten's channel-major ordering keeps each channel's
    # S-block contiguous — so the new candidate S-blocks land contiguously
    # at the end too, no interleaving required.
    w = mod.weight
    n = k * multiplicity
    zero_cols = torch.zeros(w.shape[0], n, *w.shape[2:], device=w.device, dtype=w.dtype)
    mod.weight = nn.Parameter(torch.cat([w.data, zero_cols], dim=1))
    _bump_in_attr(mod, n)


def _extend_affine(mod: nn.Module, k: int) -> None:
    device = mod.weight.device if getattr(mod, "weight", None) is not None else "cpu"
    dtype = mod.weight.dtype if getattr(mod, "weight", None) is not None else torch.float32
    if getattr(mod, "weight", None) is not None:
        mod.weight = nn.Parameter(torch.cat([mod.weight.data, torch.ones(k, device=device, dtype=dtype)]))
    if getattr(mod, "bias", None) is not None:
        mod.bias = nn.Parameter(torch.cat([mod.bias.data, torch.zeros(k, device=device, dtype=dtype)]))
    if getattr(mod, "running_mean", None) is not None:
        mod.running_mean = torch.cat([mod.running_mean, torch.zeros(k, device=device, dtype=dtype)])
    if getattr(mod, "running_var", None) is not None:
        mod.running_var = torch.cat([mod.running_var, torch.ones(k, device=device, dtype=dtype)])
    for attr in ("num_features", "num_channels"):
        if hasattr(mod, attr):
            setattr(mod, attr, getattr(mod, attr) + k)


def _bump_out_attr(mod: nn.Module, k: int) -> None:
    for attr in ("out_channels", "out_features"):
        if hasattr(mod, attr):
            setattr(mod, attr, getattr(mod, attr) + k)


def _bump_in_attr(mod: nn.Module, k: int) -> None:
    for attr in ("in_channels", "in_features"):
        if hasattr(mod, attr):
            setattr(mod, attr, getattr(mod, attr) + k)


class CandidateBank(nn.Module):
    """Owns the bookkeeping for candidates attached across every eligible
    coupling group of a wrapped model. The candidates themselves are NOT
    separate parameters of this module — they live inside the real
    model's tensors (see module docstring); this class only tracks which
    trailing indices, per group, are candidates, and drives
    materialize/refill.
    """

    def __init__(self, model: nn.Module, bundles: Dict[int, IndexBundle], k: int = 8):
        super().__init__()
        object.__setattr__(self, "model", model)
        self.k = k
        self.handles: Dict[int, CandidateHandle] = {}
        for tap, bd in bundles.items():
            if bd.size == 0 or not bd.producer_slots:
                continue
            if not bd.consumer_slots:
                # No downstream consumer => this group is a model OUTPUT
                # (e.g. the classifier head). Never grow the output head —
                # its width is the task's number of classes/targets, not a
                # free capacity dimension (matches the terminal-layer rule
                # of the old TSR engine and the framework's convention).
                continue
            self._attach(bd)

    def _has_ln_gn(self, bd: IndexBundle) -> bool:
        modules = _modules_by_name(self.model)
        for s in bd.affine_slots:
            mod = modules.get(s.module_name)
            if isinstance(mod, _LN_GN_TYPES):
                return True
        return False

    def _attach(self, bd: IndexBundle) -> None:
        if self._has_ln_gn(bd):
            raise UnsupportedGroupError(
                f"group tap={bd.tap} touches LayerNorm/GroupNorm; condition (N) "
                "does not hold (Lemma 5.6) — Tier 2 split-norm handling required, "
                "not yet implemented in CandidateBank v1."
            )
        modules = _modules_by_name(self.model)
        base_size = bd.size
        k = self.k

        # Dedupe by module name: a BatchNorm contributes 4 ParamSlots
        # (weight/bias/running_mean/running_var) for ONE module, and each
        # extend_* helper below already extends everything that module
        # owns in a single call — calling it once per slot would
        # over-extend by a factor of (slots per module).
        producer_names = dict.fromkeys(s.module_name for s in bd.producer_slots)
        affine_names = dict.fromkeys(s.module_name for s in bd.affine_slots)
        consumer_mult: Dict[str, int] = {}
        for s in bd.consumer_slots:
            consumer_mult[s.module_name] = s.multiplicity  # one weight-slot per consumer module

        for name in producer_names:
            _extend_producer_weight(modules[name], k)
        for name in affine_names:
            _extend_affine(modules[name], k)
        for name, mult in consumer_mult.items():
            _extend_consumer_weight_zero(modules[name], k, multiplicity=mult)

        self.handles[bd.tap] = CandidateHandle(
            tap=bd.tap, bundle=bd, k=k, base_size=base_size, alive=[True] * k,
        )

    def candidate_slice(self, tap: int) -> slice:
        h = self.handles[tap]
        return slice(h.base_size, h.base_size + h.k)
