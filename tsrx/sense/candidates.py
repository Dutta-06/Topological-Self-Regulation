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

    def __init__(self, model: nn.Module, bundles: Dict[int, IndexBundle], k: int = 8,
                 only_taps: Optional[List[int]] = None,
                 skip_unsupported: bool = False):
        """
        Args:
            only_taps: attach candidates ONLY to these groups. Required for
                architectures where some groups are structurally unsafe to
                resize even though they look ordinary to the coupling
                engine — multi-head attention's q/k/v projections are the
                motivating case: `view(B, L, h, d_head)` silently requires
                the width to stay divisible by the head count, and the
                engine cannot see that (it models module->module edges, not
                the QK^T / AV matmuls). Use `tsrx.graph.safety.safe_taps`
                to derive this list empirically rather than by hand.
            skip_unsupported: record LayerNorm/GroupNorm groups in
                `self.skipped` and carry on, instead of raising. A
                transformer has exactly one such group -- the d_model
                residual stream -- and freezing it is the intended design,
                so aborting the whole bank for it is wrong. Off by default
                so existing callers keep the loud failure.
        """
        super().__init__()
        object.__setattr__(self, "model", model)
        self.k = k
        self.handles: Dict[int, CandidateHandle] = {}
        self.skipped: Dict[int, str] = {}
        for tap, bd in bundles.items():
            if only_taps is not None and tap not in only_taps:
                self.skipped[tap] = "not in only_taps"
                continue
            if bd.size == 0 or not bd.producer_slots:
                continue
            if not bd.consumer_slots:
                # No downstream consumer => this group is a model OUTPUT
                # (e.g. the classifier head). Never grow the output head —
                # its width is the task's number of classes/targets, not a
                # free capacity dimension (matches the terminal-layer rule
                # of the old TSR engine and the framework's convention).
                continue
            if skip_unsupported and self._has_ln_gn(bd):
                self.skipped[tap] = "LayerNorm/GroupNorm on this axis (condition N)"
                continue
            self._attach(bd)

        # Final pass: zero every port block AFTER all groups are attached.
        # Per-group zeroing during _attach is not sufficient — attaching
        # group B adds Kaiming rows to a module whose candidate columns
        # group A already zeroed, repopulating the candidate x candidate
        # corner. Function preservation survives that (the corner exits
        # through B's own zeroed port), but it leaves a nonzero block that
        # makes the dormancy invariant non-uniform and lets candidates wire
        # into each other. Zeroing once at the end is simpler and safer.
        self.zero_ports()

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

    # ------------------------------------------------------------------
    # Dormancy enforcement (Definition 3.2 / Lemma 2.2)
    # ------------------------------------------------------------------

    def _port_columns(self):
        """Yield (module, column_slice) for every candidate port block.

        A consumer's input axis is laid out [ real (base*mult) |
        candidates (k*mult) ], so the candidate port is always the
        trailing k*mult columns.
        """
        modules = _modules_by_name(self.model)
        for h in self.handles.values():
            seen = set()
            for slot in h.bundle.consumer_slots:
                if slot.module_name in seen:
                    continue
                seen.add(slot.module_name)
                mod = modules.get(slot.module_name)
                if mod is None or getattr(mod, "weight", None) is None:
                    continue
                n = h.k * slot.multiplicity
                total_cols = mod.weight.shape[1]
                yield mod, slice(total_cols - n, total_cols)

    @torch.no_grad()
    def zero_ports(self) -> None:
        """Re-zero every candidate port column. MUST be called after every
        optimizer.step().

        The whole construction depends on candidates being SILENT: u_c is
        defined (Definition 3.2) as the port gradient evaluated at
        v_c = 0, and Lemma 2.2's function-preservation holds only there.
        But the port columns carry a real, nonzero gradient (u_c IS that
        gradient), so any ordinary optimizer will happily train them away
        from zero — after which the "candidates" are just extra live
        channels, u_c degenerates into an ordinary weight gradient, and
        the model being evaluated no longer matches the model being
        counted. Measured drift without this call: logits move by 5.3
        after ONE SGD step and ~1e6 after ten.

        Masking after the step (rather than excluding the columns from
        the optimizer) is deliberate: the port is a SLICE of a shared
        weight tensor, not a standalone parameter, so it cannot be put in
        its own param group. Momentum/weight-decay may still accumulate
        for those entries, but re-zeroing the values each step makes the
        forward pass exactly dormant regardless.
        """
        for mod, cols in self._port_columns():
            mod.weight.data[:, cols] = 0.0

    @torch.no_grad()
    def max_port_magnitude(self) -> float:
        """Largest |port weight| across all groups — 0.0 iff fully dormant.
        Cheap assertion for tests and for a periodic training-time check."""
        worst = 0.0
        for mod, cols in self._port_columns():
            block = mod.weight.data[:, cols]
            if block.numel():
                worst = max(worst, float(block.abs().max().item()))
        return worst

    # ------------------------------------------------------------------
    # Accounting
    # ------------------------------------------------------------------

    def deployed_params(self) -> int:
        """Trainable parameter count of the DEPLOYED model, i.e. what
        remains after `detach()` strips every candidate row/column.

        Computed by measuring each tensor's real extent per axis rather
        than by subtracting a per-group candidate cost. The subtractive
        form double-counts the candidate x candidate corner block of any
        module that is simultaneously a consumer of one candidate-bearing
        group (axis 1) and a producer of another (axis 0) — it removes
        k_in*(out) and k_out*(in) but the true candidate mass is
        k_in*base_out + k_out*base_in + k_in*k_out, so the corner is
        subtracted twice. Measured error on resnet18/k=4: -2,352 params,
        growing as O(k^2), always in the flattering direction.
        """
        modules = _modules_by_name(self.model)

        # real output extent per producing module, real input extent per consuming module
        real_out: Dict[str, int] = {}
        real_in: Dict[str, int] = {}
        for h in self.handles.values():
            for slot in h.bundle.producer_slots:
                real_out[slot.module_name] = h.base_size
            for slot in h.bundle.affine_slots:
                real_out[slot.module_name] = h.base_size
            for slot in h.bundle.consumer_slots:
                real_in[slot.module_name] = h.base_size * slot.multiplicity

        total = 0
        for name, mod in modules.items():
            for pname, p in list(mod.named_parameters(recurse=False)):
                if not p.requires_grad:
                    continue
                dims = list(p.shape)
                if len(dims) >= 1 and name in real_out:
                    dims[0] = real_out[name]
                if len(dims) >= 2 and name in real_in:
                    dims[1] = real_in[name]
                n = 1
                for d in dims:
                    n *= d
                total += n
        return total

    def detach(self) -> nn.Module:
        """Strip all candidate rows and columns from the model, leaving the pure deployed model."""
        modules = _modules_by_name(self.model)
        for h in self.handles.values():
            k = h.k
            base_size = h.base_size

            # 1. Producers
            producers_seen = set()
            for s in h.bundle.producer_slots:
                if s.module_name in producers_seen:
                    continue
                producers_seen.add(s.module_name)
                mod = modules[s.module_name]
                mod.weight = nn.Parameter(mod.weight.data[:base_size])
                if getattr(mod, "bias", None) is not None:
                    mod.bias = nn.Parameter(mod.bias.data[:base_size])
                _bump_out_attr(mod, -k)

            # 2. Affines
            affines_seen = set()
            for s in h.bundle.affine_slots:
                if s.module_name in affines_seen:
                    continue
                affines_seen.add(s.module_name)
                mod = modules[s.module_name]
                if getattr(mod, "weight", None) is not None:
                    mod.weight = nn.Parameter(mod.weight.data[:base_size])
                if getattr(mod, "bias", None) is not None:
                    mod.bias = nn.Parameter(mod.bias.data[:base_size])
                if getattr(mod, "running_mean", None) is not None:
                    mod.running_mean = mod.running_mean[:base_size]
                if getattr(mod, "running_var", None) is not None:
                    mod.running_var = mod.running_var[:base_size]
                for attr in ("num_features", "num_channels"):
                    if hasattr(mod, attr):
                        setattr(mod, attr, getattr(mod, attr) - k)

            # 3. Consumers
            consumers_seen = set()
            for s in h.bundle.consumer_slots:
                if s.module_name in consumers_seen:
                    continue
                consumers_seen.add(s.module_name)
                mod = modules[s.module_name]
                mult = s.multiplicity
                mod.weight = nn.Parameter(mod.weight.data[:, : base_size * mult])
                _bump_in_attr(mod, -(k * mult))

        self.handles.clear()
        return self.model

