"""Incumbent removal saliency (Theorem 4.2 / Eq. 14-15).

<u_j, v_j> = E[a_j * dL/da_j] is, by the chain rule, exactly
sum_s w[s,j]*grad[s,j] summed over the consumer's out axis — the
classical weight*grad Taylor criterion (Molchanov et al.), here derived
rather than assumed. No extra forward/backward pass needed: it reads the
SAME .grad tensors a live training step already produced, at the REAL
(non-candidate) index range instead of the candidate range topo.py reads.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


def first_order_saliency(model: nn.Module, bundle, group_base_size: int) -> torch.Tensor:
    """|<u_j,v_j>| per real unit j in [0, group_base_size) of this group,
    summed across all of the group's consumers (mirrors Theorem 5.7's
    concatenated-port sum for the candidate side)."""
    modules = dict(model.named_modules())
    device = next(model.parameters()).device
    total = torch.zeros(group_base_size, device=device)
    for slot in bundle.consumer_slots:
        mod = modules[slot.module_name]
        w, g = mod.weight, mod.weight.grad
        if g is None:
            continue
        mult = slot.multiplicity
        real_cols = w.data[:, : group_base_size * mult]
        real_grad = g[:, : group_base_size * mult]
        prod = (real_cols * real_grad)
        # sum over the out axis and over each unit's multiplicity block
        prod = prod.reshape(w.shape[0], group_base_size, mult, *w.shape[2:])
        total = total + prod.sum(dim=tuple(d for d in range(prod.dim()) if d != 1))
    return total.abs()


class ActivationStats:
    """Running per-channel E[a_j^2] for each group's producing tensor.

    Needed for the second-order half of Eq. (14). Registers a forward hook
    on one producer module per group and accumulates the mean square of
    each channel over the batch and any spatial/temporal axes.
    """

    def __init__(self, model: nn.Module, bank) -> None:
        self.sums: dict = {}
        self.counts: dict = {}
        self._handles = []
        modules = dict(model.named_modules())
        for tap, h in bank.handles.items():
            prod = h.bundle.producer_slots[0].module_name
            mod = modules.get(prod)
            if mod is None:
                continue
            self._handles.append(mod.register_forward_hook(self._make_hook(tap, mod)))

    @staticmethod
    def _channel_axis(mod: nn.Module, out: torch.Tensor) -> int:
        """Which axis of this module's OUTPUT carries the group's channels.

        Conv/BatchNorm emit (B, C, *spatial) -> axis 1.
        Linear emits (..., out_features) -> the LAST axis, which for a
        sequence model is (B, L, D) with D at axis 2, NOT axis 1. Assuming
        axis 1 there silently averages over the feature dimension and
        returns per-timestep statistics: E[a_j^2] would be computed for the
        wrong index set entirely, and Eq. (14)'s second-order term would
        rank incumbents by noise.
        """
        if isinstance(mod, nn.Linear):
            return out.dim() - 1
        return 1

    def _make_hook(self, tap: int, owner: nn.Module):
        def hook(mod, inp, out):
            # `copy.deepcopy(model)` copies a module's _forward_hooks, but a
            # function object is not deep-copied -- so the copy's hook still
            # writes into THIS ActivationStats. A forward pass on any copied
            # model (e.g. a detached/stripped one, which has base_size
            # channels instead of base_size + k) would then silently
            # overwrite E[a_j^2] with statistics measured on a different
            # architecture, and Eq. (14)'s second-order term would rank
            # incumbents using the wrong model. Only accept the module this
            # instance actually registered on.
            if mod is not owner:
                return
            if not torch.is_tensor(out) or out.dim() < 2:
                return
            with torch.no_grad():
                axis = self._channel_axis(mod, out)
                dims = [d for d in range(out.dim()) if d != axis]
                ms = out.detach().pow(2).mean(dim=dims)
            prev = self.sums.get(tap)
            self.sums[tap] = ms if prev is None or prev.shape != ms.shape else prev + ms
            self.counts[tap] = 1 if prev is None or prev.shape != ms.shape else self.counts[tap] + 1
        return hook

    def mean_sq(self, tap: int) -> Optional[torch.Tensor]:
        s = self.sums.get(tap)
        if s is None:
            return None
        return s / max(self.counts.get(tap, 1), 1)

    def reset(self) -> None:
        self.sums.clear()
        self.counts.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def port_norm_sq(model: nn.Module, bundle, group_base_size: int) -> torch.Tensor:
    """||v_j||^2 per real unit j: the squared norm of unit j's outgoing
    weights, summed over every consumer of the group."""
    modules = dict(model.named_modules())
    device = next(model.parameters()).device
    total = torch.zeros(group_base_size, device=device)
    for slot in bundle.consumer_slots:
        mod = modules[slot.module_name]
        w = mod.weight
        mult = slot.multiplicity
        real = w.data[:, : group_base_size * mult]
        real = real.reshape(w.shape[0], group_base_size, mult, *w.shape[2:])
        total = total + real.pow(2).sum(dim=tuple(d for d in range(real.dim()) if d != 1))
    return total


def removal_score(first_order: torch.Tensor, mean_sq_activation: Optional[torch.Tensor],
                   H_max: Optional[float], v_j_norm_sq: Optional[torch.Tensor]) -> torch.Tensor:
    """Full RHS of (14): |<u_j,v_j>| + (H_max/2) E[a_j^2] ||v_j||^2.

    Remark 4.3: near a stationary point the first-order term vanishes for
    EVERY incumbent (u_j -> 0 since v_j is a trained parameter), so the
    second-order term is what actually ranks incumbents late in training.
    Pass None for the extra terms to get the first-order-only score (valid
    early/mid-training, per Remark 4.3(2)).
    """
    if mean_sq_activation is None or H_max is None or v_j_norm_sq is None:
        return first_order
    return first_order + 0.5 * H_max * mean_sq_activation * v_j_norm_sq
