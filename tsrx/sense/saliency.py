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
