"""h_c(v): the second-order term of Eq. (11), estimated by the single-probe
method Remark 3.11 suggests — perturb the port to t*v, measure L(t), and
solve the quadratic model for h_c rather than computing a Hessian.

    L(t) = L(0) + t<u_c,v> + (t^2/2) h_c(v) + O(t^3)
    =>  h_c(v) = 2*(L(t) - L(0) - t*<u_c,v>) / t^2

Requires one extra forward pass (no backward) per group being scored, run
under torch.no_grad(); this is Prediction 9.1's use case too (a pure
forward-pass measurement, no training).
"""

from typing import Dict

import torch
import torch.nn as nn

from tsrx.sense.candidates import CandidateBank


@torch.no_grad()
def estimate_hc(bank: CandidateBank, tap: int, loss_fn, batch,
                 uc_norms: torch.Tensor, t_probe: float = 1e-4) -> torch.Tensor:
    """h_c(v*) per candidate at v* = -u_c/||u_c|| (the direction Theorem 3.4
    optimizes over), by probing at t_probe and solving the quadratic model.

    Args:
        bank: the attached CandidateBank.
        tap: which group to score.
        loss_fn: callable(model, batch) -> scalar loss tensor (no_grad-safe).
        batch: whatever `loss_fn` needs.
        uc_norms: this group's ||u_c|| per candidate (from topo.compute_uc_norms).
        t_probe: perturbation size. Sensitive: too large picks up O(t^3)
            curvature drift (measured 4.7 vs the converged 1.64 at
            t_probe=1e-2 on a real ResNet-18/BN block); too small hits
            float precision noise (this was verified to converge cleanly
            at 1e-4..1e-5 in float64 — in float32 training the noise
            floor is higher, so re-check this default against a direct
            finite-difference probe before trusting it at fp32/fp16).

    Returns:
        (k,) tensor of h_c estimates; entries where h_c<=0 (no valid
        quadratic model at this probe size — Assumption in Section 4.2's
        "standing assumption") are returned as NaN so callers can exclude
        them from the max in gamma_ell, per the framework text.
    """
    h = bank.handles[tap]
    modules = dict(bank.model.named_modules())
    L0 = loss_fn(bank.model, batch).item()

    out = torch.full((h.k,), float("nan"))
    for c in range(h.k):
        if uc_norms[c].item() == 0.0:
            continue
        saved = {}
        for slot in h.bundle.consumer_slots:
            mod = modules[slot.module_name]
            w = mod.weight
            n = h.k * slot.multiplicity
            start = w.shape[1] - n
            cslice = slice(start + c * slot.multiplicity, start + (c + 1) * slot.multiplicity)
            saved[slot.module_name] = w.data[:, cslice].clone()

            # direction v* = -u_c/||u_c|| applied uniformly across this
            # consumer's slice of the port (the readout is already the
            # true gradient direction at this slot, so reuse it directly
            # rather than re-deriving v* per slot).
            g = mod.weight.grad
            if g is not None:
                g_slice = g[:, cslice]
                v = -g_slice / uc_norms[c].clamp_min(1e-12)
                w.data[:, cslice] = t_probe * v

        Lt = loss_fn(bank.model, batch).item()

        for slot in h.bundle.consumer_slots:
            mod = modules[slot.module_name]
            n = h.k * slot.multiplicity
            start = mod.weight.shape[1] - n
            cslice = slice(start + c * slot.multiplicity, start + (c + 1) * slot.multiplicity)
            mod.weight.data[:, cslice] = saved[slot.module_name]

        u_dot_v = -uc_norms[c].item()  # <u_c, v*> = -||u_c|| by construction of v*
        hc = 2.0 * (Lt - L0 - t_probe * u_dot_v) / (t_probe ** 2)
        out[c] = hc if hc > 0 else float("nan")

    return out
