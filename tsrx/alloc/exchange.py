"""TSR-X Budget-Conserving Exchange Operator (Definition 4.5 & Theorem 4.6).

Finds the equimarginal fixed points of architecture space by:
  1. Locating the argmax growth candidate: ell' = argmax_ell gamma_ell
  2. Locating the argmin removal incumbent: ell = argmin_ell rho_ell
  3. Accepting the exchange if:
       gamma_ell' > (kappa_ell / kappa_ell') * rho_ell + delta / kappa_ell'
     and post-exchange cost satisfies C(n') <= B.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tsrx.alloc.cost import kappa_params
from tsrx.edit.edits import prune_group_index, materialize_candidate
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.saliency import port_norm_sq, removal_score
from tsrx.sense.topo import WindowedSignal


@dataclass
class ExchangeDecision:
    action: str  # "exchange" | "pure_grow" | "pure_prune" | "none"
    grow_tap: Optional[int] = None
    grow_cand_idx: Optional[int] = None
    grow_gamma: float = 0.0
    prune_tap: Optional[int] = None
    prune_idx: Optional[int] = None
    prune_rho: float = 0.0
    delta_loss_bound: float = 0.0
    details: dict = field(default_factory=dict)


def evaluate_exchange(
    bank: CandidateBank,
    windowed_signal: WindowedSignal,
    saliency_sum: Dict[int, torch.Tensor],
    n_seen: int,
    budget_params: Optional[int] = None,
    delta: float = 1e-7,
    prune_tolerance: float = 1e-3,
    min_size_per_group: int = 8,
    max_size_per_group: int = 1024,
    act_stats=None,
    H_max: float = 1.0,
) -> ExchangeDecision:
    """Evaluate candidate growth and incumbent removal across all coupling groups."""
    model = bank.model
    current_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 1. Gather growth densities gamma_ell for all groups
    growth_options: List[dict] = []
    for tap, h in bank.handles.items():
        if h.base_size >= max_size_per_group:
            continue
        kp = kappa_params(h.bundle, model)
        if kp <= 0:
            continue
        cand_u = windowed_signal.mean(tap)
        if cand_u is None:
            continue
        best_c = int(cand_u.argmax().item())
        best_u_val = float(cand_u[best_c].item())
        gamma = best_u_val / kp

        growth_options.append({
            "tap": tap,
            "cand_idx": best_c,
            "u_val": best_u_val,
            "kappa": kp,
            "gamma": gamma,
            "size": h.base_size,
        })

    # 2. Gather removal densities rho_ell for all groups
    removal_options: List[dict] = []
    for tap, h in bank.handles.items():
        if h.base_size <= min_size_per_group:
            continue
        sal_tensor = saliency_sum.get(tap)
        if sal_tensor is None:
            continue
        kp = kappa_params(h.bundle, model)
        if kp <= 0:
            continue
        avg_sal = sal_tensor / max(n_seen, 1)
        # Saliency only on real units [0, base_size)
        real_sal = avg_sal[: h.base_size]

        # Full Eq. (14): |<u_j,v_j>| + (H_max/2) E[a_j^2] ||v_j||^2.
        # Remark 4.3: v_j is a TRAINED parameter, so u_j -> 0 for every
        # incumbent as the inner problem converges and the first-order term
        # alone stops discriminating exactly when most pruning happens. The
        # second-order term survives, and it upper-bounds the true loss jump
        # from removal — which is what makes Theorem 4.6's descent guarantee
        # hold against a bound rather than against an underestimate.
        if act_stats is not None:
            ms = act_stats.mean_sq(tap)
            if ms is not None and ms.numel() >= h.base_size:
                vsq = port_norm_sq(model, h.bundle, h.base_size)
                real_sal = removal_score(real_sal, ms[: h.base_size].to(real_sal.device),
                                          H_max, vsq.to(real_sal.device))
        min_j = int(real_sal.argmin().item())
        min_rho_val = float(real_sal[min_j].item())
        rho = min_rho_val / kp
        # Reference scale for the RELATIVE pure-prune test: the group's own
        # median saliency. Median (not mean/max) so a few dominant units
        # can't mask a genuinely dead tail, and it degrades gracefully as
        # the whole group's saliency decays toward zero near convergence.
        sal_scale = float(real_sal.median().item()) if real_sal.numel() else 0.0

        removal_options.append({
            "tap": tap,
            "unit_idx": min_j,
            "saliency": min_rho_val,
            "saliency_scale": sal_scale,
            "kappa": kp,
            "rho": rho,
            "size": h.base_size,
        })

    if not growth_options and not removal_options:
        return ExchangeDecision(action="none")

    growth_options.sort(key=lambda x: -x["gamma"])
    removal_options.sort(key=lambda x: x["rho"])

    best_g = growth_options[0] if growth_options else None
    worst_p = removal_options[0] if removal_options else None

    # Case A: Budget Slack — Pure Growth (Theorem 4.7(ii)).
    # An absent budget means UNLIMITED slack, not "no growth allowed": gating
    # this on `budget_params is not None` made pure growth unreachable in
    # unconstrained mode, leaving prune+exchange as the only moves and biasing
    # every unconstrained run monotonically downward in width.
    if best_g:
        cost_after_grow = current_params + best_g["kappa"]
        budget_ok_grow = (budget_params is None) or (cost_after_grow <= budget_params)
        if budget_ok_grow and best_g["kappa"] * best_g["gamma"] > delta:
            return ExchangeDecision(
                action="pure_grow",
                grow_tap=best_g["tap"],
                grow_cand_idx=best_g["cand_idx"],
                grow_gamma=best_g["gamma"],
                delta_loss_bound=-best_g["kappa"] * best_g["gamma"],
                details=best_g,
            )

    # Case B: Pure Removal — unit contributes ~nothing.
    # The tolerance is RELATIVE to the group's own saliency scale, not an
    # absolute constant. |<u_j,v_j>| carries units of (loss x activation) that
    # differ per site (Remark 3.11), and u_j -> 0 for EVERY incumbent as the
    # inner problem converges (Remark 4.3), so any fixed absolute threshold is
    # eventually crossed by every unit in the network regardless of whether it
    # is actually useless — turning late training into indiscriminate pruning.
    if worst_p:
        scale = worst_p.get("saliency_scale", 0.0)
        abs_cost = worst_p["kappa"] * worst_p["rho"]
        relatively_dead = scale > 0 and (abs_cost / scale) < prune_tolerance
        if relatively_dead:
            return ExchangeDecision(
                action="pure_prune",
                prune_tap=worst_p["tap"],
                prune_idx=worst_p["unit_idx"],
                prune_rho=worst_p["rho"],
                delta_loss_bound=abs_cost,
                details=worst_p,
            )

    # Case C: Budget-Conserving Exchange (Definition 4.5 & Eq. 18)
    if best_g and worst_p:
        # Check accept condition: gamma_ell' > (kappa_ell / kappa_ell') * rho_ell + delta / kappa_ell'
        # Equivalent to: kappa_ell' * gamma_ell' > kappa_ell * rho_ell + delta
        gain = best_g["kappa"] * best_g["gamma"]
        cost = worst_p["kappa"] * worst_p["rho"]

        net_delta = current_params + best_g["kappa"] - worst_p["kappa"]
        budget_ok = (budget_params is None) or (net_delta <= budget_params)

        if gain > cost + delta and budget_ok:
            return ExchangeDecision(
                action="exchange",
                grow_tap=best_g["tap"],
                grow_cand_idx=best_g["cand_idx"],
                grow_gamma=best_g["gamma"],
                prune_tap=worst_p["tap"],
                prune_idx=worst_p["unit_idx"],
                prune_rho=worst_p["rho"],
                delta_loss_bound=cost - gain,
                details={"grow": best_g, "prune": worst_p},
            )

    return ExchangeDecision(action="none")


def apply_exchange(
    decision: ExchangeDecision,
    bank: CandidateBank,
    optimizer: Optional[torch.optim.Optimizer] = None,
    eps: float = 1e-3,
) -> None:
    """Execute an accepted structural decision on the model."""
    if decision.action == "none":
        return

    # 1. Apply pruning first if requested
    if decision.action in ("exchange", "pure_prune"):
        assert decision.prune_tap is not None and decision.prune_idx is not None
        h_prune = bank.handles[decision.prune_tap]
        prune_group_index(bank.model, h_prune.bundle, idx=decision.prune_idx, optimizer=optimizer, bank=bank)

    # 2. Apply growth if requested
    if decision.action in ("exchange", "pure_grow"):
        assert decision.grow_tap is not None and decision.grow_cand_idx is not None
        materialize_candidate(bank, tap=decision.grow_tap, cand_idx=decision.grow_cand_idx, eps=eps, optimizer=optimizer)
