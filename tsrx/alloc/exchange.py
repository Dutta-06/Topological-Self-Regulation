"""TSR-X Budget-Conserving Exchange Operator (Definition 4.5 & Theorem 4.6).

Two regimes, dispatched by `evaluate_structural_update` (the entry point;
`bench/train_tsrx.py` calls this, not `evaluate_exchange` directly):

  FEASIBILITY (deployed > B_t): the budget constraint is violated. Prune the
  argmin-rho unit in every prunable group, cheapest-damage-first, until
  projected deployed <= B_t or the batch cap is hit. This bypasses the
  relative-deadness test on purpose — under a shrinking budget, "prune
  something" is a constraint to satisfy, not a quality bar to clear. At most
  one prune per group per call, so no group's indices are invalidated mid
  batch (pruning group A never touches group B's own index range, even when
  A and B share a module on opposite axes — see `edits.prune_group_index`).
  Theorem 4.6's descent guarantee does NOT cover these prunes; they are
  honest constraint cost, not exchange.

  OPTIMALITY (deployed <= B_t): the original single-decision equimarginal
  logic (`evaluate_exchange`) — argmax growth density vs argmin removal
  density, exchange/grow/prune by Definition 4.5's accept rule.

Both regimes compare against the SAME `deployed_params` (from
`bank.deployed_params()`), never `sum(p.numel() for p in model.parameters())`
— the latter includes dormant candidate mass and made the optimality
regime's pure-growth case permanently unreachable under any budget.
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
    regime: str = ""  # "feasibility" | "optimality"
    reason: str = ""  # populated especially when action == "none"
    grow_tap: Optional[int] = None
    grow_cand_idx: Optional[int] = None
    grow_gamma: float = 0.0
    prune_tap: Optional[int] = None
    prune_idx: Optional[int] = None
    prune_rho: float = 0.0
    delta_loss_bound: float = 0.0
    details: dict = field(default_factory=dict)


def _gather_removal_options(
    bank: CandidateBank,
    saliency_sum: Dict[int, torch.Tensor],
    n_seen: int,
    min_size_per_group: int,
    act_stats=None,
    H_max: float = 1.0,
) -> List[dict]:
    """One removal option per prunable group (argmin real-unit saliency)."""
    model = bank.model
    options: List[dict] = []
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
        real_sal = avg_sal[: h.base_size]

        first_order_total = float(real_sal.sum().item())
        second_order_total = 0.0
        if act_stats is not None:
            ms = act_stats.mean_sq(tap)
            if ms is not None and ms.numel() >= h.base_size:
                vsq = port_norm_sq(model, h.bundle, h.base_size)
                scored = removal_score(real_sal, ms[: h.base_size].to(real_sal.device),
                                        H_max, vsq.to(real_sal.device))
                second_order_total = float((scored - real_sal).sum().item())
                real_sal = scored

        min_j = int(real_sal.argmin().item())
        min_rho_val = float(real_sal[min_j].item())
        rho = min_rho_val / kp
        sal_scale = float(real_sal.median().item()) if real_sal.numel() else 0.0
        denom = first_order_total + second_order_total
        first_order_share = (first_order_total / denom) if denom > 0 else 1.0

        options.append({
            "tap": tap,
            "unit_idx": min_j,
            "saliency": min_rho_val,
            "saliency_scale": sal_scale,
            "kappa": kp,
            "rho": rho,
            "size": h.base_size,
            "first_order_share": first_order_share,
        })
    return options


def _feasibility_prunes(
    bank: CandidateBank,
    saliency_sum: Dict[int, torch.Tensor],
    n_seen: int,
    deployed_params: int,
    budget_at_t: int,
    min_size_per_group: int,
    max_prunes_per_update: int,
    act_stats=None,
    H_max: float = 1.0,
) -> List[ExchangeDecision]:
    options = _gather_removal_options(bank, saliency_sum, n_seen, min_size_per_group, act_stats, H_max)
    # Cheapest damage-per-param-freed first (Remark 4.3's rho ordering still
    # applies here — feasibility only waives the DEADNESS test, not the
    # preference for low-damage removals).
    options.sort(key=lambda x: x["rho"])

    decisions: List[ExchangeDecision] = []
    projected = deployed_params
    for opt in options:
        if projected <= budget_at_t:
            break
        if len(decisions) >= max_prunes_per_update:
            break
        decisions.append(ExchangeDecision(
            action="pure_prune",
            regime="feasibility",
            reason="over_budget",
            prune_tap=opt["tap"],
            prune_idx=opt["unit_idx"],
            prune_rho=opt["rho"],
            delta_loss_bound=opt["kappa"] * opt["rho"],
            details=opt,
        ))
        projected -= opt["kappa"]

    if not decisions:
        reason = "min_size_reached" if not options else "budget_gap_exceeds_batch_cap"
        decisions = [ExchangeDecision(action="none", regime="feasibility", reason=reason,
                                       details={"deployed": deployed_params, "budget_at_t": budget_at_t})]
    return decisions


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
    deployed_params: Optional[int] = None,
) -> ExchangeDecision:
    """Optimality-regime core: argmax growth density vs argmin removal
    density, Definition 4.5's accept rule. Assumes the caller has already
    established `deployed_params <= budget_params` (or that there is no
    budget) — `evaluate_structural_update` is the entry point that enforces
    that split; call this directly only in tests / unconstrained mode.

    `deployed_params` should be `bank.deployed_params()`. Falling back to
    `sum(p.numel() for p in model.parameters())` (candidate-inflated) is
    kept only for callers that predate the dormancy-accounting fix.
    """
    model = bank.model
    current_params = deployed_params if deployed_params is not None else sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

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
    removal_options = _gather_removal_options(bank, saliency_sum, n_seen, min_size_per_group, act_stats, H_max)

    if not growth_options and not removal_options:
        return ExchangeDecision(action="none", regime="optimality", reason="no_options")

    growth_options.sort(key=lambda x: -x["gamma"])
    removal_options.sort(key=lambda x: x["rho"])

    best_g = growth_options[0] if growth_options else None
    worst_p = removal_options[0] if removal_options else None
    skip_reasons: List[str] = []

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
                regime="optimality",
                reason="accepted",
                grow_tap=best_g["tap"],
                grow_cand_idx=best_g["cand_idx"],
                grow_gamma=best_g["gamma"],
                delta_loss_bound=-best_g["kappa"] * best_g["gamma"],
                details=best_g,
            )
        skip_reasons.append("grow_over_budget" if not budget_ok_grow else "grow_gain_below_delta")

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
                regime="optimality",
                reason="accepted",
                prune_tap=worst_p["tap"],
                prune_idx=worst_p["unit_idx"],
                prune_rho=worst_p["rho"],
                delta_loss_bound=abs_cost,
                details=worst_p,
            )
        skip_reasons.append("not_relatively_dead")

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
                regime="optimality",
                reason="accepted",
                grow_tap=best_g["tap"],
                grow_cand_idx=best_g["cand_idx"],
                grow_gamma=best_g["gamma"],
                prune_tap=worst_p["tap"],
                prune_idx=worst_p["unit_idx"],
                prune_rho=worst_p["rho"],
                delta_loss_bound=cost - gain,
                details={"grow": best_g, "prune": worst_p},
            )
        skip_reasons.append("exchange_over_budget" if not budget_ok else "exchange_gain_below_cost")

    return ExchangeDecision(action="none", regime="optimality",
                             reason=";".join(skip_reasons) or "no_options",
                             details={"best_growth": best_g, "worst_removal": worst_p})


def evaluate_structural_update(
    bank: CandidateBank,
    windowed_signal: WindowedSignal,
    saliency_sum: Dict[int, torch.Tensor],
    n_seen: int,
    deployed_params: int,
    budget_at_t: Optional[int],
    delta: float = 1e-7,
    prune_tolerance: float = 1e-3,
    min_size_per_group: int = 8,
    max_size_per_group: int = 1024,
    act_stats=None,
    H_max: float = 1.0,
    max_prunes_per_update: int = 4,
) -> List[ExchangeDecision]:
    """Entry point: dispatch feasibility vs optimality regime by comparing
    `deployed_params` (bank.deployed_params(), NOT candidate-inflated) to
    `budget_at_t` (the annealed cap from `schedule.budget_at`, or None for
    unconstrained). Always returns at least one ExchangeDecision, even when
    action == "none", so every update leaves a reason in the decision trace.
    """
    if budget_at_t is not None and deployed_params > budget_at_t:
        return _feasibility_prunes(
            bank, saliency_sum, n_seen, deployed_params, budget_at_t,
            min_size_per_group, max_prunes_per_update, act_stats, H_max,
        )

    dec = evaluate_exchange(
        bank=bank,
        windowed_signal=windowed_signal,
        saliency_sum=saliency_sum,
        n_seen=n_seen,
        budget_params=budget_at_t,
        delta=delta,
        prune_tolerance=prune_tolerance,
        min_size_per_group=min_size_per_group,
        max_size_per_group=max_size_per_group,
        act_stats=act_stats,
        H_max=H_max,
        deployed_params=deployed_params,
    )
    return [dec]


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
