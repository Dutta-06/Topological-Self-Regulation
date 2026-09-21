"""Structured pruning on TSR-X's own coupling groups (the baseline's core).

Model-agnostic: works on any module the coupling engine can trace. Channels
are removed with `prune_group_index`, so every producer, affine and consumer
slot of a group is edited by index exactly as the controller edits it, and
the parameter count is the real tensor extent. Used by bench/prune_baseline.py
(vision) and bench/prune_baseline_ts.py (forecasting).

Importance criteria, each a {tap: tensor(size)} of per-channel scores:
    importance_l1        L1 norm of the channel's producer weights
    importance_bnscale   mean |gamma| over the group's affine (BatchNorm) scales;
                         falls back to L1 for groups without an affine slot
    make_importance_taylor(...)
                         |<u_j, v_j>| accumulated over minibatches via
                         tsrx.sense.saliency.first_order_saliency -- the
                         released controller's own removal statistic

`prune_to_target` ranks channels globally after per-group mean normalisation
and removes the least important until the deployed count is at or below the
target, recomputing importance after every round.
"""

from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from tsrx.alloc.cost import kappa_params
from tsrx.edit.edits import prune_group_index
from tsrx.graph.bundle import IndexBundle, build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.saliency import first_order_saliency


def deployed_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def editable_bundles(model: nn.Module, example: torch.Tensor,
                     allowed_taps: Optional[Iterable[int]] = None) -> Dict[int, IndexBundle]:
    """Exactly the groups CandidateBank attaches to: a producer and a consumer,
    quantum 1. `allowed_taps` further restricts to a probed-safe set (the
    forecasting harness's allowlist), so the baseline never touches a group
    TSR-X was denied."""
    traced = trace_model(model.eval(), (example,))
    bundles = build_all_bundles(discover_groups(traced), model)
    out = {t: b for t, b in bundles.items()
           if b.size and b.producer_slots and b.consumer_slots and b.quantum == 1}
    if allowed_taps is not None:
        allowed = set(allowed_taps)
        out = {t: b for t, b in out.items() if t in allowed}
    return out


def importance_l1(model: nn.Module, bundles: Dict[int, IndexBundle]) -> Dict[int, torch.Tensor]:
    modules = dict(model.named_modules())
    out = {}
    for tap, bd in bundles.items():
        total = torch.zeros(bd.size)
        seen = set()
        for slot in bd.producer_slots:
            if slot.module_name in seen:
                continue
            seen.add(slot.module_name)
            w = modules[slot.module_name].weight.detach().float().cpu()
            total += w.abs().reshape(w.shape[0], -1).sum(1)
        out[tap] = total
    return out


def importance_bnscale(model: nn.Module, bundles: Dict[int, IndexBundle]) -> Dict[int, torch.Tensor]:
    modules = dict(model.named_modules())
    fallback = importance_l1(model, bundles)
    out = {}
    for tap, bd in bundles.items():
        gammas, seen = [], set()
        for slot in bd.affine_slots:
            if slot.module_name in seen:
                continue
            seen.add(slot.module_name)
            mod = modules[slot.module_name]
            if getattr(mod, "weight", None) is not None:
                gammas.append(mod.weight.detach().float().cpu().abs())
        out[tap] = torch.stack(gammas).mean(0) if gammas else fallback[tap]
    return out


def make_importance_taylor(batches: Callable[[], Iterable], loss_of: Callable[[nn.Module, object], torch.Tensor],
                           n_batches: int) -> Callable:
    """Taylor importance from `n_batches` of `batches()`; `loss_of(model, batch)`
    returns the minibatch loss (moving the batch to the device itself)."""
    def importance(model, bundles):
        model.eval()                        # frozen normalisation statistics; gradients still flow
        acc = {tap: torch.zeros(bd.size) for tap, bd in bundles.items()}
        for i, batch in enumerate(batches()):
            if i >= n_batches:
                break
            model.zero_grad(set_to_none=True)
            loss_of(model, batch).backward()
            for tap, bd in bundles.items():
                acc[tap] += first_order_saliency(model, bd, bd.size).detach().cpu()
        model.zero_grad(set_to_none=True)
        return acc
    return importance


def prune_to_target(model: nn.Module, bundles: Dict[int, IndexBundle], target: int,
                    importance_fn: Callable, min_width: int = 8,
                    round_fraction: float = 0.05, log: Optional[Callable] = None,
                    tol: float = 0.01) -> List[dict]:
    """Remove the globally least important channels until deployed_params <= target.

    A round removes at most `round_fraction` of the channels still above the
    floor (at least one) before importance is recomputed, so index shifts and
    interactions are respected without a recompute per channel. Returns one
    record per removed channel.

    Endgame: a channel whose removal would land more than `tol` under the
    target is skipped in favour of the next-ranked channel that fits. Without
    this, a model with a very expensive group -- the channel-mixing TCN, whose
    head costs pred_len * n_vars per channel -- overshoots by one whole channel
    whenever the crossing removal lands in that group. Far from the target no
    candidate is skipped, so the order of removal is unchanged there. If nothing
    fits, the least-undershooting candidate is taken.
    """
    events: List[dict] = []
    start = deployed_params(model)
    while deployed_params(model) > target:
        scores = importance_fn(model, bundles)
        ranked = []
        for tap, bd in bundles.items():
            if bd.size <= min_width:
                continue
            s = scores[tap]
            norm = s / s.mean().clamp_min(1e-12)
            ranked.extend((float(norm[idx]), tap, idx) for idx in range(bd.size))
        if not ranked:
            raise RuntimeError("every editable group is at its width floor before the target was reached")
        ranked.sort()

        headroom = sum(max(0, bd.size - min_width) for bd in bundles.values())
        budget = max(1, int(round_fraction * headroom))
        removed_this_round: Dict[int, List[int]] = {}
        floor = target * (1.0 - tol)
        fallback = None                                      # (undershoot, score, tap, idx)
        for score, tap, idx in ranked:
            if budget == 0 or deployed_params(model) <= target:
                break
            taken = removed_this_round.setdefault(tap, [])
            if bundles[tap].size <= min_width:
                continue
            before = deployed_params(model)
            cost = kappa_params(bundles[tap], model)
            if before - cost < floor:                        # would cross the target by more than tol
                under = floor - (before - cost)
                if fallback is None or under < fallback[0]:
                    fallback = (under, score, tap, idx)
                continue
            shifted = idx - sum(1 for j in taken if j < idx)   # earlier removals shift later indices
            prune_group_index(model, bundles[tap], shifted)
            taken.append(idx)
            budget -= 1
            events.append({"tap": tap, "index": idx, "importance": score,
                           "params_before": before, "params_after": deployed_params(model)})
        if deployed_params(model) > target and not any(removed_this_round.values()):
            if fallback is None:
                raise RuntimeError("every editable group is at its width floor before the target was reached")
            _, score, tap, idx = fallback                    # nothing fits: take the smallest undershoot
            before = deployed_params(model)
            prune_group_index(model, bundles[tap], idx)
            events.append({"tap": tap, "index": idx, "importance": score,
                           "params_before": before, "params_after": deployed_params(model)})
        if log:
            log(f"  round: {deployed_params(model):,} params ({len(events)} removed)")
    if log:
        log(f"pruned {start:,} -> {deployed_params(model):,} (target {target:,}, "
            f"{len(events)} channels removed)")
    return events
