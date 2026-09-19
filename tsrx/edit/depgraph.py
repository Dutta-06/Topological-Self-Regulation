"""DepGraph (Fang et al., CVPR 2023) as a pruning baseline, through torch-pruning.

The other criteria in `tsrx.edit.pruning` remove channels with TSR-X's own
edit. This one hands the model to the DepGraph library instead: its
`DependencyGraph` discovers the coupled layers, its `GroupMagnitudeImportance`
(the paper's group-norm criterion, L2 aggregated over every layer of a group)
ranks them, and its `MetaPruner` performs the removal. What is kept from the
paper's protocol is the comparison itself:

  * the same plastic set -- every Conv/Linear whose output channels are not a
    producer of a TSR-X coupling group is passed to DepGraph as an ignored
    layer, so the classifier's task axis and the input stay fixed;
  * the same width floor -- a group is never pruned below `min_width`;
  * the same exact target -- pruning stops at the matched C2 deployed count,
    with the last group trimmed so the undershoot is under one channel's cost.

Sparse learning (the regulariser DepGraph can train with before pruning) is
deliberately not used: it changes the reference's training recipe, and the
comparison rests on every arm pruning the same reference.

Groups are ranked globally after per-group mean normalisation, as the other
criteria are (`global_pruning=True`; `local` prunes every group by the same
ratio instead). Iteration is by torch-pruning's own step schedule: each step
removes `step_fraction` of the initial channel count, importance recomputed.

`depgraph_group_report` checks, before anything is pruned, that DepGraph's
groups and TSR-X's coupling groups agree on the plastic set: the same producer
modules, grouped the same way. The driver refuses to run a cell where they do
not, since the count of the pruned model would then not be on TSR-X's lattice.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tsrx.alloc.cost import kappa_params
from tsrx.edit.pruning import deployed_params, editable_bundles
from tsrx.graph.bundle import IndexBundle

_ROOT_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


def _require_tp():
    try:
        import torch_pruning as tp
    except ImportError as e:  # pragma: no cover
        raise ImportError("the depgraph criterion needs torch-pruning: pip install 'torch-pruning>=1.5'") from e
    return tp


def producer_modules(bundles: Dict[int, IndexBundle]) -> Dict[str, int]:
    """module name -> tap, for every module whose output axis a bundle indexes."""
    out: Dict[str, int] = {}
    for tap, bd in bundles.items():
        for slot in bd.producer_slots:
            out.setdefault(slot.module_name, tap)
    return out


def ignored_layers(model: nn.Module, bundles: Dict[int, IndexBundle]) -> List[nn.Module]:
    """Every Conv/Linear whose output channels TSR-X may not edit."""
    allowed = producer_modules(bundles)
    return [m for name, m in model.named_modules()
            if isinstance(m, _ROOT_TYPES) and name not in allowed]


def depgraph_group_report(model: nn.Module, example: torch.Tensor,
                          bundles: Dict[int, IndexBundle]) -> dict:
    """Compare DepGraph's pruning groups with TSR-X's coupling groups on the
    plastic set. Returns {"matched": n, "mismatched": [...], "missing": [...]}:
    a DepGraph group is matched when the set of modules whose output it prunes
    equals the producer set of one bundle; a bundle is missing when no group
    has its producers."""
    tp = _require_tp()
    DG = tp.DependencyGraph().build_dependency(model.eval(), example_inputs=example)
    name_of = {m: n for n, m in model.named_modules()}
    by_producers = {}
    for tap, bd in bundles.items():
        by_producers[frozenset(s.module_name for s in bd.producer_slots)] = tap
    matched, mismatched, seen = 0, [], set()
    for group in DG.get_all_groups(ignored_layers=ignored_layers(model, bundles), root_module_types=_ROOT_TYPES):
        outs = frozenset(name_of[dep.target.module] for dep, _ in group
                         if DG.is_out_channel_pruning_fn(dep.handler) and dep.target.module in name_of
                         and isinstance(dep.target.module, _ROOT_TYPES))
        if outs in by_producers:
            matched += 1
            seen.add(by_producers[outs])
        else:
            mismatched.append(sorted(outs))
    missing = [t for t in bundles if t not in seen]
    return {"matched": matched, "mismatched": mismatched, "missing": missing}


def prune_depgraph_to_target(
    model: nn.Module,
    example: torch.Tensor,
    bundles: Dict[int, IndexBundle],
    target: int,
    min_width: int = 8,
    step_fraction: float = 0.005,
    global_pruning: bool = True,
    p: int = 2,
    log: Optional[Callable] = None,
) -> Tuple[List[dict], Dict[int, IndexBundle]]:
    """Prune `model` in place with DepGraph until deployed_params <= target.

    `bundles` is TSR-X's editable set on the unpruned model (`editable_bundles`);
    it fixes the ignored layers, the per-channel cost used to trim the final
    group, and the tap each event is attributed to. Returns (events, bundles
    re-traced on the pruned model), one event per pruned group.
    """
    tp = _require_tp()
    report = depgraph_group_report(model, example, bundles)
    if report["mismatched"] or report["missing"]:
        raise RuntimeError(f"DepGraph groups do not match TSR-X's coupling groups: {report}")

    allowed = producer_modules(bundles)
    name_of = {m: n for n, m in model.named_modules()}
    importance = tp.importance.GroupMagnitudeImportance(p=p, group_reduction="mean", normalizer="mean")
    max_ratio = 0.95
    steps = max(1, math.ceil(max_ratio / step_fraction))
    pruner = tp.pruner.MetaPruner(
        model.eval(), example, importance=importance, global_pruning=global_pruning,
        pruning_ratio=max_ratio, iterative_steps=steps, max_pruning_ratio=1.0,
        ignored_layers=ignored_layers(model, bundles),
    )

    events: List[dict] = []
    start = deployed_params(model)
    step = 0
    while deployed_params(model) > target:
        step += 1
        if step > steps:
            raise RuntimeError(f"DepGraph schedule exhausted at {deployed_params(model):,} params "
                               f"(target {target:,}); every group is at its floor")
        for group in pruner.step(interactive=True):
            current = deployed_params(model)
            if current <= target:
                break
            root = group[0].dep.target.module
            name = name_of[root]
            tap = allowed[name]
            proposed = list(group[0].idxs)
            width = pruner.DG.get_out_channels(root)
            headroom = width - min_width
            if headroom <= 0 or not proposed:
                continue
            kappa = kappa_params(bundles[tap], model)        # live shapes: exact per-channel cost now
            needed = math.ceil((current - target) / kappa) if kappa > 0 else len(proposed)
            n = min(len(proposed), headroom, needed)
            if n < len(proposed):
                # importance is defined on a full-width group; score every channel,
                # then keep the n least important of the ones DepGraph proposed
                full = pruner.DG.get_pruning_group(root, group[0].dep.handler, list(range(width)))
                scores = pruner.estimate_importance(full).detach().cpu().flatten()
                ranked = sorted(proposed, key=lambda i: float(scores[i]))[:n]
                group.prune(idxs=ranked)
            else:
                group.prune()
            after = deployed_params(model)
            events.append({"tap": tap, "module": name, "step": step, "removed": n,
                           "width_before": width, "width_after": width - n,
                           "params_before": current, "params_after": after})
        if log:
            log(f"  step {step}: {deployed_params(model):,} params ({sum(e['removed'] for e in events)} channels removed)")

    pruned = editable_bundles(model, example)
    if set(pruned) != set(bundles):
        raise RuntimeError("the pruned model does not re-trace to the same coupling groups")
    for tap, bd in pruned.items():
        if bd.size < min_width and bd.size < bundles[tap].size:
            raise RuntimeError(f"group {tap} was pruned below the width floor: {bd.size}")
    if log:
        log(f"pruned {start:,} -> {deployed_params(model):,} (target {target:,}, "
            f"{sum(e['removed'] for e in events)} channels removed in {len(events)} group edits)")
    return events, pruned
