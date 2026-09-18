"""Which coupling groups is it actually SAFE to resize?

The coupling engine (graph/groups.py) models module -> module edges. Two
things live outside that model, and both produce groups that look perfectly
resizable and are not:

  (N) Normalizer span. LayerNorm/GroupNorm compute statistics ACROSS the
      feature axis, so a dormant candidate perturbs every real channel
      through mu/sigma even with its output port at exactly zero --
      Lemma 2.2's function preservation fails and u_c stops being the
      topological derivative at the silent point. BatchNorm is per-channel
      and is fine (condition (N) holds exactly).

  (R) Shape-hardcoded reshapes. Multi-head attention splits a projection
      with `view(B, L, h, d_head)`, which silently requires |G| % h == 0.
      The engine sees `q -> o` as an ordinary producer/consumer pair and
      has no representation of QK^T or AV (activation-activation matmuls
      own no parameter slot). Resizing that group yields a model that
      crashes on the next forward pass:
          RuntimeError: shape '[2, 6, 4, 8]' is invalid for input of size 372

(N) is detectable statically from the bundle's affine slots. (R) is not --
it depends on arithmetic inside a `view` call. So this module checks (R)
the only way that is actually sound: it performs the edit on a throwaway
copy and runs a forward pass. Empirical, architecture-agnostic, and it
cannot be fooled by a topology nobody anticipated.

Run `probe_reallocatable` once at setup; a group that fails here would
otherwise fail hours into a training run.
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tsrx.graph.bundle import IndexBundle

_LN_GN_TYPES = (nn.LayerNorm, nn.GroupNorm)


def violates_norm_condition(bundle: IndexBundle, model: nn.Module) -> bool:
    """(N): does a LayerNorm/GroupNorm act on this group's axis?"""
    modules = dict(model.named_modules())
    return any(isinstance(modules.get(s.module_name), _LN_GN_TYPES)
               for s in bundle.affine_slots)


def survives_resize(build_fn, bundle_tap: int, example_input) -> Tuple[bool, str]:
    """(R): prune one index from this group on a FRESH model and run forward.

    `build_fn()` must return a newly constructed model; the edit is applied
    to that throwaway instance, never to the caller's model.
    """
    from tsrx.edit.edits import prune_group_index
    from tsrx.graph.bundle import build_all_bundles
    from tsrx.graph.groups import discover_groups
    from tsrx.graph.trace import trace_model

    try:
        m = build_fn()
        traced = trace_model(m.eval(), (example_input,))
        bundles = build_all_bundles(discover_groups(traced), m)
        bd = bundles.get(bundle_tap)
        if bd is None:
            return False, "tap not present"
        if bd.size <= 1:
            return False, "group too small to prune"
        prune_group_index(m, bd, idx=0)
        with torch.no_grad():
            m.eval()(example_input)
        return True, "ok"
    except Exception as e:  # noqa: BLE001 - any failure means "not safe"
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def probe_reallocatable(build_fn, example_input, verbose: bool = False) -> Dict[int, dict]:
    """Classify every coupling group of `build_fn()` as safe or not.

    Returns {tap: {"safe": bool, "reason": str, "size": int, "producers": [...]}}
    """
    from tsrx.graph.bundle import build_all_bundles
    from tsrx.graph.groups import discover_groups
    from tsrx.graph.trace import trace_model

    model = build_fn()
    traced = trace_model(model.eval(), (example_input,))
    bundles = build_all_bundles(discover_groups(traced), model)

    out: Dict[int, dict] = {}
    for tap, bd in sorted(bundles.items()):
        producers = sorted({s.module_name for s in bd.producer_slots})
        info = {"size": bd.size, "producers": producers, "safe": False, "reason": ""}

        if not bd.producer_slots or not bd.consumer_slots:
            info["reason"] = "no producer/consumer (model output; width is the task)"
        elif violates_norm_condition(bd, model):
            info["reason"] = "(N) LayerNorm/GroupNorm spans this axis"
        else:
            ok, why = survives_resize(build_fn, tap, example_input)
            info["safe"] = ok
            info["reason"] = "ok" if ok else f"(R) resize breaks forward: {why}"
        out[tap] = info
        if verbose:
            flag = "SAFE  " if info["safe"] else "unsafe"
            print(f"  tap {tap:>3} size={bd.size:>4} {flag} {producers} :: {info['reason']}")
    return out


def safe_taps(build_fn, example_input, verbose: bool = False) -> List[int]:
    return [t for t, i in probe_reallocatable(build_fn, example_input, verbose).items() if i["safe"]]
