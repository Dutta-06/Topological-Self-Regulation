"""C3 control: random reallocation to the same parameter budget.

C2 answers "is the discovered SHAPE good?". It cannot answer "did the
topological signal find it?" — a non-uniform width allocation might beat a
uniform one for reasons having nothing to do with u_c. C3 closes that: same
coupling groups, same total parameter count, widths drawn at RANDOM.

    TSR-X / C2 > C3   =>  the signal carries information
    TSR-X / C2 ~ C3   =>  any non-uniform allocation of this size would do,
                          and the sensing machinery is not earning its keep

C3 must differ from C2 in exactly one variable, so it reuses the same
coupling engine (`discover_groups` / `build_all_bundles`) and the same
rebuild path (`bench/resize.py::resize_model_to_widths`). The only
difference is where the width dict comes from.
"""

from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from bench.resize import resize_model_to_widths
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model


def attachable_taps(model: nn.Module, example_input) -> Dict[int, int]:
    """{tap: baseline_size} for every coupling group TSR-X could resize.

    Mirrors CandidateBank's eligibility rule: a group needs producers AND
    consumers. A group with no consumer is a model OUTPUT (the classifier
    head / forecast head) whose width is set by the task, not free capacity
    — resizing it would change the problem, not the architecture.
    """
    traced = trace_model(model.eval(), (example_input,))
    bundles = build_all_bundles(discover_groups(traced), model)
    return {
        tap: bd.size
        for tap, bd in bundles.items()
        if bd.size > 0 and bd.producer_slots and bd.consumer_slots
    }


def random_widths_at_budget(
    build_fn: Callable[[], nn.Module],
    example_input: torch.Tensor,
    target_params: int,
    min_size: int = 8,
    seed: int = 0,
    tol: float = 0.005,
    max_iter: int = 60,
    spread: float = 0.5,
) -> Dict[str, int]:
    """Random per-group widths whose rebuilt model has ~`target_params`.

    Draws a per-group factor w_i ~ U(1-spread, 1+spread), then binary-searches
    a single global scale s so that widths round(base_i * w_i * s) reproduce
    the target parameter count. Binary search (rather than solving in closed
    form) because parameter count is a non-linear, integer-rounded function of
    the widths once groups feed each other.

    Returns {str(tap): width}, the format `resize_model_to_widths` expects.
    """
    rng = np.random.default_rng(seed)
    base = attachable_taps(build_fn(), example_input)
    taps = sorted(base)
    factors = rng.uniform(1.0 - spread, 1.0 + spread, size=len(taps))

    def widths_at(scale: float) -> Dict[str, int]:
        return {
            str(t): max(min_size, int(round(base[t] * f * scale)))
            for t, f in zip(taps, factors)
        }

    def params_at(scale: float) -> int:
        m = build_fn()
        m = resize_model_to_widths(m, widths_at(scale), example_input)
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    lo, hi = 0.05, 4.0
    if params_at(hi) < target_params:
        return widths_at(hi)  # cannot reach the target even fully grown

    best, best_err = widths_at(1.0), float("inf")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = params_at(mid)
        err = abs(p - target_params) / max(target_params, 1)
        if err < best_err:
            best, best_err = widths_at(mid), err
        if err <= tol:
            break
        if p > target_params:
            hi = mid
        else:
            lo = mid
    return best


def build_c3_model(
    build_fn: Callable[[], nn.Module],
    example_input: torch.Tensor,
    target_params: int,
    min_size: int = 8,
    seed: int = 0,
) -> tuple:
    """Return (model, widths, achieved_params, rel_error_vs_target)."""
    widths = random_widths_at_budget(
        build_fn, example_input, target_params, min_size=min_size, seed=seed
    )
    model = resize_model_to_widths(build_fn(), widths, example_input)
    achieved = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rel = abs(achieved - target_params) / max(target_params, 1)
    return model, widths, achieved, rel
