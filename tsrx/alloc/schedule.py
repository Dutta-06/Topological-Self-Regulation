"""Annealed budget schedule for the two-regime controller (see
`tsrx/alloc/exchange.py` module docstring). Linear ramp from the baseline
param count down to the target, held after `end_frac * total_steps` —
continuous pressure instead of a cliff, so weights can recover between
feasibility prunes and the run is guaranteed to reach the target by a
known step.
"""

from typing import Optional


def budget_at(
    step: int,
    total_steps: int,
    base_params: int,
    target_params: Optional[int],
    end_frac: float = 0.5,
) -> Optional[int]:
    """Budget cap B(t) at `step`. None means unconstrained (no cap at all)."""
    if target_params is None:
        return None
    end_step = max(1, int(end_frac * total_steps))
    if step >= end_step:
        return target_params
    frac = step / end_step
    return int(round(base_params + frac * (target_params - base_params)))
