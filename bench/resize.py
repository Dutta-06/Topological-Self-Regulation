"""Rebuild an arbitrary model at recorded per-group widths — the shared
step behind every C2 (static-matched) control, vision or timeseries.

Model-agnostic: it runs the same coupling engine used during TSR-X
training (`discover_groups` / `build_all_bundles`) on whatever model and
example input it's given, then resizes every producer/affine/consumer
slot of each recorded group consistently (Definition 5.3). Nothing here
is specific to CNNs or classification.
"""

import torch
import torch.nn as nn

from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model


def resize_model_to_widths(model: nn.Module, widths: dict, example_input) -> nn.Module:
    """Rebuild `model` so every coupling group has the recorded width.

    Uses the same coupling engine as training, so the resize is applied
    consistently across every producer / affine / consumer slot of each
    group (Definition 5.3) rather than per-module.
    """
    traced = trace_model(model.eval(), (example_input,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, model)
    modules = dict(model.named_modules())

    for tap_str, target in widths.items():
        tap = int(tap_str)
        bd = bundles.get(tap)
        if bd is None or bd.size == target:
            continue

        seen = set()
        for slot in bd.producer_slots:
            if slot.module_name in seen:
                continue
            seen.add(slot.module_name)
            mod = modules[slot.module_name]
            w = mod.weight
            mod.weight = nn.Parameter(torch.empty(target, *w.shape[1:], device=w.device, dtype=w.dtype))
            nn.init.kaiming_uniform_(mod.weight.reshape(target, -1), a=5 ** 0.5)
            if getattr(mod, "bias", None) is not None:
                mod.bias = nn.Parameter(torch.zeros(target, device=w.device, dtype=w.dtype))
            for attr in ("out_channels", "out_features"):
                if hasattr(mod, attr):
                    setattr(mod, attr, target)

        seen = set()
        for slot in bd.affine_slots:
            if slot.module_name in seen:
                continue
            seen.add(slot.module_name)
            mod = modules[slot.module_name]
            dev = mod.weight.device if getattr(mod, "weight", None) is not None else "cpu"
            dt = mod.weight.dtype if getattr(mod, "weight", None) is not None else torch.float32
            if getattr(mod, "weight", None) is not None:
                mod.weight = nn.Parameter(torch.ones(target, device=dev, dtype=dt))
            if getattr(mod, "bias", None) is not None:
                mod.bias = nn.Parameter(torch.zeros(target, device=dev, dtype=dt))
            if getattr(mod, "running_mean", None) is not None:
                mod.running_mean = torch.zeros(target, device=dev, dtype=dt)
            if getattr(mod, "running_var", None) is not None:
                mod.running_var = torch.ones(target, device=dev, dtype=dt)
            for attr in ("num_features", "num_channels"):
                if hasattr(mod, attr):
                    setattr(mod, attr, target)

        seen = set()
        for slot in bd.consumer_slots:
            if slot.module_name in seen:
                continue
            seen.add(slot.module_name)
            mod = modules[slot.module_name]
            w = mod.weight
            new_in = target * slot.multiplicity
            mod.weight = nn.Parameter(torch.empty(w.shape[0], new_in, *w.shape[2:], device=w.device, dtype=w.dtype))
            nn.init.kaiming_uniform_(mod.weight.reshape(w.shape[0], -1), a=5 ** 0.5)
            for attr in ("in_channels", "in_features"):
                if hasattr(mod, attr):
                    setattr(mod, attr, new_in)

    return model
