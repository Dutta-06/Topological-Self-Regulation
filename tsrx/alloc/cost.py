"""Cost coefficients (Section 5.4, Eq. 25): kappa_G, the marginal cost of
one new index in group G, for both the parameter-count and inference-FLOPs
budgets. Computed exactly from the IndexBundle — no shape-threading
re-derivation of the kind that made the old tsr/flops.py silently wrong
for any non-sequential topology (it assumed named_modules() order ==
data-flow order; the coupling engine already knows the real graph).
"""

from typing import Dict

import torch
import torch.fx as fx
import torch.nn as nn

from tsrx.graph.bundle import IndexBundle
from tsrx.graph.trace import TracedModel, node_shape


def _other_axes_product(tensor: torch.Tensor, axis: int) -> int:
    p = 1
    for d, size in enumerate(tensor.shape):
        if d != axis:
            p *= size
    return p


def param_count(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def kappa_params(bundle: IndexBundle, model: nn.Module) -> float:
    """Marginal trainable-parameter cost of one new index in this group
    (Eq. 25, C = parameter count). Buffers (running_mean/var) are
    excluded automatically — they aren't nn.Parameter — matching the
    framework's "parameter tensors" scope; they still allocate memory but
    are not part of the reported params budget."""
    modules = dict(model.named_modules())
    total = 0.0
    seen = set()
    for slot in bundle.slots:
        key = (slot.module_name, slot.param_name)
        if key in seen:
            continue
        seen.add(key)
        mod = modules.get(slot.module_name)
        t = getattr(mod, slot.param_name, None) if mod is not None else None
        if not isinstance(t, nn.Parameter):
            continue
        total += _other_axes_product(t, slot.axis) * slot.multiplicity
    return total


def _module_output_shapes(traced: TracedModel) -> Dict[str, torch.Size]:
    shapes: Dict[str, torch.Size] = {}
    for node in traced.gm.graph.nodes:
        if node.op == "call_module":
            shp = node_shape(node)
            if shp is not None:
                shapes[node.target] = shp
    return shapes


def _spatial_size(shape: torch.Size) -> int:
    """Product of every axis after the channel axis (dim=1); 1 for a
    plain (B, C) tensor (Linear), H*W for conv2d, L for conv1d."""
    p = 1
    for d in shape[2:]:
        p *= int(d)
    return p


def kappa_flops(bundle: IndexBundle, model: nn.Module, traced: TracedModel) -> float:
    """Marginal inference-FLOPs cost of one new index (2*MACs convention,
    matching the rest of the repo's FLOPs accounting). Reads real output
    shapes from the traced model (ShapeProp) rather than re-deriving
    stride/padding/dilation arithmetic."""
    modules = dict(model.named_modules())
    out_shapes = _module_output_shapes(traced)
    total = 0.0
    seen = set()
    for slot in bundle.slots:
        key = (slot.module_name, slot.param_name)
        if key in seen:
            continue
        seen.add(key)
        mod = modules.get(slot.module_name)
        t = getattr(mod, slot.param_name, None) if mod is not None else None
        if not isinstance(t, nn.Parameter):
            continue
        shp = out_shapes.get(slot.module_name)
        spatial = _spatial_size(shp) if shp is not None else 1
        if slot.param_name == "bias":
            total += spatial * slot.multiplicity  # one add per output position
        else:
            total += 2 * _other_axes_product(t, slot.axis) * spatial * slot.multiplicity
    return total


def cost_report(bundles: Dict[int, IndexBundle], model: nn.Module, traced: TracedModel) -> Dict[int, dict]:
    return {
        tap: {
            "kappa_params": kappa_params(bd, model),
            "kappa_flops": kappa_flops(bd, model, traced),
            "size": bd.size,
        }
        for tap, bd in bundles.items()
    }
