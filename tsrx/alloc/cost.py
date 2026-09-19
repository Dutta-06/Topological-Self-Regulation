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


def model_flops(model: nn.Module, example: torch.Tensor) -> int:
    """Forward FLOPs of `model` on one input of `example`'s shape, 2*MACs over
    convolutions and linear layers -- the convention used throughout the repo's
    reporting (scripts/regenerate_vision_summaries.py). Batch dimension is
    divided out."""
    total = 0
    handles = []

    def conv_hook(mod, inp, out):
        nonlocal total
        k = 1
        for d in mod.kernel_size:
            k *= d
        spatial = 1
        for d in out.shape[2:]:
            spatial *= d
        total += 2 * k * (mod.in_channels // mod.groups) * out.shape[1] * spatial

    def linear_hook(mod, inp, out):
        nonlocal total
        positions = 1
        for d in out.shape[1:-1]:
            positions *= d
        total += 2 * mod.in_features * mod.out_features * positions

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(example[:1])
    model.train(was_training)
    for h in handles:
        h.remove()
    return int(total)


def deployed_flops(bank, example: torch.Tensor) -> int:
    """Forward FLOPs of the DEPLOYED model: candidates stripped from a copy,
    counted exactly, the model itself untouched. The FLOP analogue of
    `CandidateBank.deployed_params()`."""
    return model_flops(bank.detached_copy(), example)


def make_cost_fn(mode: str, traced: TracedModel):
    """The per-index marginal cost the controller ranks by: `kappa_params`
    (the released default) or `kappa_flops` bound to the traced shapes, which
    do not change with width."""
    if mode == "params":
        return kappa_params
    if mode == "flops":
        return lambda bundle, model: kappa_flops(bundle, model, traced)
    raise ValueError(f"unknown cost mode {mode!r}; expected 'params' or 'flops'")


def cost_report(bundles: Dict[int, IndexBundle], model: nn.Module, traced: TracedModel) -> Dict[int, dict]:
    return {
        tap: {
            "kappa_params": kappa_params(bd, model),
            "kappa_flops": kappa_flops(bd, model, traced),
            "size": bd.size,
        }
        for tap, bd in bundles.items()
    }
