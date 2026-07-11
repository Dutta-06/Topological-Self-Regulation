"""
Utility functions for TSR: optimizer management after structural changes.

The central problem: when we add or remove neurons, the optimizer's internal
state (momentum buffers, adaptive learning rate accumulators) references old
parameter tensors with old shapes. Simply calling add_param_group() accumulates
stale references. We need to either:
  (a) Rebuild the optimizer from scratch, preserving state for unchanged params
  (b) Surgically update the optimizer's internal state

We implement (a) because it's simpler and more reliable. The cost of optimizer
rebuild every K=200 steps is negligible.
"""

from typing import Dict, List, Optional, Tuple, Type
import copy

import torch
import torch.nn as nn
from torch.optim import Optimizer


def rebuild_optimizer(
    model: nn.Module,
    optimizer_cls: Type[Optimizer],
    optimizer_kwargs: dict,
    old_optimizer: Optional[Optimizer] = None,
    gate_lr_multiplier: float = 1.0,
    act_lr_multiplier: float = 0.1,
) -> Optimizer:
    """Rebuild optimizer from scratch with proper parameter groups.

    Creates separate parameter groups for:
      1. Standard weights and biases (base LR)
      2. Gate parameters (gate_lr_multiplier × base LR)
      3. Activation mixture weights (act_lr_multiplier × base LR)

    If an old optimizer is provided, preserves momentum/adaptive state
    for parameters whose shapes haven't changed (identified by data_ptr).

    Args:
        model: The TSR model with possibly changed structure.
        optimizer_cls: Optimizer class (e.g., torch.optim.Adam).
        optimizer_kwargs: Keyword arguments for the optimizer (lr, weight_decay, etc.).
        old_optimizer: Previous optimizer to salvage state from (optional).
        gate_lr_multiplier: LR multiplier for gate parameters.
        act_lr_multiplier: LR multiplier for activation weight parameters.

    Returns:
        Fresh optimizer instance with correct parameter references.
    """
    base_lr = optimizer_kwargs.get("lr", 0.001)

    # Separate parameters into groups
    weight_params = []
    gate_params = []
    act_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "gate" in name:
            gate_params.append(param)
        elif "act_weights" in name:
            act_params.append(param)
        else:
            weight_params.append(param)

    param_groups = []
    if weight_params:
        param_groups.append({
            "params": weight_params,
            "lr": base_lr,
            "group_name": "weights",
        })
    if gate_params:
        param_groups.append({
            "params": gate_params,
            "lr": base_lr * gate_lr_multiplier,
            "group_name": "gates",
        })
    if act_params:
        param_groups.append({
            "params": act_params,
            "lr": base_lr * act_lr_multiplier,
            "group_name": "activations",
        })

    # Remove 'lr' from kwargs since it's in param_groups
    kwargs = {k: v for k, v in optimizer_kwargs.items() if k != "lr"}
    new_optimizer = optimizer_cls(param_groups, **kwargs)

    # Attempt to preserve optimizer state for unchanged parameters
    if old_optimizer is not None:
        _transfer_optimizer_state(old_optimizer, new_optimizer)

    return new_optimizer


def _transfer_optimizer_state(
    old_optimizer: Optimizer,
    new_optimizer: Optimizer,
) -> None:
    """Transfer optimizer state from old to new for parameters that are
    literally the same object (identity match), not just same-shape-at-the-
    same-position.

    A structural event (grow/prune/insert) only reassigns `self.weight = nn.
    Parameter(...)` etc. on the LAYER(S) IT ACTUALLY TOUCHES — every other
    layer's parameter tensors are untouched Python objects. But
    `rebuild_optimizer` re-walks `model.named_parameters()` from scratch, so
    if a layer is inserted/grown BEFORE another layer in module order, every
    param after it shifts position within its param group's `params` list.
    The previous (group_name, position_index, shape) key was therefore
    silently matching an untouched param's old momentum to the WRONG
    (different, coincidentally same-shape) param whenever an event landed
    upstream of it — corrupting Adam state on nearly every structural event
    instead of just re-initializing it. Matching by `id(param)` instead is
    exact: an unchanged layer's weight is the same object before and after,
    so its state transfers; a grown/replaced layer's weight is a brand new
    object, so it correctly falls through to fresh optimizer-default state.

    This is still best-effort — it's okay if some state is lost. The
    optimizer will re-accumulate momentum/adaptive stats within a few steps.
    """
    old_state = old_optimizer.state
    old_state_by_id: Dict[int, dict] = {
        id(param): state for param, state in old_state.items()
    }

    transferred = 0
    for group in new_optimizer.param_groups:
        for param in group["params"]:
            old_entry = old_state_by_id.get(id(param))
            if old_entry is None:
                continue
            new_entry = {}
            for state_key, state_val in old_entry.items():
                if isinstance(state_val, torch.Tensor):
                    if state_val.dim() == 0:
                        # Scalar tensor (e.g., 'step' in Adam) — always copy
                        new_entry[state_key] = state_val.clone()
                    elif state_val.shape == param.shape:
                        new_entry[state_key] = state_val.clone()
                    # If shape doesn't match, skip (let optimizer re-init) —
                    # shouldn't happen under identity match, but stay safe.
                else:
                    # Non-tensor state (int, float, etc.) — always copy
                    new_entry[state_key] = state_val
            if new_entry:
                new_optimizer.state[param] = new_entry
                transferred += 1


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_effective_parameters(model: nn.Module) -> int:
    """Count effective parameters (gated neurons contribute proportionally).

    For TSR layers, each neuron's parameters are weighted by its gate value.
    A neuron with gate ≈ 0 effectively contributes 0 parameters.
    A neuron with gate ≈ 1 contributes its full parameter count.

    This gives the "effective" parameter count that would correspond to
    a properly sparse implementation.
    """
    from tsr.layers.tsr_linear import TSRLinear
    from tsr.layers.tsr_conv import TSRConv2d

    total = 0.0
    accounted_modules = set()

    for name, module in model.named_modules():
        if isinstance(module, TSRLinear):
            gate_vals = module.gate_values()
            # Weight params: each output neuron contributes in_features + 1 (bias) params
            params_per_neuron = module.in_features + (1 if module.bias is not None else 0)
            total += (gate_vals * params_per_neuron).sum().item()
            # Gate param itself: 1 per neuron
            total += gate_vals.sum().item()
            # Activation weights: always active (shared across neurons)
            total += len(module.act_weights)
            accounted_modules.add(name)

        elif isinstance(module, TSRConv2d):
            gate_vals = module.gate_values()
            kh, kw = module.kernel_size
            params_per_channel = module.in_channels * kh * kw + (
                1 if module.bias is not None else 0
            )
            total += (gate_vals * params_per_channel).sum().item()
            total += gate_vals.sum().item()
            total += len(module.act_weights)
            accounted_modules.add(name)

    # Add parameters from non-TSR modules (norms, classifier heads, etc.)
    for name, param in model.named_parameters():
        # Check if this param belongs to an already-accounted TSR module
        is_accounted = any(name.startswith(m + ".") for m in accounted_modules)
        if not is_accounted and param.requires_grad:
            total += param.numel()

    return int(total)
