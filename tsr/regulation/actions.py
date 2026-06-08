"""
Structural actions for the TSR regulation engine.

These functions execute the topology changes decided by the signal module:
  - prune_neurons_paired: Remove dead neurons and update adjacent layers
  - grow_neurons_paired: Add new neurons and update adjacent layers
  - apply_structural_update: Orchestrate a full structural update step

All actions handle the critical paired-layer update correctly:
  - Pruning layer L's output neurons → also prune layer L+1's input channels
  - Growing layer L's output neurons → also grow layer L+1's input channels
  - Norm layers between them are resized accordingly

Growth trigger logic:
  - Primary: absolute bottleneck score > threshold (for well-calibrated thresholds)
  - Fallback: if no absolute trigger fires but loss is plateauing, grow the
    most bottlenecked layer if its score > mean + 1*std of all layer scores.
    This ensures growth even with small absolute gradient magnitudes (common in
    tiny seed networks).
"""

from typing import Dict, List, Optional, Tuple
import logging
import math

import torch
import torch.nn as nn

from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_conv import TSRConv2d
from tsr.layers.tsr_norm import TSRGroupNorm
from tsr.regulation.monitor import StructuralPlasticityMonitor
from tsr.regulation.signals import (
    compute_death_signal,
    compute_bottleneck_signal,
    compute_growth_neurons,
)

logger = logging.getLogger(__name__)


class StructuralEvent:
    """Record of a structural change for logging and analysis."""

    def __init__(
        self,
        step: int,
        layer_name: str,
        action: str,
        details: dict,
    ):
        self.step = step
        self.layer_name = layer_name
        self.action = action  # "prune", "grow", "rewire", "insert_layer"
        self.details = details

    def __repr__(self):
        return (
            f"StructuralEvent(step={self.step}, layer={self.layer_name}, "
            f"action={self.action}, details={self.details})"
        )

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "layer_name": self.layer_name,
            "action": self.action,
            **self.details,
        }


def _get_layer_pair(
    model: nn.Module,
    layer_name: str,
) -> Tuple[Optional[nn.Module], Optional[nn.Module], Optional[nn.Module]]:
    """Find the TSR layer, its downstream TSR layer, and any norm layer between them.

    Walks the model's module tree in order to find:
      1. The target layer (by name)
      2. The next TSR layer after it (if any)
      3. Any TSRGroupNorm between them

    Args:
        model: The full TSR model.
        layer_name: Name of the layer to find the pair for.

    Returns:
        (target_layer, next_tsr_layer, norm_between) — any can be None.
    """
    tsr_modules = []
    norm_modules = []

    for name, module in model.named_modules():
        if isinstance(module, (TSRLinear, TSRConv2d)):
            tsr_modules.append((name, module))
        elif isinstance(module, TSRGroupNorm):
            norm_modules.append((name, module))

    # Find the target layer's index
    target_idx = None
    for i, (name, _) in enumerate(tsr_modules):
        if name == layer_name:
            target_idx = i
            break

    if target_idx is None:
        return None, None, None

    target_layer = tsr_modules[target_idx][1]

    # Find the next TSR layer (if exists)
    next_layer = None
    if target_idx + 1 < len(tsr_modules):
        next_layer = tsr_modules[target_idx + 1][1]

    # Find norm layer associated with target (usually right after it)
    # Convention: norm layers are named with the same prefix or appear
    # sequentially after their conv/linear layer
    norm_between = None
    target_found = False
    for name, module in model.named_modules():
        if name == layer_name:
            target_found = True
            continue
        if target_found:
            if isinstance(module, TSRGroupNorm):
                norm_between = module
                break
            if isinstance(module, (TSRLinear, TSRConv2d)):
                break  # Hit next TSR layer without finding norm

    return target_layer, next_layer, norm_between


def prune_neurons_paired(
    model: nn.Module,
    layer_name: str,
    indices: torch.Tensor,
    step: int,
) -> Optional[StructuralEvent]:
    """Prune neurons from a layer and correctly update all downstream dependencies.

    This is the safe version that handles:
      1. Removing output neurons/channels from the target layer
      2. Removing the corresponding input channels from the next layer
      3. Resizing any GroupNorm between them

    Args:
        model: The full TSR model.
        layer_name: Name of the layer to prune.
        indices: 1D tensor of neuron/channel indices to remove.
        step: Current training step (for logging).

    Returns:
        StructuralEvent describing what happened, or None if nothing changed.
    """
    if len(indices) == 0:
        return None

    target, next_layer, norm = _get_layer_pair(model, layer_name)
    if target is None:
        return None

    old_size = (
        target.out_features if isinstance(target, TSRLinear) else target.out_channels
    )

    # 1. Prune the target layer
    if isinstance(target, TSRLinear):
        target.prune_neurons(indices)
        new_size = target.out_features
    elif isinstance(target, TSRConv2d):
        target.prune_channels(indices)
        new_size = target.out_channels
    else:
        return None

    # 2. Update the next layer's input dimension
    if next_layer is not None:
        if isinstance(next_layer, TSRLinear) and isinstance(target, TSRConv2d):
            # Bridging Conv -> Linear: multiply by spatial size (4x4 = 16)
            linear_indices = []
            for idx in indices.tolist():
                linear_indices.extend(range(idx * 16, (idx + 1) * 16))
            next_layer.prune_input_channels(
                torch.tensor(linear_indices, device=indices.device)
            )
        elif isinstance(next_layer, (TSRLinear, TSRConv2d)):
            next_layer.prune_input_channels(indices)

    # 3. Resize any norm layer between them
    if norm is not None:
        norm.resize(new_size)

    event = StructuralEvent(
        step=step,
        layer_name=layer_name,
        action="prune",
        details={
            "indices": indices.tolist(),
            "old_size": old_size,
            "new_size": new_size,
            "num_pruned": len(indices),
        },
    )
    logger.info(
        f"Step {step}: Pruned {len(indices)} neurons from {layer_name} "
        f"({old_size} → {new_size})"
    )
    return event


def grow_neurons_paired(
    model: nn.Module,
    layer_name: str,
    n: int,
    step: int,
    init_scale: float = 0.001,
) -> Optional[StructuralEvent]:
    """Grow neurons in a layer and correctly update all downstream dependencies.

    New neurons are initialized "asleep" (gate ≈ 0) with small random weights.
    They must earn their place via gradient signal.

    Args:
        model: The full TSR model.
        layer_name: Name of the layer to grow.
        n: Number of neurons/channels to add.
        step: Current training step (for logging).
        init_scale: Scale for new weight initialization.

    Returns:
        StructuralEvent describing what happened, or None if nothing changed.
    """
    if n <= 0:
        return None

    target, next_layer, norm = _get_layer_pair(model, layer_name)
    if target is None:
        return None

    old_size = (
        target.out_features if isinstance(target, TSRLinear) else target.out_channels
    )

    # 1. Grow the target layer
    if isinstance(target, TSRLinear):
        target.grow_neurons(n, init_scale=init_scale)
        new_size = target.out_features
    elif isinstance(target, TSRConv2d):
        target.grow_channels(n, init_scale=init_scale)
        new_size = target.out_channels
    else:
        return None

    # 2. Update the next layer's input dimension
    if next_layer is not None:
        if isinstance(next_layer, TSRLinear) and isinstance(target, TSRConv2d):
            # Bridging Conv -> Linear: multiply by spatial size (4x4 = 16)
            next_layer.grow_input_channels(n * 16)
        elif isinstance(next_layer, (TSRLinear, TSRConv2d)):
            next_layer.grow_input_channels(n)

    # 3. Resize any norm layer between them
    if norm is not None:
        norm.resize(new_size)

    event = StructuralEvent(
        step=step,
        layer_name=layer_name,
        action="grow",
        details={
            "old_size": old_size,
            "new_size": new_size,
            "num_grown": n,
            "init_scale": init_scale,
        },
    )
    logger.info(
        f"Step {step}: Grew {n} neurons in {layer_name} ({old_size} → {new_size})"
    )
    return event


def apply_structural_update(
    model: nn.Module,
    monitor: StructuralPlasticityMonitor,
    step: int,
    # Death signal params
    death_threshold: float = 0.01,
    min_neurons: int = 4,
    # Growth signal params
    growth_enabled: bool = True,
    growth_rate: float = 0.1,
    max_neurons: int = 512,
    bottleneck_threshold: float = 1.5,
    init_scale: float = 0.001,
    depth_adaptation_enabled: bool = True,
) -> List[StructuralEvent]:
    """Execute one full structural update cycle across all layers.

    This is the main entry point called every K training steps.
    It queries the monitor for statistics, computes signals, and
    applies the appropriate structural actions.

    Order of operations:
      1. Compute death signals for all layers
      2. Prune dead neurons (with paired updates)
      3. Compute bottleneck signals for all layers
      4. Grow neurons in bottlenecked layers (with paired updates)
      5. Reset monitor statistics for modified layers

    Args:
        model: The full TSR model.
        monitor: The structural plasticity monitor with accumulated stats.
        step: Current training step.
        death_threshold: Gate/activation threshold for death detection.
        min_neurons: Minimum neurons per layer.
        growth_enabled: Whether to allow neuron growth.
        growth_rate: Fraction of current width to add when growing.
        max_neurons: Maximum neurons per layer.
        bottleneck_threshold: Minimum bottleneck score to trigger growth.
        init_scale: Weight initialization scale for new neurons.

    Returns:
        List of StructuralEvents that occurred.
    """
    events: List[StructuralEvent] = []

    if not monitor.is_ready():
        logger.debug(f"Step {step}: Monitor not ready, skipping structural update")
        return events

    tsr_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, (TSRLinear, TSRConv2d)):
            tsr_layers[name] = module

    modified_layers = set()

    # ── Phase 1: Pruning ──
    for layer_name, layer in tsr_layers.items():
        stats = monitor.get_layer_stats(layer_name)
        if stats is None or not stats.is_ready:
            continue

        dead_indices = compute_death_signal(
            stats, layer, threshold=death_threshold, min_neurons=min_neurons
        )
        if len(dead_indices) > 0:
            event = prune_neurons_paired(model, layer_name, dead_indices, step)
            if event is not None:
                events.append(event)
                modified_layers.add(layer_name)

    # ── Phase 2: Growth ──
    if growth_enabled:
        # Collect bottleneck scores for all layers
        layer_scores = {}
        layer_widths = {}

        for layer_name, layer in tsr_layers.items():
            stats = monitor.get_layer_stats(layer_name)
            if stats is None or not stats.is_ready:
                continue

            score = compute_bottleneck_signal(
                stats, layer, max_neurons=max_neurons
            )
            layer_scores[layer_name] = score

            if isinstance(layer, TSRLinear):
                layer_widths[layer_name] = layer.out_features
            elif isinstance(layer, TSRConv2d):
                layer_widths[layer_name] = layer.out_channels

        # Log all scores for diagnostics
        if layer_scores:
            scores_str = ", ".join(
                f"{name}={score:.4f}" for name, score in layer_scores.items()
            )
            logger.info(
                f"Step {step}: Bottleneck scores: [{scores_str}], "
                f"threshold={bottleneck_threshold:.4f}"
            )

        # Primary growth: absolute threshold
        grew_any = False
        for layer_name, score in layer_scores.items():
            n_grow = compute_growth_neurons(
                score,
                layer_widths[layer_name],
                growth_rate=growth_rate,
                bottleneck_threshold=bottleneck_threshold,
                max_neurons=max_neurons,
            )
            if n_grow > 0:
                event = grow_neurons_paired(
                    model, layer_name, n_grow, step, init_scale=init_scale
                )
                if event is not None:
                    events.append(event)
                    modified_layers.add(layer_name)
                    grew_any = True

        # Fallback growth: if no absolute trigger fired and loss is plateauing,
        # grow the most bottlenecked layer if it's a statistical outlier
        if not grew_any and len(layer_scores) >= 2 and monitor.is_loss_plateau():
            scores = list(layer_scores.values())
            mean_score = sum(scores) / len(scores)
            std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
            adaptive_threshold = mean_score + max(std_score, 1e-6)

            best_layer = max(layer_scores, key=layer_scores.get)
            best_score = layer_scores[best_layer]

            if best_score > adaptive_threshold and best_score > 0:
                n_grow = max(1, math.ceil(layer_widths[best_layer] * growth_rate))
                n_grow = min(n_grow, max_neurons - layer_widths[best_layer])
                if n_grow > 0:
                    logger.info(
                        f"Step {step}: Adaptive growth trigger — "
                        f"loss plateau + {best_layer} score {best_score:.4f} > "
                        f"adaptive threshold {adaptive_threshold:.4f}"
                    )
                    event = grow_neurons_paired(
                        model, best_layer, n_grow, step, init_scale=init_scale
                    )
                    if event is not None:
                        events.append(event)
                        modified_layers.add(best_layer)

    # ── Phase 3: Reset statistics for modified layers ──
    for layer_name in modified_layers:
        monitor.reset_layer(layer_name)

    if events:
        logger.info(
            f"Step {step}: {len(events)} structural events "
            f"({sum(1 for e in events if e.action == 'prune')} prune, "
            f"{sum(1 for e in events if e.action == 'grow')} grow)"
        )
    else:
        logger.debug(f"Step {step}: No structural events (scores below threshold)")

    return events
