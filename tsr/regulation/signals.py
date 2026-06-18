"""
Signal computation for the TSR regulation engine.

Maps raw per-neuron statistics from the monitor into discrete structural
decisions:
  - Death signal: which neurons should be pruned
  - Bottleneck signal: which layers need more capacity
  - Layer insertion signal: where a new layer should be inserted

These signals implement the regulation rule:
    R : (activation_stats, gradient_stats, loss_trajectory) → Δ_topology

Design principle: signals are conservative. False positives (pruning a
useful neuron, growing when unnecessary) are more costly than false
negatives (keeping a dead neuron a bit longer). Thresholds are set to
favor stability.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsr.regulation.monitor import LayerStats, StructuralPlasticityMonitor
from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_conv import TSRConv2d


def gate_sparsity_penalty(model: nn.Module) -> torch.Tensor:
    """Differentiable L1 penalty on open gates across all TSR layers.

    Without this term, the only force acting on a gate logit is the task
    loss, which rarely drives a gate down the ~7.6 logits needed to cross
    the death threshold (sigmoid(gate) < 0.01). Adding a small multiple of
    this penalty to the training loss applies steady downward pressure on
    every gate, so that neurons the task does not actively need decay toward
    closed and become prunable. Useful neurons resist the pressure because
    the task gradient holds their gate open — this is the mechanism that
    makes pruning (and thus *bidirectional* structural change) actually work.

    The penalty is the mean of sigmoid(gate) over every gated neuron/channel
    in the network, so its scale is independent of network size.

    Args:
        model: A model containing TSRLinear / TSRConv2d layers.

    Returns:
        Scalar tensor (mean open-gate value). Zero if the model has no gates.
    """
    gate_sums = []
    gate_counts = 0
    for module in model.modules():
        if isinstance(module, (TSRLinear, TSRConv2d)):
            # Differentiable: sigmoid of the raw gate logits (not gate_values(),
            # which detaches). Summing keeps a single graph across layers.
            open_frac = torch.sigmoid(module.gate)
            gate_sums.append(open_frac.sum())
            gate_counts += open_frac.numel()

    if gate_counts == 0:
        return torch.zeros((), device=next(model.parameters()).device)

    return torch.stack(gate_sums).sum() / gate_counts


def compute_death_signal(
    layer_stats: LayerStats,
    layer: nn.Module,
    threshold: float = 0.01,
    min_neurons: int = 4,
    current_step: int = 0,
    newborn_protect_steps: int = 400,
) -> torch.Tensor:
    """Identify dead neurons/channels that should be pruned.

    A neuron is "dead" if:
      1. Its gate activation is below threshold (it's nearly off), AND
      2. Its mean activation magnitude over the monitoring window is below threshold

    Neurons grown during training (neuron_birth_step >= 0) are protected from pruning
    for newborn_protect_steps steps after birth — enough time for the task gradient
    to either lift the gate or confirm the neuron is genuinely useless.

    Args:
        layer_stats: Statistics from the monitoring window.
        layer: The TSR layer to inspect.
        threshold: Activation/gate threshold below which a neuron is dead.
        min_neurons: Minimum neurons to keep in the layer (never prune below this).
        current_step: Current training step.
        newborn_protect_steps: Steps after birth during which a neuron cannot be pruned.

    Returns:
        1D tensor of indices of dead neurons. Empty tensor if none are dead.
    """
    # Get current gate values
    if isinstance(layer, TSRLinear):
        gate_vals = layer.gate_values()
        num_neurons = layer.out_features
    elif isinstance(layer, TSRConv2d):
        gate_vals = layer.gate_values()
        num_neurons = layer.out_channels
    else:
        return torch.tensor([], dtype=torch.long)

    # Get mean activation magnitude from monitor
    mean_act = layer_stats.mean_activation_magnitude()
    if mean_act is None:
        return torch.tensor([], dtype=torch.long)

    # Ensure sizes match (stats might be from before a structural change)
    if mean_act.shape[0] != num_neurons:
        return torch.tensor([], dtype=torch.long)

    # Dead = gate low AND activation low
    is_dead = (gate_vals.cpu() < threshold) & (mean_act < threshold)

    # Protect newborns: neurons with birth_step >= 0 are grown neurons; skip them
    # until they've had newborn_protect_steps to earn their keep (or confirm uselessness).
    if newborn_protect_steps > 0 and hasattr(layer, 'neuron_birth_step'):
        birth = layer.neuron_birth_step.cpu()
        is_grown = birth >= 0  # -1 means original neuron → never protected
        age = torch.where(is_grown, current_step - birth, torch.full_like(birth, newborn_protect_steps + 1))
        is_dead = is_dead & (age >= newborn_protect_steps)

    dead_indices = is_dead.nonzero(as_tuple=True)[0]

    # Enforce minimum neuron count
    max_prunable = num_neurons - min_neurons
    if max_prunable <= 0:
        return torch.tensor([], dtype=torch.long)

    if len(dead_indices) > max_prunable:
        # Sort by activation magnitude (prune the most dead first)
        dead_activations = mean_act[dead_indices]
        _, sort_order = dead_activations.sort()
        dead_indices = dead_indices[sort_order[:max_prunable]]

    return dead_indices


def compute_bottleneck_signal(
    layer_stats: LayerStats,
    layer: nn.Module,
    max_neurons: int = 512,
) -> float:
    """Compute a scale-invariant bottleneck score for a layer (0 to ~1).

    A layer is bottlenecked when:
      1. High utilization: most neurons are active (at capacity)
      2. High gradient uniformity: gradient is spread evenly across neurons
         (the layer can't specialize further — it needs more capacity)
      3. High activation saturation: neurons are near their gate ceiling

    The score is:
        score = utilization × gradient_uniformity × saturation

    All components are [0, 1] normalized, making the score scale-invariant
    (independent of loss magnitude, learning rate, or batch size).

    Args:
        layer_stats: Statistics from the monitoring window.
        layer: The TSR layer to inspect.
        max_neurons: Maximum allowed neurons (no growth if already at max).

    Returns:
        Bottleneck score in [0, 1]. 0.0 if data is insufficient.
    """
    mean_grad = layer_stats.mean_gradient_magnitude()
    mean_act = layer_stats.mean_activation_magnitude()
    if mean_grad is None or mean_act is None:
        return 0.0

    if isinstance(layer, TSRLinear):
        total = layer.out_features
        effective = layer.effective_neurons()
        gate_vals = layer.gate_values()
    elif isinstance(layer, TSRConv2d):
        total = layer.out_channels
        effective = layer.effective_channels()
        gate_vals = layer.gate_values()
    else:
        return 0.0

    # Already at max capacity
    if total >= max_neurons:
        return 0.0

    # Ensure sizes match
    if mean_grad.shape[0] != total:
        return 0.0

    # ── Component 1: Utilization (fraction of neurons that are active) ──
    utilization = effective / max(total, 1)  # [0, 1]

    # ── Component 2: Gradient uniformity ──
    # High uniformity = all neurons carry similar load = bottleneck
    # Use 1 - normalized_std: if std is low relative to mean, uniformity is high
    grad_mean = mean_grad.mean().item()
    grad_std = mean_grad.std().item() if total > 1 else 0.0
    if grad_mean > 1e-10:
        # Coefficient of variation: std/mean. Low CV = high uniformity
        cv = grad_std / grad_mean
        gradient_uniformity = 1.0 / (1.0 + cv)  # [0.5, 1] for CV in [0, 1]
    else:
        gradient_uniformity = 0.0  # No gradient = no bottleneck signal

    # ── Component 3: Gate saturation ──
    # How many neurons have gates near their ceiling (> 0.9)?
    # High saturation = layer is maximally utilizing its existing capacity
    gate_vals_cpu = gate_vals.cpu()
    saturation = (gate_vals_cpu > 0.9).float().mean().item()  # [0, 1]

    # ── Combine ──
    score = utilization * gradient_uniformity * max(saturation, 0.1)  # floor saturation at 0.1
    return score


def compute_growth_neurons(
    bottleneck_score: float,
    current_width: int,
    growth_rate: float = 0.1,
    bottleneck_threshold: float = 0.1,
    max_neurons: int = 512,
) -> int:
    """Compute how many neurons to add based on bottleneck score.

    Growth is proportional to current width (multiplicative growth)
    and only triggered when bottleneck score exceeds threshold.

    Args:
        bottleneck_score: Output of compute_bottleneck_signal.
        current_width: Current neuron/channel count.
        growth_rate: Fraction of current width to add.
        bottleneck_threshold: Minimum bottleneck score to trigger growth.
        max_neurons: Maximum allowed neurons.

    Returns:
        Number of neurons to add (0 if no growth needed).
    """
    if bottleneck_score < bottleneck_threshold:
        return 0

    n = max(1, math.ceil(current_width * growth_rate))
    # Don't exceed max
    n = min(n, max_neurons - current_width)
    return max(0, n)


def compute_layer_insertion_signal(
    monitor: StructuralPlasticityMonitor,
) -> Tuple[Optional[str], float]:
    """Determine if and where a new layer should be inserted.

    Uses inter-layer gradient norm ratios to identify the point of
    maximum gradient bottleneck. Inspired by the topological derivative
    approach (§3.4 of the research document).

    The insertion point is between the two adjacent layers with the
    highest ratio of gradient norms:
        ratio(i) = grad_norm(layer_i+1) / grad_norm(layer_i)

    A high ratio means layer_i+1 sees much larger gradients than layer_i,
    suggesting the representational gap between them is too large.

    Args:
        monitor: The structural plasticity monitor.

    Returns:
        Tuple of (layer_name_to_insert_after, max_ratio).
        (None, 0.0) if insufficient data.
    """
    layers = monitor.get_tsr_layers()
    layer_names = list(layers.keys())

    if len(layer_names) < 2:
        return None, 0.0

    # Collect per-layer mean gradient norms
    grad_norms = {}
    for name in layer_names:
        stats = monitor.get_layer_stats(name)
        if stats is None:
            continue
        mean_grad = stats.mean_gradient_magnitude()
        if mean_grad is None:
            continue
        grad_norms[name] = mean_grad.mean().item()

    if len(grad_norms) < 2:
        return None, 0.0

    # Compute adjacent-layer gradient norm ratios
    ordered_names = [n for n in layer_names if n in grad_norms]
    max_ratio = 0.0
    best_name = None

    for i in range(len(ordered_names) - 1):
        name_i = ordered_names[i]
        name_j = ordered_names[i + 1]
        norm_i = grad_norms[name_i]
        norm_j = grad_norms[name_j]

        if norm_i > 0:
            ratio = norm_j / norm_i
        else:
            ratio = float("inf") if norm_j > 0 else 1.0

        if ratio > max_ratio:
            max_ratio = ratio
            best_name = name_i

    return best_name, max_ratio
