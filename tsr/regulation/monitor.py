"""
StructuralPlasticityMonitor: Tracks activation and gradient statistics
over a sliding window for each TSR layer.

This is the sensory system of the regulation engine. It registers hooks
on TSR layers to collect:
  - Per-neuron/channel mean activation magnitude
  - Per-neuron/channel activation variance
  - Per-neuron/channel mean gradient magnitude (backward pass)
  - Per-neuron/channel gradient variance

These statistics are consumed by the signal computation module to decide
when and where to prune, grow, or rewire.

Design note: We track both activations AND gradients. The research document
only monitored activations — but gradient magnitude is the more informative
growth signal (high gradient = layer is bottlenecked and would benefit from
more capacity).
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_conv import TSRConv2d


class LayerStats:
    """Statistics accumulator for a single TSR layer over a sliding window.

    Stores per-neuron (or per-channel) statistics from forward and backward
    passes. All tensors are on CPU to avoid GPU memory accumulation.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        # Sliding windows of per-neuron statistics
        self.activation_magnitudes: List[torch.Tensor] = []  # mean |activation| per neuron
        self.activation_variances: List[torch.Tensor] = []   # var of activation per neuron
        self.gradient_magnitudes: List[torch.Tensor] = []    # mean |gradient| per neuron
        self.gradient_variances: List[torch.Tensor] = []     # var of gradient per neuron

    def add_activation_stats(
        self, mean_magnitude: torch.Tensor, variance: torch.Tensor
    ) -> None:
        """Record activation statistics from one forward pass.

        Args:
            mean_magnitude: Per-neuron mean |activation|, shape (num_neurons,).
            variance: Per-neuron activation variance, shape (num_neurons,).
        """
        self.activation_magnitudes.append(mean_magnitude.detach().cpu())
        self.activation_variances.append(variance.detach().cpu())
        if len(self.activation_magnitudes) > self.window_size:
            self.activation_magnitudes.pop(0)
            self.activation_variances.pop(0)

    def add_gradient_stats(
        self, mean_magnitude: torch.Tensor, variance: torch.Tensor
    ) -> None:
        """Record gradient statistics from one backward pass.

        Args:
            mean_magnitude: Per-neuron mean |gradient|, shape (num_neurons,).
            variance: Per-neuron gradient variance, shape (num_neurons,).
        """
        self.gradient_magnitudes.append(mean_magnitude.detach().cpu())
        self.gradient_variances.append(variance.detach().cpu())
        if len(self.gradient_magnitudes) > self.window_size:
            self.gradient_magnitudes.pop(0)
            self.gradient_variances.pop(0)

    @property
    def is_ready(self) -> bool:
        """Whether enough samples have been collected for reliable statistics."""
        return len(self.activation_magnitudes) >= self.window_size

    def mean_activation_magnitude(self) -> Optional[torch.Tensor]:
        """Average per-neuron activation magnitude over the window."""
        if not self.activation_magnitudes:
            return None
        # Handle variable neuron counts by using only the most recent consistent size
        current_size = self.activation_magnitudes[-1].shape[0]
        compatible = [
            t for t in self.activation_magnitudes if t.shape[0] == current_size
        ]
        if not compatible:
            return None
        return torch.stack(compatible).mean(dim=0)

    def mean_gradient_magnitude(self) -> Optional[torch.Tensor]:
        """Average per-neuron gradient magnitude over the window."""
        if not self.gradient_magnitudes:
            return None
        current_size = self.gradient_magnitudes[-1].shape[0]
        compatible = [
            t for t in self.gradient_magnitudes if t.shape[0] == current_size
        ]
        if not compatible:
            return None
        return torch.stack(compatible).mean(dim=0)

    def mean_activation_variance(self) -> Optional[torch.Tensor]:
        """Average per-neuron activation variance over the window."""
        if not self.activation_variances:
            return None
        current_size = self.activation_variances[-1].shape[0]
        compatible = [
            t for t in self.activation_variances if t.shape[0] == current_size
        ]
        if not compatible:
            return None
        return torch.stack(compatible).mean(dim=0)

    def mean_gradient_variance(self) -> Optional[torch.Tensor]:
        """Average per-neuron gradient variance over the window."""
        if not self.gradient_variances:
            return None
        current_size = self.gradient_variances[-1].shape[0]
        compatible = [
            t for t in self.gradient_variances if t.shape[0] == current_size
        ]
        if not compatible:
            return None
        return torch.stack(compatible).mean(dim=0)

    def clear(self) -> None:
        """Reset all accumulated statistics."""
        self.activation_magnitudes.clear()
        self.activation_variances.clear()
        self.gradient_magnitudes.clear()
        self.gradient_variances.clear()


class StructuralPlasticityMonitor:
    """Monitors TSR layers via hooks and accumulates per-neuron statistics.

    Registers forward and backward hooks on all TSR layers in a model.
    The collected statistics drive the regulation rule R:
        R : (activation_stats, gradient_stats, loss_trajectory) → Δ_topology

    Usage:
        monitor = StructuralPlasticityMonitor(model, window=100)
        # ... training loop runs, hooks collect data ...
        stats = monitor.get_layer_stats("layer_name")
        dead = monitor.get_dead_neurons("layer_name", threshold=0.01)
        bottleneck = monitor.get_bottleneck_scores()
    """

    def __init__(self, model: nn.Module, window: int = 100):
        """
        Args:
            model: The TSR model to monitor.
            window: Number of forward/backward passes to average over.
        """
        self.model = model
        self.window = window
        self.layer_stats: Dict[str, LayerStats] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._tsr_layers: Dict[str, nn.Module] = {}

        # Loss trajectory (scalar per step)
        self.loss_history: List[float] = []
        self.loss_window = window

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on all TSR layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, (TSRLinear, TSRConv2d)):
                self.layer_stats[name] = LayerStats(self.window)
                self._tsr_layers[name] = module

                # Forward hook: track activations
                fh = module.register_forward_hook(self._make_forward_hook(name))
                self._hooks.append(fh)

                # Backward hook: track gradients of output
                bh = module.register_full_backward_hook(self._make_backward_hook(name))
                self._hooks.append(bh)

    def _make_forward_hook(self, layer_name: str):
        """Create a forward hook that records activation statistics."""
        def hook(module, input, output):
            with torch.no_grad():
                if output.dim() == 4:
                    # Conv output: (batch, channels, H, W) → per-channel stats
                    # Reduce over batch + spatial dims
                    mean_mag = output.abs().mean(dim=(0, 2, 3))  # (channels,)
                    var = output.var(dim=(0, 2, 3))               # (channels,)
                elif output.dim() == 2:
                    # Linear output: (batch, features) → per-neuron stats
                    mean_mag = output.abs().mean(dim=0)  # (features,)
                    var = output.var(dim=0)               # (features,)
                else:
                    return  # skip unexpected shapes

                self.layer_stats[layer_name].add_activation_stats(mean_mag, var)
        return hook

    def _make_backward_hook(self, layer_name: str):
        """Create a backward hook that records gradient statistics."""
        def hook(module, grad_input, grad_output):
            with torch.no_grad():
                if grad_output[0] is None:
                    return
                grad = grad_output[0]

                if grad.dim() == 4:
                    mean_mag = grad.abs().mean(dim=(0, 2, 3))
                    var = grad.var(dim=(0, 2, 3))
                elif grad.dim() == 2:
                    mean_mag = grad.abs().mean(dim=0)
                    var = grad.var(dim=0)
                else:
                    return

                self.layer_stats[layer_name].add_gradient_stats(mean_mag, var)
        return hook

    def record_loss(self, loss_value: float) -> None:
        """Record a scalar loss value for loss trajectory analysis."""
        self.loss_history.append(loss_value)
        if len(self.loss_history) > self.loss_window:
            self.loss_history.pop(0)

    def get_layer_stats(self, layer_name: str) -> Optional[LayerStats]:
        """Get the statistics accumulator for a named layer."""
        return self.layer_stats.get(layer_name)

    def get_all_stats(self) -> Dict[str, LayerStats]:
        """Get statistics for all monitored layers."""
        return self.layer_stats

    def get_tsr_layers(self) -> Dict[str, nn.Module]:
        """Get all monitored TSR layers by name."""
        return self._tsr_layers

    def is_ready(self) -> bool:
        """Whether all layers have filled their monitoring windows."""
        return all(stats.is_ready for stats in self.layer_stats.values())

    def is_loss_plateau(self, patience: int = 50, threshold: float = 0.001) -> bool:
        """Check if loss has plateaued (no improvement in `patience` steps).

        Args:
            patience: Number of recent loss values to check.
            threshold: Minimum relative improvement to NOT be a plateau.

        Returns:
            True if loss has not improved by more than threshold fraction.
        """
        if len(self.loss_history) < patience:
            return False

        recent = self.loss_history[-patience:]
        first_half = sum(recent[: patience // 2]) / (patience // 2)
        second_half = sum(recent[patience // 2 :]) / (patience - patience // 2)

        if first_half == 0:
            return True

        relative_improvement = (first_half - second_half) / abs(first_half)
        return relative_improvement < threshold

    def reset_layer(self, layer_name: str) -> None:
        """Clear statistics for a specific layer (after structural change)."""
        if layer_name in self.layer_stats:
            self.layer_stats[layer_name].clear()

    def refresh_hooks(self) -> None:
        """Re-discover TSR layers and re-register hooks.

        Width changes resize parameters in place, so the module objects (and
        their hooks) survive. Depth changes (inserting a new block) add brand
        new modules that have no hooks and no LayerStats entry, and may shift
        the names of existing layers. After any such change, call this to
        rebuild the hook set against the current module tree. Statistics for
        layers that still exist by name are preserved; new layers start empty
        and removed layers are dropped.
        """
        self.remove_hooks()

        old_stats = self.layer_stats
        self.layer_stats = {}
        self._tsr_layers = {}

        for name, module in self.model.named_modules():
            if isinstance(module, (TSRLinear, TSRConv2d)):
                # Preserve accumulated stats if this name persists, else fresh.
                self.layer_stats[name] = old_stats.get(name, LayerStats(self.window))
                self._tsr_layers[name] = module

                fh = module.register_forward_hook(self._make_forward_hook(name))
                self._hooks.append(fh)
                bh = module.register_full_backward_hook(self._make_backward_hook(name))
                self._hooks.append(bh)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()
