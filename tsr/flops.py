"""
Lightweight FLOPs counter for TSR networks.

Counts theoretical inference FLOPs (what a properly sparse implementation
would use) based on effective neuron/channel counts, not the full dense
tensor sizes. This is the standard approach in the sparse training literature
(RigL does the same).

Also tracks cumulative training FLOPs over the training trajectory, accounting
for the changing topology size at each step. This is the key measurement for
Benchmark 5 (time-to-accuracy).

FLOPs convention:
  - One multiply-accumulate = 2 FLOPs (1 multiply + 1 add)
  - Backward pass ≈ 2× forward pass FLOPs
  - Total per step = 3 × forward_flops × batch_size (forward + backward)
"""

from typing import Optional

import torch.nn as nn

from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_conv import TSRConv2d


def compute_layer_flops(module: nn.Module, input_shape: tuple) -> int:
    """Compute theoretical inference FLOPs for a single layer.

    Uses effective (gated) neuron/channel counts for TSR layers.
    For non-TSR layers, uses full dimensions.

    Args:
        module: A layer module.
        input_shape: Shape of the input tensor (excluding batch dim).

    Returns:
        Theoretical FLOPs for one sample.
    """
    if isinstance(module, TSRLinear):
        # FLOPs = 2 × in_features × effective_out_features
        effective_out = module.effective_neurons()
        flops = 2 * module.in_features * effective_out
        if module.bias is not None:
            flops += effective_out  # bias addition
        return flops

    elif isinstance(module, TSRConv2d):
        # FLOPs = 2 × in_channels × effective_out_channels × kH × kW × oH × oW
        effective_out = module.effective_channels()
        kh, kw = module.kernel_size
        # Compute output spatial dimensions
        if len(input_shape) >= 2:
            h_in, w_in = input_shape[-2], input_shape[-1]
            sh, sw = module.stride
            ph, pw = module.padding
            h_out = (h_in + 2 * ph - kh) // sh + 1
            w_out = (w_in + 2 * pw - kw) // sw + 1
        else:
            h_out, w_out = 1, 1

        flops = 2 * module.in_channels * effective_out * kh * kw * h_out * w_out
        if module.bias is not None:
            flops += effective_out * h_out * w_out
        return flops

    elif isinstance(module, nn.Linear):
        return 2 * module.in_features * module.out_features + (
            module.out_features if module.bias is not None else 0
        )

    elif isinstance(module, nn.Conv2d):
        kh, kw = module.kernel_size
        if len(input_shape) >= 2:
            h_in, w_in = input_shape[-2], input_shape[-1]
            sh, sw = module.stride
            ph, pw = module.padding
            h_out = (h_in + 2 * ph - kh) // sh + 1
            w_out = (w_in + 2 * pw - kw) // sw + 1
        else:
            h_out, w_out = 1, 1
        return 2 * module.in_channels * module.out_channels * kh * kw * h_out * w_out

    return 0


def compute_model_flops(model: nn.Module, input_shape: tuple) -> int:
    """Compute total theoretical inference FLOPs for a TSR model.

    Walks all layers and sums their FLOPs. Uses effective (gated) counts
    for TSR layers.

    Args:
        model: The TSR model.
        input_shape: Input shape (C, H, W) for images.

    Returns:
        Total theoretical inference FLOPs for one sample.
    """
    import torch.nn.functional as F

    total_flops = 0
    current_shape = input_shape  # (C, H, W)

    # Walk through model layers in order
    for name, module in model.named_modules():
        if isinstance(module, (TSRLinear, nn.Linear)):
            layer_flops = compute_layer_flops(module, current_shape)
            total_flops += layer_flops
            # Update shape
            if isinstance(module, TSRLinear):
                current_shape = (module.out_features,)
            else:
                current_shape = (module.out_features,)

        elif isinstance(module, (TSRConv2d, nn.Conv2d)):
            layer_flops = compute_layer_flops(module, current_shape)
            total_flops += layer_flops
            # Update shape
            if isinstance(module, TSRConv2d):
                out_c = module.out_channels
                sh, sw = module.stride
                ph, pw = module.padding
                kh, kw = module.kernel_size
            else:
                out_c = module.out_channels
                sh, sw = module.stride
                ph, pw = module.padding
                kh, kw = module.kernel_size

            if len(current_shape) >= 2:
                h_in, w_in = current_shape[-2], current_shape[-1]
                h_out = (h_in + 2 * ph - kh) // sh + 1
                w_out = (w_in + 2 * pw - kw) // sw + 1
                current_shape = (out_c, h_out, w_out)

    return total_flops


class CumulativeFLOPsTracker:
    """Tracks cumulative training FLOPs over the training trajectory.

    At each training step, adds the FLOPs for that step based on the
    current topology size. This accounts for TSR's changing per-step
    cost as the network grows/shrinks.

    Usage:
        tracker = CumulativeFLOPsTracker()

        for step in range(max_steps):
            # ... training step ...
            forward_flops = compute_model_flops(model, input_shape)
            tracker.record_step(forward_flops, batch_size)

        total = tracker.total_flops
        history = tracker.flops_history  # for plotting
    """

    def __init__(self):
        self.total_flops: int = 0
        self.structural_overhead_flops: int = 0
        self.flops_history: list = []  # (step, cumulative_flops)
        self._step = 0

    def record_step(
        self,
        forward_flops_per_sample: int,
        batch_size: int,
        is_structural_update: bool = False,
    ) -> None:
        """Record FLOPs for one training step.

        Training step FLOPs = 3 × forward_flops × batch_size
        (forward pass + backward pass, where backward ≈ 2× forward)

        Args:
            forward_flops_per_sample: Forward pass FLOPs for one sample.
            batch_size: Number of samples in the batch.
            is_structural_update: Whether this step included structural updates.
        """
        step_flops = 3 * forward_flops_per_sample * batch_size
        self.total_flops += step_flops
        self.flops_history.append((self._step, self.total_flops))
        self._step += 1

    def record_structural_overhead(self, overhead_flops: int) -> None:
        """Record FLOPs spent on structural update computations.

        This is tracked separately for honest reporting.

        Args:
            overhead_flops: Estimated FLOPs for the structural update.
        """
        self.structural_overhead_flops += overhead_flops
        self.total_flops += overhead_flops

    @property
    def overhead_percentage(self) -> float:
        """Structural overhead as a percentage of total training FLOPs."""
        if self.total_flops == 0:
            return 0.0
        return 100.0 * self.structural_overhead_flops / self.total_flops

    def state_dict(self) -> dict:
        return {
            "total_flops": self.total_flops,
            "structural_overhead_flops": self.structural_overhead_flops,
            "flops_history": self.flops_history,
            "step": self._step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.total_flops = state["total_flops"]
        self.structural_overhead_flops = state["structural_overhead_flops"]
        self.flops_history = state["flops_history"]
        self._step = state["step"]
