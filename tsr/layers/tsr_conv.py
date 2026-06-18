"""
TSRConv2d: Convolutional layer with per-channel gating and activation mixing.

Same gating philosophy as TSRLinear but applied to output channels of a 2D
convolution. Each output channel has a learnable gate controlling its contribution.
The activation mixture is shared across all channels in the layer.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsr.layers.tsr_linear import ACTIVATION_FNS, ACTIVATION_NAMES, NUM_ACTIVATIONS


class TSRConv2d(nn.Module):
    """Conv2d with per-channel differentiable gating and learnable activation mixing.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (filters).
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides of the input. Default: 0.
        bias: If True, adds a learnable bias. Default: True.
        gate_init: Initial gate logit. Default 3.0 → sigmoid ≈ 0.95.
        act_init: Dominant activation at initialization. Default 'relu'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        gate_init: float = 3.0,
        act_init: str = "relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        # Core conv parameters: (out_channels, in_channels, kH, kW)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Per-channel gate logits
        self.gate = nn.Parameter(torch.full((out_channels,), gate_init))

        # Birth step per channel: -1 = original (never protected), >0 = grown at that step
        self.register_buffer('neuron_birth_step', torch.full((out_channels,), -1, dtype=torch.long))

        # Shared activation mixture logits
        act_idx = ACTIVATION_NAMES.index(act_init) if act_init in ACTIVATION_NAMES else 0
        act_logits = torch.zeros(NUM_ACTIVATIONS)
        act_logits[act_idx] = 3.0
        self.act_weights = nn.Parameter(act_logits)

        self._reset_parameters()

    def _reset_parameters(self):
        """Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            if fan_in > 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with channel gating and activation mixing.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).

        Returns:
            Output tensor of shape (batch, out_channels, H', W').
        """
        # Standard convolution
        h = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)

        # Per-channel gate: broadcast over spatial dims (batch, C, H, W)
        gate_values = torch.sigmoid(self.gate)  # (out_channels,)
        h = h * gate_values.view(1, -1, 1, 1)

        # Activation mixing
        act_mix = F.softmax(self.act_weights, dim=0)
        out = torch.zeros_like(h)
        for i, act_fn in enumerate(ACTIVATION_FNS):
            out = out + act_mix[i] * act_fn(h)

        return out

    # ------------------------------------------------------------------
    # Introspection methods
    # ------------------------------------------------------------------

    def gate_values(self) -> torch.Tensor:
        """Return current channel gate activations."""
        with torch.no_grad():
            return torch.sigmoid(self.gate)

    def effective_channels(self) -> int:
        """Count channels with gate activation above 0.5."""
        return int((self.gate_values() > 0.5).sum().item())

    def activation_distribution(self) -> dict:
        """Return the current activation mixture as a named dict."""
        with torch.no_grad():
            mix = F.softmax(self.act_weights, dim=0)
            return {name: mix[i].item() for i, name in enumerate(ACTIVATION_NAMES)}

    def dominant_activation(self) -> str:
        """Return the name of the dominant activation function."""
        with torch.no_grad():
            idx = self.act_weights.argmax().item()
            return ACTIVATION_NAMES[idx]

    # ------------------------------------------------------------------
    # Structural modification methods
    # ------------------------------------------------------------------

    def prune_channels(self, indices_to_remove: torch.Tensor) -> None:
        """Remove output channels at the given indices.

        NOTE: Caller must update downstream layer's input dimension.

        Args:
            indices_to_remove: 1D tensor of channel indices to remove.
        """
        if len(indices_to_remove) == 0:
            return

        keep_mask = torch.ones(self.out_channels, dtype=torch.bool)
        keep_mask[indices_to_remove] = False
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]

        self.weight = nn.Parameter(self.weight.data[keep_indices])
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.data[keep_indices])
        self.gate = nn.Parameter(self.gate.data[keep_indices])
        self.neuron_birth_step = self.neuron_birth_step[keep_indices]
        self.out_channels = len(keep_indices)

    def prune_input_channels(self, indices_to_remove: torch.Tensor) -> None:
        """Remove input channels (called when upstream layer prunes).

        Args:
            indices_to_remove: 1D tensor of input channel indices to remove.
        """
        if len(indices_to_remove) == 0:
            return

        keep_mask = torch.ones(self.in_channels, dtype=torch.bool)
        keep_mask[indices_to_remove] = False
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]

        self.weight = nn.Parameter(self.weight.data[:, keep_indices])
        self.in_channels = len(keep_indices)

    def grow_channels(self, n: int, init_scale: float = 0.001, newborn_gate: float = 0.0, step: int = 0) -> None:
        """Add n new output channels, born alive at newborn_gate logit.

        New channels start with phantom-initialized (or small random) weights and a
        gate logit of newborn_gate (default 0.0 → sigmoid≈0.5, alive and gradient-accessible).
        The caller should seed weights via _seed_grown_from_phantom after calling this.

        NOTE: Caller must update downstream layer's input dimension.

        Args:
            n: Number of channels to add.
            init_scale: Scale factor for weight initialization (overwritten by phantom seed).
            newborn_gate: Gate logit for new channels. 0.0 → sigmoid=0.5 (alive, not open).
            step: Current training step, recorded for newborn protection.
        """
        if n <= 0:
            return

        device = self.weight.device
        dtype = self.weight.dtype

        new_w = (
            torch.randn(n, self.in_channels, *self.kernel_size, device=device, dtype=dtype)
            * init_scale
        )
        new_g = torch.full((n,), newborn_gate, device=device, dtype=dtype)

        self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=0))
        self.gate = nn.Parameter(torch.cat([self.gate.data, new_g], dim=0))

        if self.bias is not None:
            new_b = torch.zeros(n, device=device, dtype=dtype)
            self.bias = nn.Parameter(torch.cat([self.bias.data, new_b], dim=0))

        new_birth = torch.full((n,), step, dtype=torch.long, device=device)
        self.neuron_birth_step = torch.cat([self.neuron_birth_step, new_birth])

        self.out_channels += n

    def grow_input_channels(self, n: int) -> None:
        """Add input channels (called when upstream layer grows).

        Args:
            n: Number of input channels to add.
        """
        if n <= 0:
            return

        device = self.weight.device
        dtype = self.weight.dtype

        new_cols = torch.zeros(
            self.out_channels, n, *self.kernel_size, device=device, dtype=dtype
        )
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_cols], dim=1))
        self.in_channels += n

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"effective={self.effective_channels()}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None}, "
            f"dominant_act={self.dominant_activation()}"
        )
