"""
TSRLinear: Linear layer with differentiable neuron gating and activation mixing.

Each output neuron has:
  - A learnable gate (sigmoid) that controls its contribution
  - A shared learnable activation mixture (softmax over ReLU, Tanh, GELU, SiLU)

The gate enables soft neuron importance estimation during the fast (gradient)
timescale. The slow (structural) timescale uses gate values to identify dead
neurons for pruning and gradient signals for growth decisions.
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Activation function registry — order matters, must match act_weights indexing
ACTIVATION_FNS = [F.relu, torch.tanh, F.gelu, F.silu]
ACTIVATION_NAMES = ["relu", "tanh", "gelu", "silu"]
NUM_ACTIVATIONS = len(ACTIVATION_FNS)


class TSRLinear(nn.Module):
    """Linear layer with differentiable neuron gates and learnable activation mixing.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If True, adds a learnable bias. Default: True.
        gate_init: Initial gate logit value. Higher = more open.
            Default 3.0 → sigmoid(3.0) ≈ 0.95 (nearly fully open).
        act_init: Which activation to favor initially.
            Default 'relu' → ReLU-dominant mixture.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gate_init: float = 3.0,
        act_init: str = "relu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Core linear parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Per-neuron gate logits: sigmoid(gate) ∈ (0, 1) controls neuron contribution
        self.gate = nn.Parameter(torch.full((out_features,), gate_init))

        # Birth step per neuron: -1 = original (never protected), >0 = grown at that step
        self.register_buffer('neuron_birth_step', torch.full((out_features,), -1, dtype=torch.long))

        # Activation mixture logits: softmax(act_weights) gives mixture coefficients
        # over [ReLU, Tanh, GELU, SiLU]
        act_idx = ACTIVATION_NAMES.index(act_init) if act_init in ACTIVATION_NAMES else 0
        act_logits = torch.zeros(NUM_ACTIVATIONS)
        act_logits[act_idx] = 3.0  # softmax([3,0,0,0]) ≈ [0.84, 0.05, 0.05, 0.05]
        self.act_weights = nn.Parameter(act_logits)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Kaiming uniform initialization (same as nn.Linear default)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gating and activation mixing.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        # Linear transformation
        h = F.linear(x, self.weight, self.bias)  # (batch, out_features)

        # Apply soft neuron gate
        gate_values = torch.sigmoid(self.gate)  # (out_features,)
        h = h * gate_values

        # Activation mixing: weighted sum of multiple activations
        act_mix = F.softmax(self.act_weights, dim=0)  # (NUM_ACTIVATIONS,)
        out = torch.zeros_like(h)
        for i, act_fn in enumerate(ACTIVATION_FNS):
            out = out + act_mix[i] * act_fn(h)

        return out

    # ------------------------------------------------------------------
    # Introspection methods for the regulation engine
    # ------------------------------------------------------------------

    def gate_values(self) -> torch.Tensor:
        """Return current gate activations (0 = dead, 1 = fully active)."""
        with torch.no_grad():
            return torch.sigmoid(self.gate)

    def effective_neurons(self) -> int:
        """Count neurons with gate activation above threshold (0.5)."""
        return int((self.gate_values() > 0.5).sum().item())

    def activation_distribution(self) -> dict:
        """Return the current activation mixture as a named dict."""
        with torch.no_grad():
            mix = F.softmax(self.act_weights, dim=0)
            return {name: mix[i].item() for i, name in enumerate(ACTIVATION_NAMES)}

    def dominant_activation(self) -> str:
        """Return the name of the activation with highest mixture weight."""
        with torch.no_grad():
            idx = self.act_weights.argmax().item()
            return ACTIVATION_NAMES[idx]

    # ------------------------------------------------------------------
    # Structural modification methods (called by the regulation engine)
    # ------------------------------------------------------------------

    def prune_neurons(self, indices_to_remove: torch.Tensor) -> None:
        """Remove neurons at the given indices.

        NOTE: The caller is responsible for also updating the downstream
        layer's in_features/weight columns. This method only modifies
        this layer's output dimension.

        Args:
            indices_to_remove: 1D tensor of neuron indices to remove.
        """
        if len(indices_to_remove) == 0:
            return

        keep_mask = torch.ones(self.out_features, dtype=torch.bool)
        keep_mask[indices_to_remove] = False
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]

        self.weight = nn.Parameter(self.weight.data[keep_indices])
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.data[keep_indices])
        self.gate = nn.Parameter(self.gate.data[keep_indices])
        self.neuron_birth_step = self.neuron_birth_step[keep_indices]
        self.out_features = len(keep_indices)

    def prune_input_channels(self, indices_to_remove: torch.Tensor) -> None:
        """Remove input channels (called when upstream layer prunes neurons).

        Args:
            indices_to_remove: 1D tensor of input channel indices to remove.
        """
        if len(indices_to_remove) == 0:
            return

        keep_mask = torch.ones(self.in_features, dtype=torch.bool)
        keep_mask[indices_to_remove] = False
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]

        self.weight = nn.Parameter(self.weight.data[:, keep_indices])
        self.in_features = len(keep_indices)

    def grow_neurons(self, n: int, init_scale: float = 0.001, newborn_gate: float = 0.0, step: int = 0) -> None:
        """Add n new neurons, born alive at newborn_gate logit.

        New neurons start with phantom-initialized (or small random) weights and a
        gate logit of newborn_gate (default 0.0 → sigmoid≈0.5, alive and gradient-accessible).
        The caller should seed weights via _seed_grown_from_phantom after calling this.

        NOTE: The caller is responsible for also updating the downstream
        layer's in_features/weight columns.

        Args:
            n: Number of neurons to add.
            init_scale: Scale factor for weight initialization (overwritten by phantom seed).
            newborn_gate: Gate logit for new neurons. 0.0 → sigmoid=0.5 (alive, not open).
            step: Current training step, recorded for newborn protection.
        """
        if n <= 0:
            return

        device = self.weight.device
        dtype = self.weight.dtype

        new_w = torch.randn(n, self.in_features, device=device, dtype=dtype) * init_scale
        new_g = torch.full((n,), newborn_gate, device=device, dtype=dtype)

        self.weight = nn.Parameter(torch.cat([self.weight.data, new_w], dim=0))
        self.gate = nn.Parameter(torch.cat([self.gate.data, new_g], dim=0))

        if self.bias is not None:
            new_b = torch.zeros(n, device=device, dtype=dtype)
            self.bias = nn.Parameter(torch.cat([self.bias.data, new_b], dim=0))

        new_birth = torch.full((n,), step, dtype=torch.long, device=device)
        self.neuron_birth_step = torch.cat([self.neuron_birth_step, new_birth])

        self.out_features += n

    def grow_input_channels(self, n: int) -> None:
        """Add input channels (called when upstream layer grows neurons).

        Args:
            n: Number of input channels to add.
        """
        if n <= 0:
            return

        device = self.weight.device
        dtype = self.weight.dtype

        # Zero-init for new input connections — no disruption to existing outputs
        new_cols = torch.zeros(self.out_features, n, device=device, dtype=dtype)
        self.weight = nn.Parameter(torch.cat([self.weight.data, new_cols], dim=1))
        self.in_features += n

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"effective={self.effective_neurons()}, "
            f"bias={self.bias is not None}, "
            f"dominant_act={self.dominant_activation()}"
        )
