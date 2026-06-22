"""
Phantom-sensor growth signal for TSR.

The heuristic bottleneck score (utilization × gradient_uniformity × saturation)
is a hand-designed *proxy* for "does this layer need more capacity?". Phantom
sensors replace that proxy with a *measured* quantity.

Idea
----
Attach to each growable TSR layer a small set of `k` **phantom neurons**:
extra candidate output units with their own weights and gate logits. Phantoms
are held *dormant* — they contribute exactly zero to the network's forward
output and zero to its FLOPs — yet they still receive a gradient on their gate.

That gate-gradient is the signal. ∂L/∂(phantom gate) measures, to first order,
how much the loss would change if that unit of capacity were switched on. A
strongly negative gate-gradient (turning the phantom on would *reduce* loss)
means the layer is capacity-starved exactly along the direction the phantom
has learned. When the windowed signal for a layer's best phantom crosses a
threshold, we **materialize** that phantom: promote its learned weights into a
real neuron (a far better initialization than random-small) and reset the
probe to keep sensing.

Hard-zero forward, live gradient
--------------------------------
Each phantom's contribution to the layer's auxiliary output is scaled by
``(g - g.detach())`` where ``g = sigmoid(gate)``. This term is algebraically
zero, so the network output is identical with or without phantoms — but its
gradient w.r.t. ``gate`` equals the phantom's true marginal contribution. The
auxiliary output is added to the model logits by the trainer so the phantom
gate sits on the real loss graph.

This module is self-contained: it touches neither TSRLinear/TSRConv2d's forward
path nor the existing monitor. Enable it from the trainer; disable it and TSR
behaves exactly as before.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsr.layers.tsr_linear import TSRLinear, ACTIVATION_FNS
from tsr.layers.tsr_conv import TSRConv2d


class PhantomProbe(nn.Module):
    """A bank of `k` dormant candidate neurons attached to one TSR layer.

    Shares the host layer's input. Holds its own weights, biases, and gate
    logits — initialized like real neurons but with gates dormant — plus a
    buffer accumulating the windowed gate-gradient signal.

    Args:
        host: The TSR layer this probe senses for (TSRLinear or TSRConv2d).
        k: Number of phantom candidates.
        window: Sliding-window length for averaging the gate-gradient signal.
    """

    def __init__(self, host: nn.Module, k: int = 4, window: int = 100):
        super().__init__()
        self.k = k
        self.window = window
        self.is_conv = isinstance(host, TSRConv2d)

        if self.is_conv:
            in_ch = host.in_channels
            kh, kw = host.kernel_size
            self.weight = nn.Parameter(torch.empty(k, in_ch, kh, kw))
            self.stride = host.stride
            self.padding = host.padding
        else:
            self.weight = nn.Parameter(torch.empty(k, host.in_features))

        self.bias = nn.Parameter(torch.zeros(k))
        # Phantom gates start near zero (dormant). Their *gradient* is the signal;
        # their value stays ~0 so phantoms never meaningfully contribute.
        self.gate = nn.Parameter(torch.full((k,), -2.0))

        # Share the host's activation mixture so the phantom senses through the
        # same nonlinearity the layer actually uses. Hold a reference to the
        # host module (read act_weights at forward time) rather than storing the
        # Parameter directly — assigning an nn.Parameter as an attribute would
        # register the HOST's params as the probe's, leaking them into any
        # parameters() walk over the phantom manager.
        object.__setattr__(self, "_host", host)

        # Windowed accumulator of per-phantom gate-gradient (filled by a hook).
        self._grad_window: List[torch.Tensor] = []

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Auxiliary phantom output, hard-zeroed but gradient-carrying.

        Returns a tensor shaped like a *pooled* phantom contribution that the
        trainer adds into the model's auxiliary loss term. The value is
        identically zero (so it never changes the live output); only its
        gradient w.r.t. ``self.gate`` is meaningful.

        Args:
            x: The host layer's input for this batch.

        Returns:
            Scalar tensor (sum of the hard-zeroed phantom activations).
        """
        if self.is_conv:
            h = F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        else:
            h = F.linear(x, self.weight, self.bias)

        # Same activation mixture as the host (read live so it tracks training).
        act_mix = F.softmax(self._host.act_weights, dim=0)
        a = torch.zeros_like(h)
        for i, act_fn in enumerate(ACTIVATION_FNS):
            a = a + act_mix[i] * act_fn(h)

        # Hard-zero coefficient with a live gradient path to the gate:
        #   (g - g.detach()) == 0  in value, but d/dg == 1.
        g = torch.sigmoid(self.gate)
        coeff = (g - g.detach())  # (k,)

        # Broadcast coeff over the phantom-channel/feature dim and sum to a scalar.
        if self.is_conv:
            # a: (B, k, H, W) → weight each phantom channel by its coeff
            contrib = (a * coeff.view(1, -1, 1, 1)).sum()
        else:
            contrib = (a * coeff.view(1, -1)).sum()
        return contrib

    def record_gate_grad(self) -> None:
        """Snapshot the current phantom gate gradient into the window.

        Call after ``loss.backward()``. The accumulated mean is the measured
        growth signal per phantom.
        """
        if self.gate.grad is None:
            return
        self._grad_window.append(self.gate.grad.detach().abs().cpu())
        if len(self._grad_window) > self.window:
            self._grad_window.pop(0)

    def signal(self) -> Optional[torch.Tensor]:
        """Windowed mean |gate-gradient| per phantom, or None if no data yet."""
        if not self._grad_window:
            return None
        return torch.stack(self._grad_window).mean(dim=0)

    def best_signal(self) -> float:
        """Scalar growth signal for the host layer = strongest phantom."""
        sig = self.signal()
        return float(sig.max().item()) if sig is not None else 0.0

    def best_phantom_index(self) -> Optional[int]:
        """Index of the phantom with the strongest signal (the one to materialize)."""
        sig = self.signal()
        return int(sig.argmax().item()) if sig is not None else None

    def clear(self) -> None:
        self._grad_window.clear()


class PhantomManager(nn.Module):
    """Owns one PhantomProbe per growable TSR layer and drives the sensor loop.

    Responsibilities:
      1. Attach a probe to every TSRLinear/TSRConv2d in the model.
      2. Via forward pre-hooks, capture each layer's input for the current
         batch so probes see exactly what their host layer sees.
      3. Provide ``aux_loss(model_out)`` — the summed hard-zero phantom
         contribution — which the trainer adds to its loss so phantom gates
         land on the real backward graph.
      4. After backward, record per-phantom gate gradients.
      5. Expose ``growth_signals()`` (measured signal per layer) and
         ``materialize_weights(layer_name)`` (the winning phantom's learned
         weights, for initializing a newly grown neuron).

    It is an nn.Module so its probe parameters are registered, optimized, and
    checkpointed alongside the model.

    Args:
        model: The TSR model to attach probes to.
        k: Phantom candidates per layer.
        window: Sliding-window length for the signal.
    """

    def __init__(self, model: nn.Module, k: int = 4, window: int = 100):
        super().__init__()
        # Hold the model WITHOUT registering it as a submodule, else
        # self.parameters() would include the entire model and the optimizer
        # would double-count its params.
        object.__setattr__(self, "model", model)
        self.k = k
        self.window = window

        self.probes = nn.ModuleDict()
        self._captured_input: Dict[str, torch.Tensor] = {}
        self._hooks = []
        self._name_map: Dict[str, str] = {}  # sanitized key -> real layer name

        self._attach()

    @staticmethod
    def _key(layer_name: str) -> str:
        # ModuleDict keys can't contain '.'
        return layer_name.replace(".", "__")

    def _attach(self) -> None:
        for name, module in self.model.named_modules():
            if isinstance(module, (TSRLinear, TSRConv2d)):
                key = self._key(name)
                self.probes[key] = PhantomProbe(module, k=self.k, window=self.window)
                self._name_map[key] = name
                h = module.register_forward_pre_hook(self._make_capture_hook(key))
                self._hooks.append(h)

    def _make_capture_hook(self, key: str):
        def hook(module, inputs):
            # inputs is a tuple; the layer input is inputs[0].
            self._captured_input[key] = inputs[0]
        return hook

    def aux_loss(self) -> torch.Tensor:
        """Summed hard-zero phantom contribution for the current batch.

        Add this to the training loss. It is ~0 in value (so it does not change
        optimization of the real network) but routes gradient to every phantom
        gate. Must be called after the model's forward pass (so inputs are
        captured) and before backward.
        """
        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)
        for key, probe in self.probes.items():
            x = self._captured_input.get(key)
            if x is not None:
                total = total + probe(x)
        return total

    def record_gradients(self) -> None:
        """Snapshot phantom gate gradients into each probe's window (post-backward)."""
        for probe in self.probes.values():
            probe.record_gate_grad()

    def growth_signals(self) -> Dict[str, float]:
        """Measured per-layer growth signal (strongest phantom), keyed by real layer name."""
        return {
            self._name_map[key]: probe.best_signal()
            for key, probe in self.probes.items()
        }

    def materialize_weights(self, layer_name: str):
        """Return the winning phantom's (weight_row, bias) for a layer, or None.

        These initialize a newly grown neuron from learned features instead of
        random-small noise. The phantom's gate is left dormant in the host
        (the materialized neuron gets the host's standard 'asleep' init), but
        its *weights* carry the direction the sensor found useful.
        """
        key = self._key(layer_name)
        if key not in self.probes:
            return None
        probe = self.probes[key]
        idx = probe.best_phantom_index()
        if idx is None:
            return None
        with torch.no_grad():
            w = probe.weight.data[idx].clone()
            b = probe.bias.data[idx].clone()
        return w, b

    def reset_layer(self, layer_name: str) -> None:
        """Clear and re-randomize a layer's probe after it grows.

        After a phantom is materialized, re-initialize the probe so it keeps
        sensing for the *next* unit of capacity rather than re-reporting the
        one we just consumed.
        """
        key = self._key(layer_name)
        if key in self.probes:
            probe = self.probes[key]
            probe.clear()
            probe._reset_parameters()
            with torch.no_grad():
                probe.gate.fill_(-2.0)

    def refresh(self) -> None:
        """Rebuild probes against the current module tree (after depth changes)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        old = dict(self.probes)
        self.probes = nn.ModuleDict()
        self._name_map = {}
        self._captured_input.clear()
        for name, module in self.model.named_modules():
            if isinstance(module, (TSRLinear, TSRConv2d)):
                key = self._key(name)
                # Reuse an existing probe only if its input shape still matches
                # the host layer's; otherwise build a fresh one.
                probe = old.get(key)
                if probe is None or probe.weight.shape[1:] != module.weight.shape[1:]:
                    probe = PhantomProbe(module, k=self.k, window=self.window)
                    device = next(self.model.parameters()).device
                    probe = probe.to(device)
                self.probes[key] = probe
                self._name_map[key] = name
                h = module.register_forward_pre_hook(self._make_capture_hook(key))
                self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
