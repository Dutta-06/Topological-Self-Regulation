"""
GatedConnection: a gated skip/residual edge between two blocks.

Mirrors the gated-neuron design applied to connections (edges) rather than nodes.
TSR can grow and prune GatedConnections using the same machinery:
  - Born-alive gate (logit 0.0 → sigmoid 0.5, above death threshold)
  - Newborn protection (birth_step buffer)
  - Sparsity pressure (gate_sparsity_penalty operates on all gate parameters)
  - Phantom sensor (ConnectionPhantomProbe in phantom.py)

For vision (conv): projects src channels → dst channels via 1×1 conv; handles
spatial mismatch via AdaptiveAvgPool to match destination spatial size.
For MLP (linear): projects src features → dst features via Linear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedConnection(nn.Module):
    """Gated skip edge: dst += sigmoid(gate) * project(src).

    Args:
        src_channels: Output channels/features of the source block.
        dst_channels: Output channels/features of the destination block.
        is_conv: True for conv→conv skip (uses 1×1 conv projection);
                 False for linear→linear skip (uses Linear projection).
        gate_init: Initial gate logit. 0.0 → sigmoid=0.5 (born-alive).
        step: Training step at birth (for newborn protection).
    """

    def __init__(
        self,
        src_channels: int,
        dst_channels: int,
        is_conv: bool = True,
        gate_init: float = 0.0,
        step: int = 0,
    ):
        super().__init__()
        self.src_channels = src_channels
        self.dst_channels = dst_channels
        self.is_conv = is_conv

        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        self.register_buffer("birth_step", torch.tensor(step, dtype=torch.long))

        if is_conv:
            if src_channels != dst_channels:
                self.projection: nn.Module = nn.Conv2d(
                    src_channels, dst_channels, kernel_size=1, bias=False
                )
                nn.init.kaiming_uniform_(self.projection.weight, a=5 ** 0.5)
            else:
                self.projection = nn.Identity()
        else:
            if src_channels != dst_channels:
                self.projection = nn.Linear(src_channels, dst_channels, bias=False)
                nn.init.kaiming_uniform_(self.projection.weight, a=5 ** 0.5)
            else:
                self.projection = nn.Identity()

    def gate_value(self) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.gate).item()

    def forward(
        self,
        src: torch.Tensor,
        dst_spatial: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Return sigmoid(gate) * project(src), optionally resized to dst_spatial.

        Args:
            src: Source feature map (B, C_src, H, W) or vector (B, F_src).
            dst_spatial: (H', W') to resize to when spatial dims differ. Conv only.

        Returns:
            Tensor with same shape as the destination block output.
        """
        h = self.projection(src)
        if self.is_conv and dst_spatial is not None:
            h_dst, w_dst = dst_spatial
            if h.shape[-2] != h_dst or h.shape[-1] != w_dst:
                h = F.adaptive_avg_pool2d(h, (h_dst, w_dst))
        return torch.sigmoid(self.gate) * h
