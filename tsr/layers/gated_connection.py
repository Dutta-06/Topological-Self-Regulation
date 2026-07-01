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

    # ------------------------------------------------------------------
    # Grow-in-place: when an endpoint block's channel count changes, resize
    # the projection instead of deleting the connection. This preserves the
    # connection's learned state, gate value, and birth_step (so
    # newborn_protect_steps stays meaningful) across endpoint growth — the
    # same principle TSRConv2d.grow_channels/grow_input_channels already use
    # for neurons. Before this, a connection was deleted and rediscovered
    # from scratch every time either endpoint grew, which happens constantly
    # during the early bootstrap phase and defeated newborn protection.
    # ------------------------------------------------------------------

    def _rebuild_conv_projection(self, new_src: int, new_dst: int, weight: torch.Tensor) -> None:
        conv = nn.Conv2d(new_src, new_dst, kernel_size=1, bias=False)
        with torch.no_grad():
            conv.weight.copy_(weight)
        self.projection = conv.to(self.gate.device)
        self.src_channels = new_src
        self.dst_channels = new_dst

    def grow_dst_channels(self, n: int) -> None:
        """Destination block grew n new channels; extend the projection's output.

        New output rows are zero-init so the added capacity starts silent (no
        forward disruption) and must earn a nonzero projection via gradient,
        consistent with how new neurons start at small/zero-init weights.
        """
        if n <= 0 or not self.is_conv:
            return
        device = self.gate.device
        if isinstance(self.projection, nn.Identity):
            # Identity only holds while src==dst; growing dst breaks that, so
            # promote to a real 1x1 conv that reproduces the identity mapping
            # for existing channels and zero for the new ones.
            old_dim = self.dst_channels
            weight = torch.zeros(old_dim + n, self.src_channels, 1, 1, device=device)
            for i in range(min(old_dim, self.src_channels)):
                weight[i, i, 0, 0] = 1.0
            self._rebuild_conv_projection(self.src_channels, old_dim + n, weight)
        else:
            old_w = self.projection.weight.data
            new_rows = torch.zeros(n, old_w.shape[1], 1, 1, device=device)
            weight = torch.cat([old_w, new_rows], dim=0)
            self._rebuild_conv_projection(self.src_channels, old_w.shape[0] + n, weight)

    def grow_src_channels(self, n: int) -> None:
        """Source block grew n new channels; extend the projection's input.

        New input columns are zero-init: the new source channels contribute
        nothing until gradient pulls them open, matching grow_dst_channels.
        """
        if n <= 0 or not self.is_conv:
            return
        device = self.gate.device
        if isinstance(self.projection, nn.Identity):
            old_dim = self.src_channels
            weight = torch.zeros(self.dst_channels, old_dim + n, 1, 1, device=device)
            for i in range(min(old_dim, self.dst_channels)):
                weight[i, i, 0, 0] = 1.0
            self._rebuild_conv_projection(old_dim + n, self.dst_channels, weight)
        else:
            old_w = self.projection.weight.data
            new_cols = torch.zeros(old_w.shape[0], n, 1, 1, device=device)
            weight = torch.cat([old_w, new_cols], dim=1)
            self._rebuild_conv_projection(old_w.shape[1] + n, self.dst_channels, weight)

    def prune_dst_channels(self, indices_to_remove: torch.Tensor) -> None:
        """Destination block pruned these channel indices; shrink the projection's output."""
        if len(indices_to_remove) == 0 or not self.is_conv:
            return
        device = self.gate.device
        keep_mask = torch.ones(self.dst_channels, dtype=torch.bool)
        keep_mask[indices_to_remove] = False
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        new_dst = len(keep_indices)
        if isinstance(self.projection, nn.Identity):
            weight = torch.zeros(new_dst, self.src_channels, 1, 1, device=device)
            for new_i, old_i in enumerate(keep_indices.tolist()):
                if old_i < self.src_channels:
                    weight[new_i, old_i, 0, 0] = 1.0
            self._rebuild_conv_projection(self.src_channels, new_dst, weight)
        else:
            weight = self.projection.weight.data[keep_indices]
            self._rebuild_conv_projection(self.src_channels, new_dst, weight)

    def prune_src_channels(self, indices_to_remove: torch.Tensor) -> None:
        """Source block pruned these channel indices; shrink the projection's input."""
        if len(indices_to_remove) == 0 or not self.is_conv:
            return
        device = self.gate.device
        keep_mask = torch.ones(self.src_channels, dtype=torch.bool)
        keep_mask[indices_to_remove] = False
        keep_indices = keep_mask.nonzero(as_tuple=True)[0]
        new_src = len(keep_indices)
        if isinstance(self.projection, nn.Identity):
            weight = torch.zeros(self.dst_channels, new_src, 1, 1, device=device)
            for new_i, old_i in enumerate(keep_indices.tolist()):
                if old_i < self.dst_channels:
                    weight[old_i, new_i, 0, 0] = 1.0
            self._rebuild_conv_projection(new_src, self.dst_channels, weight)
        else:
            weight = self.projection.weight.data[:, keep_indices]
            self._rebuild_conv_projection(new_src, self.dst_channels, weight)

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
