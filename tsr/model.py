"""
TSRNetwork: The complete self-regulating neural network.

Assembles TSR layers into a VGG-style architecture (conv blocks → classifier)
that starts from a minimal seed and grows/prunes during training.

Architecture:
  - Stack of [TSRConv2d → TSRGroupNorm] blocks with MaxPool at intervals
  - Adaptive average pooling to fixed spatial size
  - TSRLinear classifier head

The model knows its own topology and can report it as a serializable state.
"""

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_conv import TSRConv2d
from tsr.layers.tsr_norm import TSRGroupNorm
from tsr.layers.gated_connection import GatedConnection


class TSRBlock(nn.Module):
    """A single TSR building block: Conv → Norm → (activation is inside Conv).

    The activation mixing is handled inside TSRConv2d, so the block is just
    Conv + Norm. Pooling is applied externally.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        gate_init: float = 3.0,
        act_init: str = "relu",
        norm_group_size: int = 8,
    ):
        super().__init__()
        self.conv = TSRConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            gate_init=gate_init,
            act_init=act_init,
        )
        self.norm = TSRGroupNorm(out_channels, target_group_size=norm_group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class TSRNetwork(nn.Module):
    """VGG-style TSR network that grows from a minimal seed.

    Args:
        in_channels: Number of input channels (3 for RGB images).
        seed_channels: List of initial channel counts for each conv block.
            Example: [8, 8] creates a 2-block network starting with 8 channels each.
        num_classes: Number of output classes.
        pool_positions: Indices of blocks after which to apply 2×2 MaxPool.
            Default: pool after every 2nd block.
        gate_init: Initial gate logit for all layers.
        act_init: Initial dominant activation for all layers.
        norm_group_size: Target group size for GroupNorm.
        classifier_hidden: Hidden layer size for the classifier head.
            If None, computed as 2× last conv channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        seed_channels: Optional[List[int]] = None,
        num_classes: int = 10,
        pool_positions: Optional[List[int]] = None,
        gate_init: float = 3.0,
        act_init: str = "relu",
        norm_group_size: int = 8,
        classifier_hidden: Optional[int] = None,
    ):
        super().__init__()

        if seed_channels is None:
            seed_channels = [8, 8]

        self.in_channels = in_channels
        self.num_classes = num_classes

        # ── Build conv blocks ──
        self.blocks = nn.ModuleList()
        prev_channels = in_channels

        for i, ch in enumerate(seed_channels):
            block = TSRBlock(
                in_channels=prev_channels,
                out_channels=ch,
                kernel_size=3,
                padding=1,
                gate_init=gate_init,
                act_init=act_init,
                norm_group_size=norm_group_size,
            )
            self.blocks.append(block)
            prev_channels = ch

        # Pool positions: default = after every 2nd block
        if pool_positions is None:
            pool_positions = [i for i in range(1, len(seed_channels), 2)]
            # Ensure at least one pool if there are ≥2 blocks
            if len(seed_channels) >= 2 and not pool_positions:
                pool_positions = [len(seed_channels) - 1]
        self.pool_positions = set(pool_positions)

        # ── Skip connections discovered by TSR ──
        # Keys are "src__dst" (block indices). ModuleDict so parameters are registered.
        self.skip_connections: nn.ModuleDict = nn.ModuleDict()

        # ── Adaptive pooling: Global Average Pool collapses spatial dims to 1×1 ──
        # GAP keeps the conv→linear bridge factor at 1 (no ×16 blowup as channels grow).
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier head ──
        flat_features = prev_channels  # GAP: each channel → 1 scalar
        if classifier_hidden is None:
            classifier_hidden = max(prev_channels * 2, 32)

        self.classifier = nn.Sequential(
            TSRLinear(flat_features, classifier_hidden, gate_init=gate_init, act_init=act_init),
            TSRGroupNorm(classifier_hidden, target_group_size=norm_group_size),
            TSRLinear(classifier_hidden, num_classes, gate_init=gate_init, act_init=act_init),
        )

    def add_skip_connection(
        self,
        src_idx: int,
        dst_idx: int,
        step: int = 0,
        gate_init: float = 0.0,
        projection_weight: Optional[torch.Tensor] = None,
    ) -> bool:
        """Materialize a skip connection from block src_idx to block dst_idx.

        Args:
            src_idx: Source block index.
            dst_idx: Destination block index (must be > src_idx).
            step: Current training step (for newborn protection).
            gate_init: Initial gate logit (0.0 → sigmoid=0.5, born-alive).
            projection_weight: Optional learned weight from the phantom probe.

        Returns:
            True if connection was added, False if it already exists or indices invalid.
        """
        key = f"{src_idx}__{dst_idx}"
        if key in self.skip_connections:
            return False
        if src_idx < 0 or dst_idx >= len(self.blocks) or src_idx >= dst_idx:
            return False

        src_ch = self.blocks[src_idx].conv.out_channels
        dst_ch = self.blocks[dst_idx].conv.out_channels
        conn = GatedConnection(src_ch, dst_ch, is_conv=True, gate_init=gate_init, step=step)

        if projection_weight is not None and not isinstance(conn.projection, nn.Identity):
            with torch.no_grad():
                if conn.projection.weight.shape == projection_weight.shape:
                    conn.projection.weight.copy_(projection_weight)

        device = next(self.parameters()).device
        conn = conn.to(device)
        self.skip_connections[key] = conn
        return True

    def prune_skip_connection(self, src_idx: int, dst_idx: int) -> bool:
        """Remove a skip connection. Returns True if it existed."""
        key = f"{src_idx}__{dst_idx}"
        if key not in self.skip_connections:
            return False
        del self.skip_connections[key]
        return True

    def insert_block(self, after_index: int) -> None:
        """Dynamically insert a new TSRBlock after the specified index.
        
        The new block is initialized as a near-identity mapping (Dirac initialization)
        to minimize disruption to the current network output.
        """
        if after_index < 0 or after_index >= len(self.blocks):
            return

        target_block = self.blocks[after_index]
        channels = target_block.conv.out_channels
        
        # Read the defaults used for the block
        gate_init = target_block.conv.gate.mean().item() if target_block.conv.gate.mean() > 0 else 5.0
        # Force gates to be fully open to preserve identity mapping
        gate_init = max(gate_init, 5.0) 
        
        act_init = target_block.conv.dominant_activation()
        norm_group_size = target_block.norm.target_group_size
        
        new_block = TSRBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            gate_init=gate_init,
            act_init=act_init,
            norm_group_size=norm_group_size,
        ).to(target_block.conv.weight.device)

        # 1. Initialize weights as Dirac delta (identity mapping for spatial dimensions)
        nn.init.dirac_(new_block.conv.weight.data)
        if new_block.conv.bias is not None:
            new_block.conv.bias.data.zero_()
            
        # 2. Insert into ModuleList
        self.blocks.insert(after_index + 1, new_block)
        
        # 3. Shift pool positions >= after_index + 1
        new_pool_positions = set()
        for p in self.pool_positions:
            if p >= after_index + 1:
                new_pool_positions.add(p + 1)
            else:
                new_pool_positions.add(p)
        self.pool_positions = new_pool_positions

        # 4. Re-index skip connections whose endpoints shifted. Without this,
        # any skip touching a block at or after the insertion point would
        # silently reference the wrong block once indices shift.
        shifted = nn.ModuleDict()
        for key, conn in self.skip_connections.items():
            src_idx, dst_idx = (int(v) for v in key.split("__"))
            if src_idx >= after_index + 1:
                src_idx += 1
            if dst_idx >= after_index + 1:
                dst_idx += 1
            shifted[f"{src_idx}__{dst_idx}"] = conn
        self.skip_connections = shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv blocks (with skip connections) → GAP → classifier.

        Args:
            x: Input images of shape (batch, channels, H, W).

        Returns:
            Logits of shape (batch, num_classes).
        """
        block_outputs: List[torch.Tensor] = []

        for i, block in enumerate(self.blocks):
            h = block(x)

            # Add contributions from any incoming skip connections
            for key, conn in self.skip_connections.items():
                src_idx, dst_idx = (int(v) for v in key.split("__"))
                if dst_idx == i:
                    h = h + conn(block_outputs[src_idx], dst_spatial=h.shape[-2:])

            block_outputs.append(h)

            x = F.max_pool2d(h, 2) if i in self.pool_positions else h

        x = self.adaptive_pool(x)
        x = x.flatten(1)  # (batch, channels) after GAP
        x = self.classifier(x)
        return x

    # ------------------------------------------------------------------
    # Topology introspection
    # ------------------------------------------------------------------

    def topology_summary(self) -> str:
        """Human-readable one-line topology summary."""
        channels = [f"{b.conv.effective_channels()}/{b.conv.out_channels}"
                    for b in self.blocks]
        classifier_info = []
        for module in self.classifier:
            if isinstance(module, TSRLinear):
                classifier_info.append(
                    f"{module.effective_neurons()}/{module.out_features}"
                )

        total = sum(p.numel() for p in self.parameters())
        skips = len(self.skip_connections)
        skip_str = f", skips={skips}" if skips else ""
        return (
            f"TSR[conv={'→'.join(channels)}, "
            f"fc={'→'.join(classifier_info)}"
            f"{skip_str}, "
            f"params={total:,}]"
        )
