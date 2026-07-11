"""
TSRTCN: Topological Self-Regulating Temporal Convolutional Network.

A TCN-style stack of causal dilated TSRConv1d blocks, mirroring the baseline
TCN structure but with per-channel gating and activation mixing. Supports
dynamic width (channel growth/pruning) and depth (block insertion) via the
TSR regulation engine.

No residual/skip-add inside a block (unlike the standard TCN reference
architecture): the generic paired-layer regulation engine (tsr/regulation/
actions.py) only resizes the *sequential* next TSR layer when a layer grows
or prunes — it has no concept of a branching residual path, so a per-block
skip connection would go out of sync with conv2's channel count the moment
either is grown/pruned (the branch and the main path would then have
mismatched channels at the point they're added). Dropping the residual add
keeps every block a plain sequential Conv1d->Conv1d chain — the same
Conv-then-Norm shape as TSRNetwork's own TSRBlock (also skip-free) — so
width growth on any conv layer is handled correctly by the existing
sequential paired-layer logic with no special-casing.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsr.layers.tsr_conv1d import TSRConv1d
from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_norm import TSRGroupNorm


class _Chomp1d(nn.Module):
    """Crops the end of a padded sequence to enforce causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class _TSRTemporalBlock(nn.Module):
    """A single causal dilated conv block with TSR gating (Conv->Norm->Conv->Norm,
    no residual add — see module docstring for why).

    Args:
        in_channels: Input channel count.
        conv1_out_channels: Output channels of the first conv.
        conv2_out_channels: Output channels of the second conv. Defaults to
            conv1_out_channels for uniform-width blocks.
    """

    def __init__(self, in_channels: int, conv1_out_channels: int, kernel_size: int,
                 dilation: int, conv2_out_channels: Optional[int] = None,
                 gate_init: float = 3.0, act_init: str = "relu",
                 norm_group_size: int = 8):
        super().__init__()
        if conv2_out_channels is None:
            conv2_out_channels = conv1_out_channels
        self.norm_group_size = norm_group_size
        padding = (kernel_size - 1) * dilation

        self.conv1 = TSRConv1d(in_channels, conv1_out_channels, kernel_size,
                               padding=padding, dilation=dilation,
                               gate_init=gate_init, act_init=act_init)
        self.chomp1 = _Chomp1d(padding)
        self.norm1 = TSRGroupNorm(conv1_out_channels, norm_group_size)
        self.drop1 = nn.Dropout(0.1)

        self.conv2 = TSRConv1d(conv1_out_channels, conv2_out_channels, kernel_size,
                               padding=padding, dilation=dilation,
                               gate_init=gate_init, act_init=act_init)
        self.chomp2 = _Chomp1d(padding)
        self.norm2 = TSRGroupNorm(conv2_out_channels, norm_group_size)
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(F.relu(self.norm1(self.chomp1(self.conv1(x)))))
        out = self.drop2(F.relu(self.norm2(self.chomp2(self.conv2(out)))))
        return out


class TSRTCNEncoder(nn.Module):
    """Stack of causal dilated TSRConv1d blocks with exponential dilation.

    Args:
        input_size: Number of input channels.
        seed_channels: Uniform channel width for all blocks (used when
            block_widths is not provided).
        num_levels: Number of blocks (used when block_widths is not provided).
        block_widths: Per-block output channel widths. When provided,
            overrides seed_channels/num_levels and each block i transitions
            from block_widths[i-1] (or input_size for block 0) to block_widths[i].
    """

    def __init__(
        self,
        input_size: int,
        seed_channels: int = 32,
        num_levels: int = 2,
        kernel_size: int = 3,
        gate_init: float = 3.0,
        act_init: str = "relu",
        norm_group_size: int = 8,
        block_widths: Optional[List[int]] = None,
        block_conv_widths: Optional[List[tuple]] = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.gate_init = gate_init
        self.act_init = act_init
        self.norm_group_size = norm_group_size
        self.blocks = nn.ModuleList()

        if block_conv_widths is not None:
            # Per-block (conv1_out, conv2_out) pairs — for exact static_final rebuild
            in_ch = input_size
            for level, (c1_out, c2_out) in enumerate(block_conv_widths):
                dilation = 2 ** level
                self.blocks.append(_TSRTemporalBlock(
                    in_ch, c1_out, kernel_size, dilation,
                    conv2_out_channels=c2_out,
                    gate_init=gate_init, act_init=act_init, norm_group_size=norm_group_size,
                ))
                in_ch = c2_out
        elif block_widths is not None:
            in_ch = input_size
            for level, width in enumerate(block_widths):
                dilation = 2 ** level
                self.blocks.append(_TSRTemporalBlock(
                    in_ch, width, kernel_size, dilation,
                    gate_init=gate_init, act_init=act_init, norm_group_size=norm_group_size,
                ))
                in_ch = width
        else:
            in_ch = input_size
            for level in range(num_levels):
                dilation = 2 ** level
                self.blocks.append(_TSRTemporalBlock(
                    in_ch, seed_channels, kernel_size, dilation,
                    gate_init=gate_init, act_init=act_init, norm_group_size=norm_group_size,
                ))
                in_ch = seed_channels

    @property
    def hidden_channels(self) -> int:
        """Live output width of the last block (not a stale cached value)."""
        if not self.blocks:
            return 0
        return self.blocks[-1].conv2.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> last-step representation (B, hidden_channels)."""
        h = x.transpose(1, 2)  # (B, C, L)
        for block in self.blocks:
            h = block(h)
        return h[:, :, -1]  # (B, hidden_channels)

    def insert_block(self, after_index: int) -> None:
        """Insert a near-identity TSRConv1d block after the given index.

        Uses Dirac initialization so the network's function is unchanged at
        the moment of insertion.
        """
        if after_index < 0 or after_index >= len(self.blocks):
            return

        ref_block = self.blocks[after_index]
        # Read the live output width from the reference block (not a stale cache)
        ch = ref_block.conv2.out_channels
        kernel_size = ref_block.conv1.kernel_size

        new_block = _TSRTemporalBlock(
            ch, ch, kernel_size, dilation=1,
            gate_init=5.0, act_init="relu",
            norm_group_size=self.norm_group_size,
        )

        # Dirac init: conv weights → identity-like, bias → 0
        with torch.no_grad():
            nn.init.dirac_(new_block.conv1.weight)
            nn.init.dirac_(new_block.conv2.weight)
            if new_block.conv1.bias is not None:
                new_block.conv1.bias.zero_()
            if new_block.conv2.bias is not None:
                new_block.conv2.bias.zero_()

        # Move to the same device as the reference block
        device = next(ref_block.parameters()).device
        new_block = new_block.to(device)

        insert_idx = after_index + 1
        self.blocks.insert(insert_idx, new_block)


class TSRTCNForecaster(nn.Module):
    """TSR-TCN for time-series forecasting.

    channel_independent (default True): PatchTST-style — one shared encoder is
    applied per channel by folding channels into the batch dimension
    ((B, L, C) -> (B*C, L, 1)), and the head predicts a single channel's
    horizon (TSRLinear(hidden, pred_len)). Head params are then INDEPENDENT of
    channel count: at electricity's 321 channels this is ~pred_len*hidden vs
    the joint head's pred_len*channels*hidden (~15M -> ~46K at h720). The
    channel-independence also matches how PatchTST wins the params-vs-accuracy
    tradeoff on multichannel forecasting.

    channel_independent=False: the original joint head
    (TSRLinear(hidden, pred_len*input_size)) predicting all channels at once.
    """

    def __init__(
        self,
        input_size: int,
        pred_len: int,
        seed_channels: int = 32,
        num_levels: int = 2,
        kernel_size: int = 3,
        gate_init: float = 3.0,
        act_init: str = "relu",
        norm_group_size: int = 8,
        channel_independent: bool = True,
        block_widths: Optional[List[int]] = None,
        block_conv_widths: Optional[List[tuple]] = None,
    ):
        super().__init__()
        self.channel_independent = channel_independent
        self.pred_len = pred_len
        self.input_size = input_size

        # Channel-independent: the encoder sees single-channel series.
        encoder_input = 1 if channel_independent else input_size
        self.encoder = TSRTCNEncoder(
            encoder_input, seed_channels, num_levels, kernel_size,
            gate_init=gate_init, act_init=act_init, norm_group_size=norm_group_size,
            block_widths=block_widths,
            block_conv_widths=block_conv_widths,
        )
        head_in = self.encoder.hidden_channels
        head_out = pred_len if channel_independent else pred_len * input_size
        self.head = TSRLinear(head_in, head_out,
                              gate_init=gate_init, act_init=act_init, head=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, pred_len, C)."""
        if self.channel_independent:
            b, l, c = x.shape
            # Fold channels into the batch: each channel is its own single-var
            # series through the shared encoder.
            x = x.permute(0, 2, 1).reshape(b * c, l, 1)  # (B*C, L, 1)
            out = self.head(self.encoder(x))              # (B*C, pred_len)
            out = out.reshape(b, c, self.pred_len)        # (B, C, pred_len)
            return out.permute(0, 2, 1)                   # (B, pred_len, C)
        out = self.head(self.encoder(x))
        return out.view(-1, self.pred_len, self.input_size)

    def insert_block(self, after_index: int) -> None:
        self.encoder.insert_block(after_index)

    def topology_state(self) -> dict:
        blocks = self.encoder.blocks
        return {
            "seed_channels": self.encoder.hidden_channels,
            "num_blocks": len(blocks),
            "channel_independent": self.channel_independent,
            "block_widths": [b.conv2.out_channels for b in blocks],
            "block_conv_widths": [(b.conv1.out_channels, b.conv2.out_channels) for b in blocks],
            "layers": [
                {"name": f"encoder.blocks.{i}.conv2", "out_size": b.conv2.out_channels, "layer_type": "conv1d"}
                for i, b in enumerate(blocks)
            ],
        }

    def topology_summary(self) -> str:
        widths = [b.conv2.out_channels for b in self.encoder.blocks]
        ci = "CI" if self.channel_independent else "joint"
        return f"TCN({ci}, blocks={len(self.encoder.blocks)}, widths={widths})"


class TSRTCNClassifier(nn.Module):
    """TSR-TCN for time-series classification.

    Always joint (whole-window, all-channel) — channel-independence is a
    forecasting-specific trick (per-channel horizon head); a classification
    label describes the whole multivariate window, not one channel.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        seed_channels: int = 32,
        num_levels: int = 2,
        kernel_size: int = 3,
        gate_init: float = 3.0,
        act_init: str = "relu",
        norm_group_size: int = 8,
        block_widths: Optional[List[int]] = None,
        block_conv_widths: Optional[List[tuple]] = None,
    ):
        super().__init__()
        self.encoder = TSRTCNEncoder(
            input_size, seed_channels, num_levels, kernel_size,
            gate_init=gate_init, act_init=act_init, norm_group_size=norm_group_size,
            block_widths=block_widths,
            block_conv_widths=block_conv_widths,
        )
        head_in = self.encoder.hidden_channels
        self.head = TSRLinear(head_in, num_classes,
                              gate_init=gate_init, act_init=act_init, head=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> (B, num_classes)."""
        return self.head(self.encoder(x))

    def insert_block(self, after_index: int) -> None:
        self.encoder.insert_block(after_index)

    def topology_state(self) -> dict:
        blocks = self.encoder.blocks
        return {
            "seed_channels": self.encoder.hidden_channels,
            "num_blocks": len(blocks),
            "block_widths": [b.conv2.out_channels for b in blocks],
            "block_conv_widths": [(b.conv1.out_channels, b.conv2.out_channels) for b in blocks],
            "layers": [
                {"name": f"encoder.blocks.{i}.conv2", "out_size": b.conv2.out_channels, "layer_type": "conv1d"}
                for i, b in enumerate(blocks)
            ],
        }

    def topology_summary(self) -> str:
        widths = [b.conv2.out_channels for b in self.encoder.blocks]
        return f"TCN(blocks={len(self.encoder.blocks)}, widths={widths})"
