"""TSR-aware layer implementations with differentiable gating and activation mixing."""

from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_conv import TSRConv2d
from tsr.layers.tsr_norm import TSRGroupNorm

__all__ = ["TSRLinear", "TSRConv2d", "TSRGroupNorm"]
