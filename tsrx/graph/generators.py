"""Table 1 (paper/tsr-framework.tex sec:coupling): per-op classification.

Each node in the traced graph is classified into one of the generator
roles below. `groups.py` uses this classification to decide, per node,
whether to mint a fresh channel-identity ("tap"), union it with an
existing one, or pass an existing tap through unchanged.
"""

from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn


class Role(Enum):
    PRODUCER_CONSUMER = auto()  # row 1: Conv/Linear — new out-tap, consumes an in-tap
    GROUPED_CONV = auto()       # row 5: grouped/depthwise conv — divisibility (+ self-couple if depthwise)
    AFFINE_NORM = auto()        # rows 3-4: BatchNorm/InstanceNorm/GroupNorm — ties to input tap
    ELEMENTWISE_MERGE = auto()  # row 2: add/iadd — unions operand taps
    RESHAPE = auto()            # row 6: flatten/view — new tap with multiplicity
    CONCAT = auto()             # row 7: cat — new tap, partitioned with offsets
    PASSTHROUGH = auto()        # activation/pool/dropout/etc — tap unchanged
    UNKNOWN = auto()            # no tensor-shape opinion; tap unchanged if resolvable, else dropped


# Module types that produce a fresh, independently-sizable output channel
# axis and consume an existing one on their input (row 1).
_PRODUCER_CONSUMER_MODULES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.Linear,
    nn.ConvTranspose1d, nn.ConvTranspose2d,
)

_AFFINE_NORM_MODULES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d,
    nn.GroupNorm,
    nn.LayerNorm,
)

_PASSTHROUGH_MODULES = (
    nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh,
    nn.Hardtanh, nn.LeakyReLU, nn.ELU, nn.Dropout, nn.Dropout2d,
    nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.Identity,
)

_PASSTHROUGH_FUNCTIONS = {
    "relu", "gelu", "silu", "sigmoid", "tanh", "dropout", "max_pool2d",
    "avg_pool2d", "adaptive_avg_pool2d", "hardtanh", "leaky_relu", "elu",
    "getattr",
}

_MERGE_FUNCTIONS = {"add", "iadd", "__add__", "__iadd__", "mul", "__mul__"}
_RESHAPE_FUNCTIONS = {"flatten", "view", "reshape"}
_CONCAT_FUNCTIONS = {"cat", "concat"}


def classify_module(mod: nn.Module) -> Role:
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and getattr(mod, "groups", 1) > 1:
        return Role.GROUPED_CONV
    if isinstance(mod, _PRODUCER_CONSUMER_MODULES):
        return Role.PRODUCER_CONSUMER
    if isinstance(mod, _AFFINE_NORM_MODULES):
        return Role.AFFINE_NORM
    if isinstance(mod, _PASSTHROUGH_MODULES):
        return Role.PASSTHROUGH
    return Role.UNKNOWN


def classify_function(name: str) -> Role:
    if name in _MERGE_FUNCTIONS:
        return Role.ELEMENTWISE_MERGE
    if name in _RESHAPE_FUNCTIONS:
        return Role.RESHAPE
    if name in _CONCAT_FUNCTIONS:
        return Role.CONCAT
    if name in _PASSTHROUGH_FUNCTIONS:
        return Role.PASSTHROUGH
    return Role.UNKNOWN


def is_depthwise(mod: nn.Conv2d) -> bool:
    return mod.groups == mod.in_channels == mod.out_channels
