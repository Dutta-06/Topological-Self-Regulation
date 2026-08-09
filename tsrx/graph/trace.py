"""fx tracing + shape propagation for the coupling engine.

We need two things from a model before we can discover coupling groups
(Definition 5.1): (1) the computation graph in topological order, and (2)
each node's output shape, so we can distinguish which axis is the channel
axis (dim=1 for NCHW/NC.../NC-L conventions used throughout this repo).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp


@dataclass
class TracedModel:
    gm: fx.GraphModule
    example_inputs: tuple


def trace_model(model: torch.nn.Module, example_inputs) -> TracedModel:
    """Symbolically trace `model` and annotate every node with its output
    shape via ShapeProp.

    Args:
        model: The module to trace.
        example_inputs: A tuple of example input tensors (batch dim can be
            small, e.g. 2, to keep tracing/propagation cheap).

    Returns:
        TracedModel wrapping the GraphModule (nodes carry
        `node.meta["tensor_meta"].shape` after this call).
    """
    gm = fx.symbolic_trace(model)
    ShapeProp(gm).propagate(*example_inputs)
    return TracedModel(gm=gm, example_inputs=example_inputs)


def node_shape(node: fx.Node) -> Optional[torch.Size]:
    meta = node.meta.get("tensor_meta")
    if meta is None:
        return None
    return meta.shape
