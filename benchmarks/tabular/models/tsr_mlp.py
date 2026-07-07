"""
TSRMLP: Topological Self-Regulating MLP for tabular data.

A stack of TSRLinear layers with per-neuron gating and activation mixing,
capable of dynamic width + depth growth during training via the TSR regulation
engine. Uses the same CatEmbedding input pipeline as the baseline MLP for a
clean matched comparison.
"""

from typing import List

import torch
import torch.nn as nn

from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_norm import TSRGroupNorm
from .common import CatEmbedding


class _TSRLinearBlock(nn.Module):
    """TSRLinear + TSRGroupNorm, wrapped as a single module for correct module-tree ordering."""

    def __init__(self, in_features: int, out_features: int, gate_init: float = 3.0,
                 act_init: str = "relu", norm_group_size: int = 8):
        super().__init__()
        self.linear = TSRLinear(in_features, out_features, gate_init=gate_init, act_init=act_init)
        self.norm = TSRGroupNorm(out_features, norm_group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(x))


class TSRMLP(nn.Module):
    """TSR-aware MLP for tabular data.

    Args:
        num_numeric: Number of numeric features.
        cat_cardinalities: Cardinality of each categorical feature.
        num_out: Output dimension (num_classes for classification, 1 for regression).
        seed_hidden: Initial hidden layer widths, e.g. [64, 64].
        cat_emb_dim: Embedding dimension for categorical features.
        gate_init: Initial gate logit for TSRLinear layers.
        act_init: Dominant activation at initialization.
        norm_group_size: Target channels per group in GroupNorm.
    """

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: List[int],
        num_out: int,
        seed_hidden: List[int] = [64, 64],
        cat_emb_dim: int = 8,
        gate_init: float = 3.0,
        act_init: str = "relu",
        norm_group_size: int = 8,
    ):
        super().__init__()
        self.cat_embedding = CatEmbedding(cat_cardinalities, cat_emb_dim)
        d_in = num_numeric + self.cat_embedding.output_dim

        self.blocks = nn.ModuleList()
        for hidden in seed_hidden:
            self.blocks.append(_TSRLinearBlock(d_in, hidden, gate_init, act_init, norm_group_size))
            d_in = hidden

        self.norm_group_size = norm_group_size
        self.head = TSRLinear(d_in, num_out, gate_init=gate_init, act_init=act_init)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_num, self.cat_embedding(x_cat)], dim=-1)

        for block in self.blocks:
            x = block(x)

        return self.head(x)

    def insert_block(self, after_index: int) -> None:
        """Insert a near-identity TSRLinear block after the given index.

        The new block is initialized with small-scale weights and gates born
        open (gate_init=5.0 → sigmoid≈0.99) so it doesn't disrupt the forward
        at birth.
        """
        if after_index < 0 or after_index >= len(self.blocks):
            return

        ref = self.blocks[after_index]
        d = ref.linear.out_features

        new_block = _TSRLinearBlock(d, d, gate_init=5.0, act_init="relu",
                                    norm_group_size=self.norm_group_size)
        with torch.no_grad():
            nn.init.eye_(new_block.linear.weight)
            if new_block.linear.bias is not None:
                new_block.linear.bias.zero_()

        # Move to the same device as the reference block
        device = next(ref.parameters()).device
        new_block = new_block.to(device)

        insert_idx = after_index + 1
        self.blocks.insert(insert_idx, new_block)

    def topology_state(self) -> dict:
        """Return per-layer widths and depth for static_final rebuild."""
        layers = []
        for i, block in enumerate(self.blocks):
            layers.append({
                "name": f"blocks.{i}.linear",
                "out_size": block.linear.out_features,
                "layer_type": "linear",
            })
        layers.append({
            "name": "head",
            "out_size": self.head.out_features,
            "layer_type": "linear",
        })
        return {
            "hidden_sizes": [b.linear.out_features for b in self.blocks],
            "depth": len(self.blocks),
            "layers": layers,
        }

    def topology_summary(self) -> str:
        widths = [b.linear.out_features for b in self.blocks]
        return f"MLP(depth={len(self.blocks)}, widths={widths})"
