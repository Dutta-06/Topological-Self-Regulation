"""
Topology state serialization, comparison, and edit distance.

Provides:
  - TopologyState: a frozen, serializable snapshot of network structure
  - topology_edit_distance: measures structural change between two states
  - Save/load to JSON for reproducibility and analysis
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class LayerTopology:
    """Topology of a single layer."""
    name: str
    layer_type: str  # "conv" or "linear"
    in_size: int     # in_channels or in_features
    out_size: int    # out_channels or out_features
    effective_size: int  # effective (gated) neurons/channels
    dominant_activation: str
    activation_distribution: Dict[str, float] = field(default_factory=dict)
    gate_mean: float = 1.0
    gate_min: float = 1.0
    gate_max: float = 1.0


@dataclass
class SkipConnectionTopology:
    """Topology of a single discovered skip connection."""
    src: int
    dst: int
    gate_value: float
    src_channels: int
    dst_channels: int
    birth_step: int


@dataclass
class TopologyState:
    """Complete frozen snapshot of network topology at a point in training."""
    step: int
    layers: List[LayerTopology] = field(default_factory=list)
    skip_connections: List[SkipConnectionTopology] = field(default_factory=list)
    pool_positions: List[int] = field(default_factory=list)
    total_params: int = 0
    effective_params: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyState":
        layers = [LayerTopology(**l) for l in d.get("layers", [])]
        skips = [SkipConnectionTopology(**s) for s in d.get("skip_connections", [])]
        return cls(
            step=d["step"],
            layers=layers,
            skip_connections=skips,
            pool_positions=d.get("pool_positions", []),
            total_params=d.get("total_params", 0),
            effective_params=d.get("effective_params", 0),
        )

    @classmethod
    def from_json(cls, s: str) -> "TopologyState":
        return cls.from_dict(json.loads(s))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "TopologyState":
        with open(path, "r") as f:
            return cls.from_json(f.read())


def capture_topology(model, step: int) -> TopologyState:
    """Capture the current topology of a TSR model.

    Args:
        model: A TSRNetwork instance.
        step: Current training step.

    Returns:
        Frozen TopologyState snapshot.
    """
    from tsr.layers.tsr_linear import TSRLinear
    from tsr.layers.tsr_conv import TSRConv2d
    from tsr.utils import count_parameters, count_effective_parameters

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, TSRLinear):
            layers.append(LayerTopology(
                name=name,
                layer_type="linear",
                in_size=module.in_features,
                out_size=module.out_features,
                effective_size=module.effective_neurons(),
                dominant_activation=module.dominant_activation(),
                activation_distribution=module.activation_distribution(),
                gate_mean=module.gate_values().mean().item(),
                gate_min=module.gate_values().min().item(),
                gate_max=module.gate_values().max().item(),
            ))
        elif isinstance(module, TSRConv2d):
            layers.append(LayerTopology(
                name=name,
                layer_type="conv",
                in_size=module.in_channels,
                out_size=module.out_channels,
                effective_size=module.effective_channels(),
                dominant_activation=module.dominant_activation(),
                activation_distribution=module.activation_distribution(),
                gate_mean=module.gate_values().mean().item(),
                gate_min=module.gate_values().min().item(),
                gate_max=module.gate_values().max().item(),
            ))

    skip_connections = []
    for key, conn in getattr(model, "skip_connections", {}).items():
        src_idx, dst_idx = (int(v) for v in key.split("__"))
        skip_connections.append(SkipConnectionTopology(
            src=src_idx,
            dst=dst_idx,
            gate_value=conn.gate_value(),
            src_channels=conn.src_channels,
            dst_channels=conn.dst_channels,
            birth_step=int(conn.birth_step.item()),
        ))

    pool_positions = sorted(getattr(model, "pool_positions", []))

    return TopologyState(
        step=step,
        layers=layers,
        skip_connections=skip_connections,
        pool_positions=pool_positions,
        total_params=count_parameters(model),
        effective_params=count_effective_parameters(model),
    )


def topology_edit_distance(t1: TopologyState, t2: TopologyState) -> float:
    """Compute an edit distance between two topology states.

    The distance accounts for:
      1. Width changes: |size_1 - size_2| for each corresponding layer
      2. Depth changes: extra or missing layers
      3. Activation changes: 1 if dominant activation differs

    Normalized by the total size of the larger topology.

    Args:
        t1: First topology state.
        t2: Second topology state.

    Returns:
        Edit distance (0 = identical, higher = more different).
    """
    distance = 0.0
    max_size = 0.0

    # Match layers by index (assumes sequential ordering)
    n1, n2 = len(t1.layers), len(t2.layers)
    n_common = min(n1, n2)

    for i in range(n_common):
        l1, l2 = t1.layers[i], t2.layers[i]
        # Width change
        width_diff = abs(l1.out_size - l2.out_size)
        distance += width_diff
        max_size += max(l1.out_size, l2.out_size)
        # Input dimension change
        in_diff = abs(l1.in_size - l2.in_size)
        distance += in_diff
        max_size += max(l1.in_size, l2.in_size)
        # Activation change
        if l1.dominant_activation != l2.dominant_activation:
            distance += 1.0
            max_size += 1.0

    # Extra layers contribute their full size
    for i in range(n_common, max(n1, n2)):
        layers = t1.layers if n1 > n2 else t2.layers
        if i < len(layers):
            extra = layers[i]
            distance += extra.out_size + extra.in_size
            max_size += extra.out_size + extra.in_size

    if max_size == 0:
        return 0.0

    return distance / max_size
