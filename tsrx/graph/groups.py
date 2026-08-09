"""Coupling-group discovery (Definition 5.1) over a traced model.

Walks the fx graph in topological order, assigning each node's output a
"tap" — an abstract handle on the channel-identity of that tensor's
channel axis (dim=1). Producer modules (Conv/Linear) mint a fresh tap;
elementwise merges (residual add) union taps together; reshape/concat
mint a new tap connected to its source(s) via a layout, not identified
with them. The union-find's equivalence classes, once every node has been
visited, are exactly the coupling groups of Definition 5.1.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch.fx as fx
import torch.nn as nn

from tsrx.graph.generators import Role, classify_function, classify_module, is_depthwise
from tsrx.graph.trace import TracedModel, node_shape


class _UnionFind:
    def __init__(self):
        self._parent: Dict[int, int] = {}
        self._next_id = 0

    def new(self) -> int:
        i = self._next_id
        self._next_id += 1
        self._parent[i] = i
        return i

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra
        return self.find(ra)


@dataclass
class Block:
    """One channel-carrying tensor slot tied to a tap."""
    module_name: str
    kind: str          # "producer_out" | "consumer_in" | "affine" | "self_couple"
    axis: int          # which axis of the module's weight this is (0=out, 1=in)
    multiplicity: int = 1
    size: Optional[int] = None


@dataclass
class ReshapeInfo:
    node_name: str
    source_tap: int
    multiplicity: int


@dataclass
class ConcatInfo:
    node_name: str
    source_taps: List[int] = field(default_factory=list)
    offsets: List[int] = field(default_factory=list)


@dataclass
class CouplingGroup:
    tap: int
    blocks: List[Block] = field(default_factory=list)
    size: Optional[int] = None

    @property
    def producers(self) -> List[Block]:
        return [b for b in self.blocks if b.kind in ("producer_out", "self_couple")]

    @property
    def consumers(self) -> List[Block]:
        return [b for b in self.blocks if b.kind == "consumer_in"]

    @property
    def affine(self) -> List[Block]:
        return [b for b in self.blocks if b.kind == "affine"]

    @property
    def is_free(self) -> bool:
        """A 'free' site (Example 5.8): exactly one producer, coupled only
        to its own immediate consumer(s) — never merged via row 2/7/8."""
        return len(self.producers) == 1


@dataclass
class GroupingResult:
    groups: List[CouplingGroup]
    tap_of_module_output: Dict[str, int]  # module_name -> tap (for producer modules)
    reshapes: List[ReshapeInfo]
    concats: List[ConcatInfo]


def _first_tensor_arg(node: fx.Node):
    for a in node.args:
        if isinstance(a, fx.Node):
            return a
    return None


def _all_tensor_args(node: fx.Node) -> List[fx.Node]:
    return [a for a in node.args if isinstance(a, fx.Node)]


def discover_groups(traced: TracedModel) -> GroupingResult:
    gm = traced.gm
    uf = _UnionFind()
    tap_of: Dict[str, Optional[int]] = {}   # fx node name -> tap (or None = not channel-resizable)
    pending: Dict[int, List[Block]] = {}    # raw (pre-union) tap id -> blocks attached to it
    reshapes: List[ReshapeInfo] = []
    concats: List[ConcatInfo] = []
    tap_of_module_output: Dict[str, int] = {}
    # Multiplicity in effect at a node's OUTPUT (row 6: a flatten/reshape
    # aliases its tap to the source but each source index now backs `mult`
    # contiguous slots downstream). Default 1; propagated through
    # passthrough ops so a norm/activation between flatten and the next
    # consumer doesn't lose it.
    mult_of: Dict[str, int] = {}

    def tap_for(n: Optional[fx.Node]) -> Optional[int]:
        return None if n is None else tap_of.get(n.name)

    def mult_for(n: Optional[fx.Node]) -> int:
        return 1 if n is None else mult_of.get(n.name, 1)

    def attach(tap: int, module_name: str, kind: str, axis: int,
               size: Optional[int] = None, multiplicity: int = 1) -> None:
        pending.setdefault(tap, []).append(Block(module_name, kind, axis, multiplicity, size))

    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            tap_of[node.name] = None  # fixed by the dataset / a stored constant, not resizable
            continue
        if node.op == "output":
            continue

        if node.op == "call_module":
            mod = gm.get_submodule(node.target)
            role = classify_module(mod)
            in_node = _first_tensor_arg(node)
            in_tap = tap_for(in_node)
            in_mult = mult_for(in_node)

            if role == Role.PRODUCER_CONSUMER:
                out_tap = uf.new()
                if in_tap is not None:
                    attach(in_tap, node.target, "consumer_in", axis=1, size=_in_size(mod), multiplicity=in_mult)
                attach(out_tap, node.target, "producer_out", axis=0, size=_out_size(mod))
                tap_of[node.name] = out_tap
                tap_of_module_output[node.target] = out_tap

            elif role == Role.GROUPED_CONV:
                if isinstance(mod, nn.Conv2d) and is_depthwise(mod):
                    out_tap = in_tap if in_tap is not None else uf.new()
                    attach(out_tap, node.target, "self_couple", axis=0, size=_out_size(mod))
                    tap_of[node.name] = out_tap
                    tap_of_module_output[node.target] = out_tap
                else:
                    out_tap = uf.new()
                    if in_tap is not None:
                        attach(in_tap, node.target, "consumer_in", axis=1, size=_in_size(mod), multiplicity=in_mult)
                    attach(out_tap, node.target, "producer_out", axis=0, size=_out_size(mod))
                    tap_of[node.name] = out_tap
                    tap_of_module_output[node.target] = out_tap

            elif role == Role.AFFINE_NORM:
                out_tap = in_tap if in_tap is not None else uf.new()
                attach(out_tap, node.target, "affine", axis=0, size=_norm_size(mod))
                tap_of[node.name] = out_tap
                mult_of[node.name] = in_mult

            else:  # PASSTHROUGH or UNKNOWN module: best-effort passthrough
                tap_of[node.name] = in_tap
                mult_of[node.name] = in_mult
            continue

        if node.op in ("call_function", "call_method"):
            fn_name = getattr(node.target, "__name__", None) or str(node.target)
            role = classify_function(fn_name)

            if role == Role.ELEMENTWISE_MERGE:
                taps = [t for t in (tap_for(a) for a in _all_tensor_args(node)) if t is not None]
                merged = None
                for t in taps:
                    merged = t if merged is None else uf.union(merged, t)
                tap_of[node.name] = merged
                mult_of[node.name] = mult_for(_first_tensor_arg(node))
                continue

            if role == Role.RESHAPE:
                src = _first_tensor_arg(node)
                src_tap = tap_for(src)
                if src_tap is None:
                    tap_of[node.name] = None
                    continue
                # Row 6: NOT a new group — alias the tap through to the
                # source and record the multiplicity S so any consumer
                # reading past this reshape registers size k*S per
                # candidate, not k (Table 1: i -> {iS,...,(i+1)S-1}).
                s = _reshape_multiplicity(node_shape(src), node_shape(node))
                reshapes.append(ReshapeInfo(node.name, src_tap, s))
                tap_of[node.name] = src_tap
                mult_of[node.name] = mult_for(src) * s
                continue

            if role == Role.CONCAT:
                srcs = node.args[0] if node.args and isinstance(node.args[0], (list, tuple)) else _all_tensor_args(node)
                srcs = [s for s in srcs if isinstance(s, fx.Node)]
                new_tap = uf.new()
                offsets, running, src_taps = [], 0, []
                for s in srcs:
                    offsets.append(running)
                    t = tap_for(s)
                    if t is not None:
                        src_taps.append(t)
                    shp = node_shape(s)
                    running += int(shp[1]) if shp is not None and len(shp) > 1 else 0
                concats.append(ConcatInfo(node.name, src_taps, offsets))
                tap_of[node.name] = new_tap
                mult_of[node.name] = 1
                continue

            # PASSTHROUGH or UNKNOWN function/method: best-effort passthrough
            src = _first_tensor_arg(node)
            tap_of[node.name] = tap_for(src)
            mult_of[node.name] = mult_for(src)
            continue

        tap_of[node.name] = None

    # ---- finalize: partition attached blocks by union-find root ----
    root_blocks: Dict[int, List[Block]] = {}
    for tap_id, blk_list in pending.items():
        root = uf.find(tap_id)
        root_blocks.setdefault(root, []).extend(blk_list)

    groups = []
    for root, blk_list in root_blocks.items():
        sizes = {b.size for b in blk_list if b.kind in ("producer_out", "self_couple") and b.size is not None}
        size = next(iter(sizes)) if len(sizes) == 1 else None
        groups.append(CouplingGroup(tap=root, blocks=blk_list, size=size))

    tap_of_module_output = {k: uf.find(v) for k, v in tap_of_module_output.items()}

    return GroupingResult(groups=groups, tap_of_module_output=tap_of_module_output,
                           reshapes=reshapes, concats=concats)


def _out_size(mod: nn.Module) -> int:
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
        return mod.out_channels
    if isinstance(mod, nn.Linear):
        return mod.out_features
    raise TypeError(type(mod))


def _in_size(mod: nn.Module) -> int:
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
        return mod.in_channels
    if isinstance(mod, nn.Linear):
        return mod.in_features
    raise TypeError(type(mod))


def _norm_size(mod: nn.Module) -> Optional[int]:
    for attr in ("num_features", "num_channels"):
        if hasattr(mod, attr):
            return getattr(mod, attr)
    if isinstance(mod, nn.LayerNorm):
        shp = mod.normalized_shape
        return shp[0] if len(shp) == 1 else None
    return None


def _reshape_multiplicity(in_shape, out_shape) -> int:
    """Best-effort S for `(B,C,H,W) -> (B, C*H*W)`-style flattens."""
    if in_shape is None or out_shape is None or len(in_shape) < 3:
        return 1
    channel_count = int(in_shape[1])
    spatial = 1
    for d in in_shape[2:]:
        spatial *= int(d)
    if len(out_shape) == 2 and int(out_shape[1]) == channel_count * spatial:
        return spatial
    return 1
