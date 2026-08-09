"""Index bundle (Definition 5.2): the layout that lets structural edits act
by index rather than by count.

A CouplingGroup from groups.py tells us WHICH modules share a channel
identity. IndexBundle adds the WHERE: for every parameter tensor and
buffer touched by the group, the axis and multiplicity needed to select
or extend along that shared index set. This is the object `edits.py` (M6)
consumes to grow/prune by index (Remark 5.4), and that `sense/candidates.py`
(M2) consumes to know which slots are a group's "port".
"""

from dataclasses import dataclass
from typing import Dict, List

import torch.nn as nn

from tsrx.graph.groups import Block, CouplingGroup, GroupingResult


@dataclass
class ParamSlot:
    """One (module, parameter, axis) tensor slot indexed by a group."""
    module_name: str
    param_name: str      # e.g. "weight", "bias", "running_mean"
    axis: int             # which axis of that tensor is indexed by the group
    role: str              # "producer_out" | "consumer_in" | "affine" | "self_couple"
    multiplicity: int = 1  # e.g. LSTM gate stack = 4, flatten = spatial size


@dataclass
class IndexBundle:
    """Per-group layout: every tensor slot the group's index set touches."""
    tap: int
    size: int
    slots: List[ParamSlot]
    quantum: int = 1   # divisibility requirement (e.g. GroupNorm's G) — edits must move q_G at a time

    @property
    def producer_slots(self) -> List[ParamSlot]:
        return [s for s in self.slots if s.role in ("producer_out", "self_couple")]

    @property
    def consumer_slots(self) -> List[ParamSlot]:
        return [s for s in self.slots if s.role == "consumer_in"]

    @property
    def affine_slots(self) -> List[ParamSlot]:
        return [s for s in self.slots if s.role == "affine"]


# weight/bias/buffer names to bundle per nn.Module type, keyed by role.
# axis follows PyTorch's parameter layout: Conv/Linear weight is (out, in, *k);
# norm affine + buffers are 1-D, indexed on axis 0.
_PRODUCER_PARAMS = ("weight", "bias")           # producer_out reads weight axis 0, bias axis 0
_CONSUMER_PARAMS = ("weight",)                   # consumer_in reads weight axis 1
_AFFINE_PARAMS = ("weight", "bias", "running_mean", "running_var")


def build_bundle(group: CouplingGroup, model: nn.Module) -> IndexBundle:
    modules: Dict[str, nn.Module] = dict(model.named_modules())
    slots: List[ParamSlot] = []
    quantum = 1

    for b in group.blocks:
        mod = modules.get(b.module_name)
        if mod is None:
            continue

        if b.kind in ("producer_out", "self_couple"):
            slots.append(ParamSlot(b.module_name, "weight", axis=0, role=b.kind))
            if getattr(mod, "bias", None) is not None:
                slots.append(ParamSlot(b.module_name, "bias", axis=0, role=b.kind))

        elif b.kind == "consumer_in":
            slots.append(ParamSlot(b.module_name, "weight", axis=1, role="consumer_in",
                                    multiplicity=b.multiplicity))

        elif b.kind == "affine":
            for pname in _AFFINE_PARAMS:
                if hasattr(mod, pname) and getattr(mod, pname) is not None:
                    slots.append(ParamSlot(b.module_name, pname, axis=0, role="affine"))
            num_groups = getattr(mod, "num_groups", None)
            if num_groups is not None and num_groups > 0:
                quantum = max(quantum, num_groups)

    return IndexBundle(tap=group.tap, size=group.size or 0, slots=slots, quantum=quantum)


def build_all_bundles(result: GroupingResult, model: nn.Module) -> Dict[int, IndexBundle]:
    return {g.tap: build_bundle(g, model) for g in result.groups}
