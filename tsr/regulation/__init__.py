"""TSR Regulation Engine: structural plasticity monitor, signals, and actions."""

from tsr.regulation.monitor import StructuralPlasticityMonitor
from tsr.regulation.signals import (
    compute_death_signal,
    compute_bottleneck_signal,
    compute_layer_insertion_signal,
)
from tsr.regulation.actions import (
    prune_neurons_paired,
    grow_neurons_paired,
    apply_structural_update,
)
from tsr.regulation.scheduler import StructuralUpdateScheduler

__all__ = [
    "StructuralPlasticityMonitor",
    "compute_death_signal",
    "compute_bottleneck_signal",
    "compute_layer_insertion_signal",
    "prune_neurons_paired",
    "grow_neurons_paired",
    "apply_structural_update",
    "StructuralUpdateScheduler",
]
