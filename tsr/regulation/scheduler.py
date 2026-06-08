"""
StructuralUpdateScheduler: Controls when structural updates occur.

Supports three modes:
  1. Periodic: update every K steps (like RigL)
  2. Adaptive: update when signals exceed threshold (more TSR-native)
  3. Periodic + annealing: reduce structural change rate over training

Includes cooldown enforcement to prevent oscillation (rapid prune-grow cycles)
and annealing to let the network stabilize as training progresses.
"""

from typing import Dict, Optional, Set
import logging

logger = logging.getLogger(__name__)


class StructuralUpdateScheduler:
    """Determines when structural updates should be applied.

    Usage:
        scheduler = StructuralUpdateScheduler(
            update_interval=200,
            cooldown_steps=50,
            anneal_structural_rate=True,
        )

        for step in range(max_steps):
            # ... training ...
            if scheduler.should_update(step):
                events = apply_structural_update(model, monitor, step, ...)
                scheduler.record_update(step, [e.layer_name for e in events])

    Args:
        update_interval: Steps between structural update checks.
        cooldown_steps: Minimum steps between updates for the same layer.
        anneal_structural_rate: Whether to reduce update frequency over training.
        anneal_start_step: When to start annealing.
        anneal_factor: Multiplicative factor applied to update_interval each period.
        max_interval: Maximum update interval after annealing.
    """

    def __init__(
        self,
        update_interval: int = 200,
        cooldown_steps: int = 50,
        anneal_structural_rate: bool = True,
        anneal_start_step: int = 5000,
        anneal_factor: float = 0.95,
        max_interval: int = 2000,
    ):
        self.base_interval = update_interval
        self.current_interval = update_interval
        self.cooldown_steps = cooldown_steps
        self.anneal_structural_rate = anneal_structural_rate
        self.anneal_start_step = anneal_start_step
        self.anneal_factor = anneal_factor
        self.max_interval = max_interval

        # Track last update step per layer (for cooldown)
        self._last_update: Dict[str, int] = {}
        # Track last global update step
        self._last_global_update: int = -update_interval  # allow first update

    def should_update(self, step: int) -> bool:
        """Check if a structural update should occur at this step.

        Args:
            step: Current training step.

        Returns:
            True if enough steps have passed since the last update.
        """
        return (step - self._last_global_update) >= self.current_interval

    def is_layer_cooled_down(self, layer_name: str, step: int) -> bool:
        """Check if a specific layer has cooled down since its last modification.

        Args:
            layer_name: Name of the TSR layer.
            step: Current training step.

        Returns:
            True if the layer can be modified (cooldown has elapsed).
        """
        last = self._last_update.get(layer_name, -self.cooldown_steps)
        return (step - last) >= self.cooldown_steps

    def record_update(self, step: int, modified_layers: list) -> None:
        """Record that a structural update occurred.

        Args:
            step: Current training step.
            modified_layers: Names of layers that were modified.
        """
        self._last_global_update = step
        for name in modified_layers:
            self._last_update[name] = step

        # Apply annealing
        if self.anneal_structural_rate and step >= self.anneal_start_step:
            # Increase interval (reduce frequency)
            self.current_interval = min(
                int(self.current_interval / self.anneal_factor),
                self.max_interval,
            )
            logger.debug(
                f"Step {step}: Structural update interval annealed to "
                f"{self.current_interval}"
            )

    def get_eligible_layers(self, step: int, all_layers: list) -> list:
        """Get layers that are eligible for structural changes (past cooldown).

        Args:
            step: Current training step.
            all_layers: Names of all TSR layers.

        Returns:
            List of layer names that can be modified.
        """
        return [
            name for name in all_layers if self.is_layer_cooled_down(name, step)
        ]

    def reset(self) -> None:
        """Reset scheduler state."""
        self._last_update.clear()
        self._last_global_update = -self.base_interval
        self.current_interval = self.base_interval

    def state_dict(self) -> dict:
        """Serialize scheduler state for checkpointing."""
        return {
            "base_interval": self.base_interval,
            "current_interval": self.current_interval,
            "last_update": dict(self._last_update),
            "last_global_update": self._last_global_update,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore scheduler state from checkpoint."""
        self.base_interval = state["base_interval"]
        self.current_interval = state["current_interval"]
        self._last_update = state["last_update"]
        self._last_global_update = state["last_global_update"]
