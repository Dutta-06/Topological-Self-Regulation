"""
TSRTrainer: Main training loop implementing the two-timescale optimization.

Fast timescale (every step):
  - Forward pass through gated TSR layers
  - Compute loss and backpropagate
  - Optimizer step (updates weights, gates, activation mixtures)

Slow timescale (every K steps):
  - Query structural plasticity monitor for statistics
  - Compute death/bottleneck signals
  - Execute structural actions (prune/grow)
  - Rebuild optimizer with new parameter references

The trainer also handles:
  - Learning rate scheduling (warmup + cosine annealing)
  - FLOPs tracking for Benchmark 5
  - Topology snapshots for analysis
  - Checkpointing with full state (model + topology + optimizer + monitor)
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from tsr.model import TSRNetwork
from tsr.regulation.monitor import StructuralPlasticityMonitor
from tsr.regulation.actions import apply_structural_update, StructuralEvent
from tsr.regulation.scheduler import StructuralUpdateScheduler
from tsr.regulation.signals import gate_sparsity_penalty
from tsr.topology import capture_topology, TopologyState
from tsr.flops import compute_model_flops, CumulativeFLOPsTracker, differentiable_effective_flops
from tsr.utils import rebuild_optimizer, count_parameters, count_effective_parameters

logger = logging.getLogger(__name__)


class TSRTrainer:
    """Two-timescale trainer for TSR networks.

    Args:
        model: TSRNetwork instance.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Configuration dict (from Hydra or default.yaml).
        device: Torch device.
    """

    def __init__(
        self,
        model: TSRNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # ── Extract config ──
        train_cfg = config.get("training", {})
        reg_cfg = config.get("regulation", {})

        self.max_epochs = train_cfg.get("max_epochs", 200)
        self.max_steps = train_cfg.get("max_steps", None)
        self.eval_interval = train_cfg.get("eval_interval", 500)
        self.log_interval = config.get("logging", {}).get("log_interval", 50)
        self.checkpoint_interval = train_cfg.get("checkpoint_interval", 2000)
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")

        # ── Optimizer ──
        optimizer_name = train_cfg.get("optimizer", "adam")
        lr = train_cfg.get("learning_rate", 0.001)
        weight_decay = train_cfg.get("weight_decay", 0.0001)
        gate_lr_mult = train_cfg.get("gate_lr_multiplier", 1.0)
        act_lr_mult = train_cfg.get("act_lr_multiplier", 0.1)

        optimizer_cls = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd": torch.optim.SGD,
        }.get(optimizer_name, torch.optim.Adam)

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = {"lr": lr, "weight_decay": weight_decay}
        self.gate_lr_mult = gate_lr_mult
        self.act_lr_mult = act_lr_mult

        # ── Phantom sensors (measured growth signal) ──
        # Built BEFORE the optimizer so its probe params can be added to it.
        # Only constructed in phantom mode; probes are optimized alongside the model.
        self.growth_signal_mode = reg_cfg.get("growth_signal_mode", "phantom")
        self.phantom_manager = None
        if self.growth_signal_mode == "phantom":
            from tsr.regulation.phantom import PhantomManager
            self.phantom_manager = PhantomManager(
                model,
                k=reg_cfg.get("phantom_k", 4),
                window=reg_cfg.get("monitor_window", 100),
            ).to(device)

        self.optimizer = rebuild_optimizer(
            model, optimizer_cls, self.optimizer_kwargs,
            gate_lr_multiplier=gate_lr_mult,
            act_lr_multiplier=act_lr_mult,
        )
        self._add_phantom_param_group()

        # ── LR Scheduler ──
        warmup_steps = train_cfg.get("warmup_steps", 500)
        min_lr = train_cfg.get("min_lr", 1e-5)
        self.warmup_steps = warmup_steps

        # Warmup + cosine annealing
        total_steps = self.max_steps or (self.max_epochs * len(train_loader))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(min_lr / lr, 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress)))

        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        # ── Structural Plasticity ──
        self.monitor = StructuralPlasticityMonitor(
            model, window=reg_cfg.get("monitor_window", 100)
        )
        self.structural_scheduler = StructuralUpdateScheduler(
            update_interval=reg_cfg.get("update_interval", 200),
            cooldown_steps=reg_cfg.get("cooldown_steps", 50),
            anneal_structural_rate=reg_cfg.get("anneal_structural_rate", True),
            anneal_start_step=reg_cfg.get("anneal_start_step", 5000),
            anneal_factor=reg_cfg.get("anneal_factor", 0.95),
        )

        self.reg_config = {
            "death_threshold": reg_cfg.get("death_threshold", 0.01),
            "min_neurons": reg_cfg.get("min_neurons_per_layer", 4),
            "growth_enabled": reg_cfg.get("growth_enabled", True),
            "growth_rate": reg_cfg.get("growth_rate", 0.1),
            "max_neurons": reg_cfg.get("max_neurons_per_layer", 512),
            "bottleneck_threshold": reg_cfg.get("bottleneck_threshold", 0.1),
            "depth_adaptation_enabled": reg_cfg.get("layer_insertion_enabled", False),
            "layer_insertion_threshold": reg_cfg.get("layer_insertion_threshold", 5.0),
            "max_blocks": reg_cfg.get("max_blocks", 16),
            "growth_signal_mode": reg_cfg.get("growth_signal_mode", "phantom"),
            "phantom_threshold": reg_cfg.get("phantom_threshold", 0.05),
        }

        # Coefficient for the gate sparsity penalty added to the training loss.
        # This is what gives pruning teeth: it applies steady downward pressure
        # on every gate so that task-irrelevant neurons decay toward closed and
        # become prunable. 0.0 disables it (growth-only behavior).
        self.gate_sparsity_coeff = reg_cfg.get("gate_sparsity_coeff", 1e-4)

        # Fixed compute price (lambda): coefficient on the differentiable
        # effective-FLOPs term added to the loss. Prices the compute the
        # network actually uses so it sizes itself. 0.0 disables it.
        self.flops_price = reg_cfg.get("flops_price", 0.0)

        # ── FLOPs Tracking ──
        self.flops_tracker = CumulativeFLOPsTracker()
        self.input_shape = (3, 32, 32)  # default for CIFAR

        # ── Training State ──
        self.global_step = 0
        self.epoch = 0
        self.best_val_acc = 0.0
        self.structural_events: List[StructuralEvent] = []
        self.topology_history: List[TopologyState] = []
        self.metrics_history: List[dict] = []

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with final metrics and training history.
        """
        logger.info(f"Starting TSR training: {self.model.topology_summary()}")
        logger.info(f"Total params: {count_parameters(self.model):,}")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Record initial topology
        self.topology_history.append(capture_topology(self.model, 0))

        for epoch in range(self.max_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()

            if self.max_steps and self.global_step >= self.max_steps:
                break

        # Final evaluation
        final_metrics = self.evaluate()
        final_metrics["total_training_flops"] = self.flops_tracker.total_flops
        final_metrics["structural_overhead_pct"] = self.flops_tracker.overhead_percentage
        final_metrics["num_structural_events"] = len(self.structural_events)
        final_metrics["final_topology"] = self.model.topology_summary()

        logger.info(f"Training complete: {final_metrics}")
        return final_metrics

    def _train_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        from tqdm import tqdm
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.max_epochs}", leave=False)

        for batch_idx, (data, target) in enumerate(pbar):
            if self.max_steps and self.global_step >= self.max_steps:
                break

            data, target = data.to(self.device), target.to(self.device)

            # ── Fast timescale: gradient step ──
            self.optimizer.zero_grad()
            output = self.model(data)
            task_loss = F.cross_entropy(output, target)
            loss = task_loss

            # Gate sparsity penalty drives unused gates toward closed so that
            # the slow-timescale death signal can actually fire (makes pruning
            # work). Useful neurons resist via the task gradient.
            if self.gate_sparsity_coeff > 0:
                loss = loss + self.gate_sparsity_coeff * gate_sparsity_penalty(self.model)

            # Compute price: penalize the effective (gated) FLOPs the network
            # uses, so it sizes itself to the budget implied by lambda.
            if self.flops_price > 0:
                loss = loss + self.flops_price * differentiable_effective_flops(
                    self.model, self.input_shape
                )

            # Phantom sensors: hard-zero in value, but routes gradient to the
            # phantom gates so we can MEASURE the marginal value of capacity.
            if self.phantom_manager is not None:
                loss = loss + self.phantom_manager.aux_loss()

            loss.backward()

            # Snapshot phantom gate gradients (the measured growth signal)
            # before the optimizer zeroes them next step.
            if self.phantom_manager is not None:
                self.phantom_manager.record_gradients()

            self.optimizer.step()
            self.lr_scheduler.step()

            # Record the task loss (not the regularized loss) for plateau
            # detection, so the growth signal tracks real task progress.
            loss_val = task_loss.item()
            epoch_loss += loss_val
            num_batches += 1
            
            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"})

            # Record loss for monitor
            self.monitor.record_loss(loss_val)

            # Track FLOPs
            forward_flops = compute_model_flops(self.model, self.input_shape)
            self.flops_tracker.record_step(forward_flops, data.size(0))

            # ── Slow timescale: structural update ──
            if self.structural_scheduler.should_update(self.global_step):
                self._structural_update_step()

            # ── Logging ──
            if self.global_step % self.log_interval == 0:
                self._log_step(loss_val)

            # ── Evaluation ──
            if self.global_step % self.eval_interval == 0 and self.global_step > 0:
                val_metrics = self.evaluate()
                self.metrics_history.append({
                    "step": self.global_step,
                    "cumulative_flops": self.flops_tracker.total_flops,
                    **val_metrics,
                })

            # ── Checkpointing ──
            if self.global_step % self.checkpoint_interval == 0 and self.global_step > 0:
                self.save_checkpoint()

            self.global_step += 1

        return epoch_loss / max(num_batches, 1)

    def _add_phantom_param_group(self) -> None:
        """Add the phantom probe parameters to the optimizer as a group.

        Phantom params live on a separate module (not model.named_parameters),
        so rebuild_optimizer doesn't see them. They must be optimized so the
        probes learn useful candidate features. Called after every optimizer
        (re)build.
        """
        if self.phantom_manager is None:
            return
        phantom_params = [p for p in self.phantom_manager.parameters() if p.requires_grad]
        if phantom_params:
            self.optimizer.add_param_group({
                "params": phantom_params,
                "lr": self.optimizer_kwargs.get("lr", 0.001),
                "group_name": "phantom",
            })

    def _structural_update_step(self) -> None:
        """Execute one slow-timescale structural update."""
        t0 = time.time()

        events = apply_structural_update(
            model=self.model,
            monitor=self.monitor,
            step=self.global_step,
            phantom_manager=self.phantom_manager,
            **self.reg_config,
        )

        if events:
            self.structural_events.extend(events)

            # If a new block was inserted, the monitor's hooks don't cover it
            # (and layer names may have shifted). Re-register hooks against the
            # current module tree before rebuilding the optimizer so the new
            # block is both monitored and optimized.
            if any(e.action == "insert_layer" for e in events):
                self.monitor.refresh_hooks()

            # Any structural change (width or depth) can alter a probe's input
            # dimension (e.g. an upstream layer grew its channels), so rebuild
            # probes against the current shapes. refresh() reuses probes whose
            # input shape is unchanged and only re-creates the stale ones.
            if self.phantom_manager is not None:
                self.phantom_manager.refresh()

            # Rebuild optimizer with new parameter references
            self.optimizer = rebuild_optimizer(
                self.model,
                self.optimizer_cls,
                self.optimizer_kwargs,
                old_optimizer=self.optimizer,
                gate_lr_multiplier=self.gate_lr_mult,
                act_lr_multiplier=self.act_lr_mult,
            )
            self._add_phantom_param_group()

            # Re-create LR scheduler to match new optimizer
            # (preserves current step)
            current_step = self.global_step
            lr = self.optimizer_kwargs.get("lr", 0.001)
            min_lr = self.config.get("training", {}).get("min_lr", 1e-5)
            total_steps = self.max_steps or (self.max_epochs * len(self.train_loader))

            def lr_lambda(step):
                if step < self.warmup_steps:
                    return step / max(self.warmup_steps, 1)
                progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
                return max(min_lr / lr, 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress)))

            # Ensure initial_lr is set for the scheduler
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda, last_epoch=current_step)

            # Record topology snapshot
            self.topology_history.append(
                capture_topology(self.model, self.global_step)
            )

            # Log structural overhead
            overhead_ms = (time.time() - t0) * 1000
            logger.info(
                f"Step {self.global_step}: Structural update took {overhead_ms:.1f}ms, "
                f"new topology: {self.model.topology_summary()}"
            )

            # Record modified layers for scheduler cooldown
            modified_layers = [e.layer_name for e in events]
            self.structural_scheduler.record_update(self.global_step, modified_layers)
        else:
            # No events but still update scheduler
            self.structural_scheduler.record_update(self.global_step, [])

    def _log_step(self, loss: float) -> None:
        """Log training metrics for current step."""
        lr = self.optimizer.param_groups[0]["lr"]
        params = count_parameters(self.model)
        effective = count_effective_parameters(self.model)

        logger.info(
            f"Step {self.global_step} | Loss: {loss:.4f} | LR: {lr:.6f} | "
            f"Params: {params:,} | Effective: {effective:,} | "
            f"FLOPs: {self.flops_tracker.total_flops:,.0f} | "
            f"Topology: {self.model.topology_summary()}"
        )

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate model on validation set.

        Returns:
            Dict with val_loss, val_accuracy, params, effective_params, inference_flops.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        self.model.train()

        val_acc = correct / max(total, 1)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        return {
            "val_loss": total_loss / max(total, 1),
            "val_accuracy": val_acc,
            "best_val_accuracy": self.best_val_acc,
            "params": count_parameters(self.model),
            "effective_params": count_effective_parameters(self.model),
            "inference_flops": compute_model_flops(self.model, self.input_shape),
        }

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save full training state.

        Args:
            path: Override checkpoint path. Default: checkpoint_dir/step_XXXXX.pt

        Returns:
            Path to saved checkpoint.
        """
        if path is None:
            path = os.path.join(
                self.checkpoint_dir, f"step_{self.global_step:06d}.pt"
            )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "structural_scheduler_state_dict": self.structural_scheduler.state_dict(),
            "flops_tracker_state_dict": self.flops_tracker.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_acc": self.best_val_acc,
            "topology_state": capture_topology(self.model, self.global_step).to_dict(),
            "config": self.config,
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load training state from checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.structural_scheduler.load_state_dict(
            checkpoint["structural_scheduler_state_dict"]
        )
        self.flops_tracker.load_state_dict(checkpoint["flops_tracker_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]

        # Re-register monitor hooks (they don't survive state_dict)
        self.monitor.remove_hooks()
        self.monitor = StructuralPlasticityMonitor(
            self.model,
            window=self.config.get("regulation", {}).get("monitor_window", 100),
        )

        logger.info(f"Checkpoint loaded: {path} (step {self.global_step})")
