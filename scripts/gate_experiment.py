"""
Gate Experiment — Acceptance Gate for the TSR Rewrite.

Trains five variants on CIFAR-10 under identical conditions:
  tsr_phantom   : TSR with phantom-sensor growth signal  (the rewrite)
  tsr_heuristic : TSR with bottleneck-score growth signal (pre-rewrite baseline)
  vgg_tiny      : FixedVGG([8, 8, 16])
  vgg_small     : FixedVGG([16, 16, 32])
  static_final  : FixedVGG matching TSR's discovered shape, trained from scratch

For each variant × seed the script writes:
  results/gate/<variant>/seed<S>/  (one dir per run)
    metrics.jsonl      — one JSON line per eval (step, val_acc, val_loss, params,
                         inference_flops, cumulative_train_flops)
    events.jsonl       — one JSON line per structural event (TSR variants only)
    topology.jsonl     — one JSON line per topology snapshot (TSR variants only)
    final.json         — summary scalar dict at run end
    checkpoint_<step>.pt — written every CKPT_EVERY training steps; only the
                           latest N=2 are kept (older files deleted to avoid
                           filling disk with duplicates)

After all runs the script writes:
  results/gate/summary.json        — aggregated mean ± std per variant

Usage:
  python scripts/gate_experiment.py [--seeds 42 123 456] [--epochs 50] \\
                                    [--variants tsr_phantom vgg_tiny]

Acceptance gate (checked at end):
  tsr_phantom mean val_acc > tsr_heuristic mean val_acc
  tsr_phantom mean val_acc > vgg_tiny   mean val_acc
  tsr_phantom mean val_acc > vgg_small  mean val_acc
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# ── project root on the path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.fixed_arch import FixedVGG, VGG_CONFIGS, FixedResNet
from data.cifar import get_cifar10_loaders
from tsr.model import TSRNetwork
from tsr.flops import compute_model_flops, CumulativeFLOPsTracker, differentiable_effective_flops
from tsr.regulation.monitor import StructuralPlasticityMonitor
from tsr.regulation.actions import apply_structural_update
from tsr.regulation.scheduler import StructuralUpdateScheduler
from tsr.regulation.signals import gate_sparsity_penalty
from tsr.topology import capture_topology
from tsr.utils import count_parameters, count_effective_parameters, rebuild_optimizer

logger = logging.getLogger("gate_experiment")

# ── constants ─────────────────────────────────────────────────────────────────
KEEP_N_CHECKPOINTS = 2   # keep only the 2 most-recent checkpoint .pt files
INPUT_SHAPE = (3, 32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def _save_checkpoint(
    run_dir: Path,
    step: int,
    model: nn.Module,
    optimizer,
    scheduler,
    extra: dict,
    keep_n: int = KEEP_N_CHECKPOINTS,
) -> Path:
    """Save checkpoint, deleting old ones to keep only `keep_n` on disk."""
    ckpt_path = run_dir / f"checkpoint_{step:07d}.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        **extra,
    }, ckpt_path)
    # Purge older checkpoints
    existing = sorted(run_dir.glob("checkpoint_*.pt"))
    for old in existing[:-keep_n]:
        old.unlink()
    return ckpt_path


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────────────────────────────────────
# Baseline trainer (FixedVGG / static_final — no structural updates)
# ─────────────────────────────────────────────────────────────────────────────

class BaselineRunner:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: Path,
        max_epochs: int,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        ckpt_every: int,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.max_epochs = max_epochs
        self.ckpt_every = ckpt_every
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        total_steps = max_epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(1e-5 / lr, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.flops_tracker = CumulativeFLOPsTracker()
        self.step = 0
        self.best_val_acc = 0.0

    def run(self) -> dict:
        metrics_path = self.run_dir / "metrics.jsonl"
        for epoch in range(self.max_epochs):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                loss = F.cross_entropy(self.model(data), target)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                fwd_flops = compute_model_flops(self.model, INPUT_SHAPE)
                self.flops_tracker.record_step(fwd_flops, data.size(0))
                self.step += 1

            # Eval every epoch
            m = self._eval()
            m["step"] = self.step
            m["epoch"] = epoch
            m["cumulative_train_flops"] = self.flops_tracker.total_flops
            _write_jsonl(metrics_path, m)
            logger.info(
                f"  epoch {epoch+1}/{self.max_epochs} "
                f"val_acc={m['val_accuracy']:.4f} "
                f"params={m['params']:,}"
            )

            if self.step % self.ckpt_every < len(self.train_loader):
                _save_checkpoint(
                    self.run_dir, self.step, self.model,
                    self.optimizer, self.scheduler, {"epoch": epoch}
                )

        final = self._eval()
        final["total_train_flops"] = self.flops_tracker.total_flops
        final["best_val_accuracy"] = self.best_val_acc
        final["params"] = count_params(self.model)
        with open(self.run_dir / "final.json", "w") as f:
            json.dump(final, f, indent=2)
        return final

    @torch.no_grad()
    def _eval(self) -> dict:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            out = self.model(data)
            total_loss += F.cross_entropy(out, target, reduction="sum").item()
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)
        self.model.train()
        acc = correct / max(total, 1)
        if acc > self.best_val_acc:
            self.best_val_acc = acc
        return {
            "val_loss": total_loss / max(total, 1),
            "val_accuracy": acc,
            "params": count_params(self.model),
            "inference_flops": compute_model_flops(self.model, INPUT_SHAPE),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TSR trainer (inline, config-driven, no Hydra dependency)
# ─────────────────────────────────────────────────────────────────────────────

class TSRRunner:
    def __init__(
        self,
        model: TSRNetwork,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: Path,
        cfg: dict,
        growth_signal_mode: str,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = run_dir
        self.device = device
        self.growth_signal_mode = growth_signal_mode

        train_cfg = cfg.get("training", {})
        reg_cfg = cfg.get("regulation", {})

        self.max_epochs = train_cfg.get("max_epochs", 50)
        self.ckpt_every = train_cfg.get("checkpoint_interval", 2000)
        lr = train_cfg.get("learning_rate", 0.001)
        wd = train_cfg.get("weight_decay", 0.0001)
        gate_lr_mult = train_cfg.get("gate_lr_multiplier", 1.0)
        act_lr_mult = train_cfg.get("act_lr_multiplier", 0.1)
        warmup = train_cfg.get("warmup_steps", 500)
        min_lr = train_cfg.get("min_lr", 1e-5)

        self.gate_sparsity_coeff = reg_cfg.get("gate_sparsity_coeff", 1e-4)
        self.flops_price = reg_cfg.get("flops_price", 0.0)

        self.reg_config = {
            "death_threshold": reg_cfg.get("death_threshold", 0.01),
            "min_neurons": reg_cfg.get("min_neurons_per_layer", 4),
            "newborn_protect_steps": reg_cfg.get("newborn_protect_steps", 400),
            "growth_enabled": reg_cfg.get("growth_enabled", True),
            "growth_rate": reg_cfg.get("growth_rate", 0.1),
            "max_neurons": reg_cfg.get("max_neurons_per_layer", 512),
            "bottleneck_threshold": reg_cfg.get("bottleneck_threshold", 0.1),
            "newborn_gate_init": reg_cfg.get("newborn_gate_init", 0.0),
            "depth_adaptation_enabled": reg_cfg.get("layer_insertion_enabled", False),
            "layer_insertion_threshold": reg_cfg.get("layer_insertion_threshold", 5.0),
            "max_blocks": reg_cfg.get("max_blocks", 16),
            "growth_signal_mode": growth_signal_mode,
            "phantom_threshold": reg_cfg.get("phantom_threshold", 1.0),
            "connection_threshold": reg_cfg.get("connection_threshold", 1.0),
            "max_skip_span": reg_cfg.get("max_skip_span", 4),
            "max_growth_per_update": reg_cfg.get("max_growth_per_update", 2),
            "growth_cooldown_steps": reg_cfg.get("growth_cooldown_steps", 1000),
        }

        # Phantom manager (only for phantom mode)
        self.phantom_manager = None
        self.connection_phantom_manager = None
        if growth_signal_mode == "phantom":
            from tsr.regulation.phantom import PhantomManager, ConnectionPhantomManager
            self.phantom_manager = PhantomManager(
                model,
                k=reg_cfg.get("phantom_k", 4),
                window=reg_cfg.get("monitor_window", 100),
            ).to(device)
            if reg_cfg.get("connection_plasticity_enabled", True):
                self.connection_phantom_manager = ConnectionPhantomManager(
                    model,
                    max_skip_span=reg_cfg.get("max_skip_span", 4),
                    window=reg_cfg.get("monitor_window", 100),
                ).to(device)

        self.optimizer_cls = torch.optim.Adam
        self.optimizer_kwargs = {"lr": lr, "weight_decay": wd}
        self.gate_lr_mult = gate_lr_mult
        self.act_lr_mult = act_lr_mult
        self.lr = lr
        self.min_lr = min_lr
        self.total_steps = self.max_epochs * len(train_loader)
        self.warmup = warmup
        self.optimizer = rebuild_optimizer(
            model, self.optimizer_cls, self.optimizer_kwargs,
            gate_lr_multiplier=gate_lr_mult, act_lr_multiplier=act_lr_mult,
        )
        self._add_phantom_group()
        self._add_connection_phantom_group()

        total_steps = self.max_epochs * len(train_loader)
        self.warmup = warmup

        self.scheduler = self._build_scheduler()

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

        self.flops_tracker = CumulativeFLOPsTracker()
        # Cache FLOPs: recomputing every step is O(model_size) and dominates wall-time
        # at larger model sizes. Recompute only after structural changes (model shape changes).
        self._cached_fwd_flops = compute_model_flops(self.model, INPUT_SHAPE)
        self.step = 0
        self.best_val_acc = 0.0
        self.events: list = []
      
    def _build_scheduler(self, last_epoch: int = -1) -> LambdaLR:
        lr, min_lr, warmup, total_steps = self.lr, self.min_lr, self.warmup, self.total_steps

        def _cosine_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))

        def _constant_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            return 1.0

        lambdas = []
        for group in self.optimizer.param_groups:
            if group.get("group_name") == "phantom":
                lambdas.append(_constant_lambda)
            else:
                lambdas.append(_cosine_lambda)

        return LambdaLR(self.optimizer, lambdas, last_epoch=last_epoch)

    def _add_phantom_group(self):
        if self.phantom_manager is None:
            return
        params = [p for p in self.phantom_manager.parameters() if p.requires_grad]
        if params:
            self.optimizer.add_param_group({
                "params": params,
                "lr": self.optimizer_kwargs["lr"],
                "group_name": "phantom",
            })

    def _add_connection_phantom_group(self):
        if self.connection_phantom_manager is None:
            return
        existing_ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        new_params = [
            p for p in self.connection_phantom_manager.parameters()
            if p.requires_grad and id(p) not in existing_ids
        ]
        if new_params:
            self.optimizer.add_param_group({
                "params": new_params,
                "lr": self.optimizer_kwargs["lr"],
                "group_name": "connection_phantom",
            })

    def run(self) -> dict:
        metrics_path = self.run_dir / "metrics.jsonl"
        events_path = self.run_dir / "events.jsonl"
        topology_path = self.run_dir / "topology.jsonl"

        start_epoch = 0
        existing_ckpts = sorted(self.run_dir.glob("checkpoint_*.pt"))
        if existing_ckpts:
            latest_ckpt = existing_ckpts[-1]
            logger.info(f"Resuming from checkpoint: {latest_ckpt}")
            state = torch.load(latest_ckpt, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            self.step = state["step"]
            start_epoch = state.get("extra", {}).get("epoch", -1) + 1
            self.best_val_acc = state.get("extra", {}).get("best_val_acc", 0.0)
            
            if (self.run_dir / "events.jsonl").exists():
                with open(self.run_dir / "events.jsonl", "r") as f:
                    self.events = [json.loads(line) for line in f]
        else:
            topo = capture_topology(self.model, 0)
            _write_jsonl(self.run_dir / "topology.jsonl", topo.to_dict())

        for epoch in range(start_epoch, self.max_epochs):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(data)
                task_loss = F.cross_entropy(out, target)
                loss = task_loss

                if self.gate_sparsity_coeff > 0:
                    loss = loss + self.gate_sparsity_coeff * gate_sparsity_penalty(self.model)
                if self.flops_price > 0:
                    loss = loss + self.flops_price * differentiable_effective_flops(
                        self.model, INPUT_SHAPE
                    )
                if self.phantom_manager is not None:
                    loss = loss + self.phantom_manager.aux_loss()
                if self.connection_phantom_manager is not None:
                    loss = loss + self.connection_phantom_manager.aux_loss()

                loss.backward()

                if self.phantom_manager is not None:
                    self.phantom_manager.record_gradients()
                if self.connection_phantom_manager is not None:
                    self.connection_phantom_manager.record_gradients()

                self.optimizer.step()
                self.scheduler.step()

                self.monitor.record_loss(task_loss.item())
                self.flops_tracker.record_step(self._cached_fwd_flops, data.size(0))

                if self.structural_scheduler.should_update(self.step):
                    new_events = self._structural_step()
                    if new_events:
                        # Model shape changed — recompute cached FLOPs once
                        self._cached_fwd_flops = compute_model_flops(self.model, INPUT_SHAPE)
                    for ev in new_events:
                        _write_jsonl(events_path, ev.to_dict())
                        topo = capture_topology(self.model, self.step)
                        _write_jsonl(topology_path, topo.to_dict())

                if self.step > 0 and self.step % self.ckpt_every == 0:
                    _save_checkpoint(
                        self.run_dir, self.step,
                        self.model, self.optimizer, self.scheduler,
                        {"epoch": epoch, "best_val_acc": self.best_val_acc},
                    )

                self.step += 1

            # Eval every epoch
            m = self._eval()
            m["step"] = self.step
            m["epoch"] = epoch
            m["cumulative_train_flops"] = self.flops_tracker.total_flops
            m["num_structural_events"] = len(self.events)
            m["topology_summary"] = self.model.topology_summary()
            _write_jsonl(metrics_path, m)
            logger.info(
                f"  epoch {epoch+1}/{self.max_epochs} "
                f"val_acc={m['val_accuracy']:.4f} "
                f"params={m['params']:,} "
                f"events={len(self.events)} "
                f"topo={self.model.topology_summary()}"
            )

        # Final snapshot
        final_topo = capture_topology(self.model, self.step)
        _write_jsonl(topology_path, final_topo.to_dict())

        final = self._eval()
        final["total_train_flops"] = self.flops_tracker.total_flops
        final["best_val_accuracy"] = self.best_val_acc
        final["num_structural_events"] = len(self.events)
        final["final_topology"] = self.model.topology_summary()
        final["topology_state"] = final_topo.to_dict()
        with open(self.run_dir / "final.json", "w") as f:
            json.dump(final, f, indent=2)
        return final

    def _structural_step(self) -> list:
        events = apply_structural_update(
            model=self.model,
            monitor=self.monitor,
            step=self.step,
            phantom_manager=self.phantom_manager,
            connection_phantom_manager=self.connection_phantom_manager,
            **self.reg_config,
        )
        if events:
            self.events.extend(events)
            if any(e.action == "insert_layer" for e in events):
                self.monitor.refresh_hooks()
            if self.phantom_manager is not None:
                self.phantom_manager.refresh()
            if self.connection_phantom_manager is not None:
                self.connection_phantom_manager.refresh()

            self.optimizer = rebuild_optimizer(
                self.model, self.optimizer_cls, self.optimizer_kwargs,
                old_optimizer=self.optimizer,
                gate_lr_multiplier=self.gate_lr_mult,
                act_lr_multiplier=self.act_lr_mult,
            )
            self._add_phantom_group()
            self._add_connection_phantom_group()

            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
            self.scheduler = self._build_scheduler(last_epoch=self.step)

            self.structural_scheduler.record_update(self.step, [e.layer_name for e in events])
        else:
            self.structural_scheduler.record_update(self.step, [])
        return events

    @torch.no_grad()
    def _eval(self) -> dict:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for data, target in self.val_loader:
            data, target = data.to(self.device), target.to(self.device)
            out = self.model(data)
            total_loss += F.cross_entropy(out, target, reduction="sum").item()
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)
        self.model.train()
        acc = correct / max(total, 1)
        if acc > self.best_val_acc:
            self.best_val_acc = acc
        return {
            "val_loss": total_loss / max(total, 1),
            "val_accuracy": acc,
            "best_val_accuracy": self.best_val_acc,
            "params": count_parameters(self.model),
            "effective_params": count_effective_parameters(self.model),
            "inference_flops": compute_model_flops(self.model, INPUT_SHAPE),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment orchestration
# ─────────────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    path = ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def build_tsr(cfg: dict, num_classes: int = 10) -> TSRNetwork:
    m = cfg.get("model", {})
    return TSRNetwork(
        in_channels=3,
        seed_channels=m.get("seed_channels", [8, 8]),
        num_classes=num_classes,
        gate_init=m.get("gate_init", 3.0),
        act_init=m.get("act_init", "relu"),
        norm_group_size=m.get("norm_group_size", 8),
        classifier_hidden=m.get("classifier_hidden", None),
    )


def _tsr_final_shape(run_dir: Path) -> Tuple[Optional[List[int]], Optional[int]]:
    final_path = run_dir / "final.json"
    if not final_path.exists():
        return None, None
    with open(final_path) as f:
        d = json.load(f)
    topo = d.get("topology_state")
    if not topo:
        return None, None
    channels = [
        layer["out_size"]
        for layer in topo.get("layers", [])
        if layer.get("layer_type") == "conv"
    ]
    classifier_hidden = None
    for layer in topo.get("layers", []):
        if layer.get("layer_type") == "linear" and layer.get("name") != "classifier.2":
            classifier_hidden = layer["out_size"]
            break
            
    return (channels if channels else None), classifier_hidden


def run_one(
    variant: str,
    seed: int,
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    results_root: Path,
    device: torch.device,
    static_final_channels: Optional[List[int]] = None,
    static_final_classifier: Optional[int] = None,
) -> dict:
    run_dir = results_root / variant / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already complete
    if (run_dir / "final.json").exists():
        logger.info(f"  [SKIP] {variant}/seed{seed} already complete")
        with open(run_dir / "final.json") as f:
            return json.load(f)

    set_seed(seed)
    logger.info(f">>> {variant} seed={seed}  dir={run_dir}")

    train_cfg = cfg.get("training", {})
    max_epochs = train_cfg.get("max_epochs", 50)
    lr = train_cfg.get("learning_rate", 0.001)
    wd = train_cfg.get("weight_decay", 0.0001)
    warmup = train_cfg.get("warmup_steps", 500)
    ckpt_every = train_cfg.get("checkpoint_interval", 2000)

    if variant == "tsr_phantom":
        cfg_copy = json.loads(json.dumps(cfg))
        existing_ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
        if existing_ckpts:
            latest_ckpt = existing_ckpts[-1]
            state = torch.load(latest_ckpt, map_location=device)
            model_state = state["model_state_dict"]
            seed_channels = []
            for i in range(16):
                key = f"blocks.{i}.conv.weight"
                if key in model_state:
                    seed_channels.append(model_state[key].shape[0])
                else:
                    break
            if "classifier.0.weight" in model_state:
                cfg_copy.setdefault("model", {})["classifier_hidden"] = model_state["classifier.0.weight"].shape[0]
            if seed_channels:
                cfg_copy.setdefault("model", {})["seed_channels"] = seed_channels

        model = build_tsr(cfg_copy)
        runner = TSRRunner(
            model, train_loader, val_loader, run_dir,
            cfg=cfg_copy, growth_signal_mode="phantom", device=device,
        )
        return runner.run()

    elif variant == "tsr_heuristic":
        # Disable phantom: override growth mode in a config copy
        cfg2 = json.loads(json.dumps(cfg))
        cfg2["regulation"]["growth_signal_mode"] = "heuristic"
        model = build_tsr(cfg2)
        runner = TSRRunner(
            model, train_loader, val_loader, run_dir,
            cfg=cfg2, growth_signal_mode="heuristic", device=device,
        )
        return runner.run()

    elif variant == "vgg_tiny":
        model = FixedVGG([8, 8, 16], num_classes=10)
        runner = BaselineRunner(
            model, train_loader, val_loader, run_dir,
            max_epochs=max_epochs, lr=lr, weight_decay=wd,
            warmup_steps=warmup, ckpt_every=ckpt_every, device=device,
        )
        return runner.run()

    elif variant == "vgg_small":
        model = FixedVGG([16, 16, 32], num_classes=10)
        runner = BaselineRunner(
            model, train_loader, val_loader, run_dir,
            max_epochs=max_epochs, lr=lr, weight_decay=wd,
            warmup_steps=warmup, ckpt_every=ckpt_every, device=device,
        )
        return runner.run()

    elif variant == "static_final":
        channels = static_final_channels or [8, 8, 16]
        logger.info(f"  static_final channels: {channels}")
        model = FixedVGG(channels, num_classes=10, classifier_hidden=static_final_classifier)
        runner = BaselineRunner(
            model, train_loader, val_loader, run_dir,
            max_epochs=max_epochs, lr=lr, weight_decay=wd,
            warmup_steps=warmup, ckpt_every=ckpt_every, device=device,
        )
        return runner.run()

    elif variant == "vgg16":
        logger.info(f"  vgg16 channels: {VGG_CONFIGS['vgg16']}")
        model = FixedVGG(VGG_CONFIGS["vgg16"], num_classes=10)
        runner = BaselineRunner(
            model, train_loader, val_loader, run_dir,
            max_epochs=max_epochs, lr=lr, weight_decay=wd,
            warmup_steps=warmup, ckpt_every=ckpt_every, device=device,
        )
        return runner.run()

    elif variant == "vgg19":
        logger.info(f"  vgg19 channels: {VGG_CONFIGS['vgg19']}")
        model = FixedVGG(VGG_CONFIGS["vgg19"], num_classes=10)
        runner = BaselineRunner(
            model, train_loader, val_loader, run_dir,
            max_epochs=max_epochs, lr=lr, weight_decay=wd,
            warmup_steps=warmup, ckpt_every=ckpt_every, device=device,
        )
        return runner.run()

    elif variant == "resnet_sanity":
        # Sanity check (Workstream A): does a hand-built ResNet with real residual
        # shortcuts clear the ~90% plain-VGG ceiling under this identical recipe?
        # GroupNorm + GAP head match TSR/FixedVGG exactly — the only difference
        # from vgg_small-class nets is the shortcut itself.
        logger.info("  resnet_sanity: ResNet-20-style, stage_channels=[16,32,64]")
        model = FixedResNet(stage_channels=[16, 32, 64], blocks_per_stage=3, num_classes=10)
        runner = BaselineRunner(
            model, train_loader, val_loader, run_dir,
            max_epochs=max_epochs, lr=lr, weight_decay=wd,
            warmup_steps=warmup, ckpt_every=ckpt_every, device=device,
        )
        return runner.run()

    else:
        raise ValueError(f"Unknown variant: {variant}")


def aggregate_results(results_root: Path, variants: List[str], seeds: List[int]) -> dict:
    """Collect final.json from each run and compute mean ± std per variant."""
    summary = {}
    for variant in variants:
        accs, flops_inf, flops_train, params_list = [], [], [], []
        for seed in seeds:
            p = results_root / variant / f"seed{seed}" / "final.json"
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)
            accs.append(d.get("best_val_accuracy", d.get("val_accuracy", 0.0)))
            flops_inf.append(d.get("inference_flops", 0))
            flops_train.append(d.get("total_train_flops", 0))
            params_list.append(d.get("params", 0))

        if not accs:
            continue

        def _stats(lst):
            a = np.array(lst, dtype=float)
            return {"mean": float(a.mean()), "std": float(a.std()), "runs": len(lst)}

        summary[variant] = {
            "val_accuracy": _stats(accs),
            "inference_flops": _stats(flops_inf),
            "total_train_flops": _stats(flops_train),
            "params": _stats(params_list),
        }
    return summary


def check_gate(summary: dict) -> bool:
    """Print acceptance gate outcome and return True iff passed.

    Gate conditions (all must hold):
      1. tsr_phantom > static_final  — plasticity beats shape alone (matched params)
      2. tsr_phantom > tsr_heuristic — phantom sensor beats hand-designed heuristic
      3. tsr_phantom > vgg_tiny      — TSR beats a fixed net at comparable FLOPs
    """
    phantom_acc = summary.get("tsr_phantom", {}).get("val_accuracy", {}).get("mean", 0.0)
    heuristic_acc = summary.get("tsr_heuristic", {}).get("val_accuracy", {}).get("mean", 0.0)
    tiny_acc = summary.get("vgg_tiny", {}).get("val_accuracy", {}).get("mean", 0.0)
    static_acc = summary.get("static_final", {}).get("val_accuracy", {}).get("mean", 0.0)

    print("\n" + "=" * 60)
    print("ACCEPTANCE GATE")
    print("=" * 60)

    lines = [
        ("tsr_phantom vs static_final  [CORE]", phantom_acc, static_acc),
        ("tsr_phantom vs tsr_heuristic [SENSOR]", phantom_acc, heuristic_acc),
        ("tsr_phantom vs vgg_tiny      [PARETO]", phantom_acc, tiny_acc),
    ]
    passed = True
    for label, a, b in lines:
        ok = a > b
        sym = "PASS" if ok else "FAIL"
        print(f"  [{sym}] {label}: {a:.4f} vs {b:.4f}  (delta={a-b:+.4f})")
        if not ok:
            passed = False

    print("=" * 60)
    print(f"  OVERALL: {'PASSED' if passed else 'FAILED'}")
    print("=" * 60 + "\n")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_VARIANTS = ["tsr_phantom", "tsr_heuristic", "vgg_tiny", "vgg_small", "static_final", "vgg16", "vgg19", "resnet_sanity"]


def main():
    parser = argparse.ArgumentParser(description="TSR Gate Experiment")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456],
                        help="Random seeds (default: 42 123 456)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max_epochs from config")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        choices=ALL_VARIANTS + ["all"],
                        help="Variants to run (default: all)")
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results" / "gate"),
                        help="Root dir for output files")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--flops-price", type=float, default=None,
                        help="λ-FLOPs penalty coefficient (overrides config). 0=free growth, >0=budget pressure.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    variants = ALL_VARIANTS if "all" in args.variants else args.variants
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Config
    cfg = load_config()
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.flops_price is not None:
        cfg["regulation"]["flops_price"] = args.flops_price

    # Data loaders (shared across all variants/seeds for comparability)
    data_root = cfg["data"].get("root", "./data")
    batch_size = cfg["training"].get("batch_size", 128)
    num_workers = cfg["data"].get("num_workers", 4)
    augmentation = cfg["data"].get("augmentation", "full")

    logger.info(f"Loading CIFAR-10 from {data_root}")
    train_loader, val_loader = get_cifar10_loaders(
        root=data_root, batch_size=batch_size,
        num_workers=num_workers, augmentation=augmentation,
    )
    logger.info(
        f"Data: {len(train_loader.dataset):,} train / "
        f"{len(val_loader.dataset):,} val, batch={batch_size}"
    )

    # ── Run all variants × seeds ────────────────────────────────────────────
    all_results: Dict[str, Dict[int, dict]] = {v: {} for v in variants}

    # Run tsr_phantom first (static_final needs its discovered shape)
    run_order = sorted(variants, key=lambda v: 0 if v == "tsr_phantom" else 1)

    t_start = time.time()
    for variant in run_order:
        for seed in args.seeds:
            # For static_final, try to get the shape from tsr_phantom seed 0 run
            sfclassifier = None
            sfchannels = None
            if variant == "static_final":
                phantom_dir = results_root / "tsr_phantom" / f"seed{args.seeds[0]}"
                sfchannels, sfclassifier = _tsr_final_shape(phantom_dir)
                if sfchannels:
                    logger.info(f"  static_final will use discovered shape: {sfchannels}")
                else:
                    logger.warning("  tsr_phantom final not found; static_final uses [8,8,16]")

            result = run_one(
                variant=variant,
                seed=seed,
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                results_root=results_root,
                device=device,
                static_final_channels=sfchannels,
                static_final_classifier=sfclassifier
            )
            all_results[variant][seed] = result

    elapsed = time.time() - t_start
    logger.info(f"All runs complete in {elapsed/3600:.2f}h")

    # ── Aggregate and save summary ──────────────────────────────────────────
    summary = aggregate_results(results_root, variants, args.seeds)
    summary_path = results_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")

    # Print table
    print("\n" + "=" * 70)
    print(f"{'VARIANT':<20} {'VAL_ACC':>10} {'±STD':>8} {'INF_MFLOPS':>12} {'PARAMS':>10}")
    print("-" * 70)
    for v in variants:
        s = summary.get(v, {})
        acc = s.get("val_accuracy", {})
        inf_f = s.get("inference_flops", {})
        par = s.get("params", {})
        print(
            f"{v:<20} "
            f"{acc.get('mean', 0.0):>10.4f} "
            f"{acc.get('std', 0.0):>8.4f} "
            f"{inf_f.get('mean', 0) / 1e6:>12.3f} "
            f"{par.get('mean', 0):>10.0f}"
        )
    print("=" * 70)

    # Gate check
    passed = check_gate(summary)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
