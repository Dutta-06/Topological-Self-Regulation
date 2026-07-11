"""
TSR Time-Series Forecasting Benchmark Runner ("Table 3" TSR row).

Trains TSRTCNForecaster on a forecasting dataset with the full structural
plasticity loop. Writes final.json in the exact same schema as the baseline
runner so it merges into the existing summary.json.

Also trains a static_final control for the matched-param plasticity ablation.

Usage:
    python benchmarks/timeseries/run_tsr_forecasting.py --dataset etth1 \
        --config benchmarks/timeseries/configs/etth1.yaml \
        --results-dir benchmarks/timeseries/results/etth1
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.timeseries.data.etth import get_ett_loaders
from benchmarks.timeseries.data.electricity import get_electricity_loaders
from benchmarks.timeseries.data.weather import get_weather_loaders
from benchmarks.timeseries.models.tsr_tcn import TSRTCNForecaster
from tsr.regulation.phantom import PhantomManager
from tsr.regulation.monitor import StructuralPlasticityMonitor
from tsr.regulation.actions import apply_structural_update, hard_prune_dead_gates
from tsr.regulation.scheduler import StructuralUpdateScheduler
from tsr.regulation.signals import gate_sparsity_penalty
from tsr.flops import differentiable_effective_flops
from tsr.topology import capture_topology
from tsr.utils import count_parameters, count_effective_parameters, rebuild_optimizer

logger = logging.getLogger("run_tsr_forecasting")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def get_loaders(dataset_name: str, data_cfg: dict, batch_size: int, pred_len: int):
    if dataset_name in ("etth1", "etth2"):
        return get_ett_loaders(
            name=dataset_name, batch_size=batch_size,
            seq_len=data_cfg["seq_len"], pred_len=pred_len,
            root=data_cfg["root"], num_workers=data_cfg.get("num_workers", 0),
        )
    elif dataset_name == "electricity":
        return get_electricity_loaders(
            batch_size=batch_size, seq_len=data_cfg["seq_len"],
            pred_len=pred_len, root=data_cfg["root"],
            num_workers=data_cfg.get("num_workers", 0),
        )
    elif dataset_name == "weather":
        return get_weather_loaders(
            batch_size=batch_size, seq_len=data_cfg["seq_len"],
            pred_len=pred_len, root=data_cfg["root"],
            num_workers=data_cfg.get("num_workers", 0),
        )
    raise ValueError(f"Unknown forecasting dataset {dataset_name}")


def _unpack_loaders(loaders, batch_size):
    """Unpack loaders tuple and infer input_size from a sample batch."""
    train_loader, val_loader, test_loader = loaders
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "input_size": input_size,
    }


class TSRForecastingRunner:
    """TSR training loop for time-series forecasting with structural plasticity."""

    def __init__(
        self,
        model: TSRTCNForecaster,
        loaders: dict,
        run_dir: Path,
        cfg: dict,
        device: torch.device,
        pred_len: int,
    ):
        self.model = model.to(device)
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]
        self.run_dir = run_dir
        self.device = device
        self.pred_len = pred_len
        self.y_mean = loaders.get("y_mean")
        self.y_std = loaders.get("y_std")

        train_cfg = cfg["training"]
        reg_cfg = cfg.get("tsr", cfg.get("regulation", {}))

        self.max_epochs = train_cfg.get("max_epochs", 100)
        lr = train_cfg.get("learning_rate", 0.001)
        wd = train_cfg.get("weight_decay", 0.0001)
        warmup = train_cfg.get("warmup_steps", 500)
        min_lr = train_cfg.get("min_lr", 1e-5)

        self.gate_sparsity_coeff = reg_cfg.get("gate_sparsity_coeff", 1e-4)
        self.flops_price = reg_cfg.get("flops_price", 0.0)
        # (encoder_input_channels, seq_len) — read live from the model so it's
        # correct whether the encoder is channel-independent (1 channel) or joint.
        self.flops_input_shape = (model.encoder.blocks[0].conv1.in_channels, cfg["data"]["seq_len"])

        self.reg_config = {
            "death_threshold": reg_cfg.get("death_threshold", 0.01),
            "min_neurons": reg_cfg.get("min_neurons_per_layer", 4),
            "newborn_protect_steps": reg_cfg.get("newborn_protect_steps", 400),
            "growth_enabled": reg_cfg.get("growth_enabled", True),
            "growth_rate": reg_cfg.get("growth_rate", 0.2),
            "max_neurons": reg_cfg.get("max_neurons_per_layer", 512),
            "bottleneck_threshold": reg_cfg.get("bottleneck_threshold", 0.1),
            "newborn_gate_init": reg_cfg.get("newborn_gate_init", 0.0),
            "depth_adaptation_enabled": reg_cfg.get("layer_insertion_enabled", True),
            "layer_insertion_threshold": reg_cfg.get("layer_insertion_threshold", 2.0),
            "max_blocks": reg_cfg.get("max_blocks", 16),
            "growth_signal_mode": "phantom",
            "phantom_threshold": reg_cfg.get("phantom_threshold", 0.005),
        }

        self.phantom_manager = PhantomManager(
            model,
            k=reg_cfg.get("phantom_k", 8),
            window=reg_cfg.get("monitor_window", 100),
        ).to(device)

        self.optimizer_kwargs = {"lr": lr, "weight_decay": wd}
        self.gate_lr_mult = train_cfg.get("gate_lr_multiplier", 1.0)
        self.act_lr_mult = train_cfg.get("act_lr_multiplier", 0.1)
        self.lr = lr
        self.min_lr = min_lr
        self.warmup = warmup
        self.total_steps = self.max_epochs * len(self.train_loader)

        self.optimizer = rebuild_optimizer(
            model, torch.optim.Adam, self.optimizer_kwargs,
            gate_lr_multiplier=self.gate_lr_mult, act_lr_multiplier=self.act_lr_mult,
        )
        self._add_phantom_group()

        self.scheduler = self._build_scheduler()

        self.monitor = StructuralPlasticityMonitor(
            model, window=reg_cfg.get("monitor_window", 100)
        )
        self.structural_scheduler = StructuralUpdateScheduler(
            update_interval=reg_cfg.get("update_interval", 200),
            cooldown_steps=reg_cfg.get("cooldown_steps", 50),
            anneal_structural_rate=reg_cfg.get("anneal_structural_rate", False),
            anneal_start_step=reg_cfg.get("anneal_start_step", 5000),
            anneal_factor=reg_cfg.get("anneal_factor", 0.95),
        )

        self.step = 0
        self.events = []
        self.structural_updates_enabled = True
        self.best_val_mse = float("inf")
        self.best_test_metrics = {}

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
        params = [p for p in self.phantom_manager.parameters() if p.requires_grad]
        if params:
            self.optimizer.add_param_group({
                "params": params,
                "lr": self.optimizer_kwargs["lr"],
                "group_name": "phantom",
            })

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, n = 0.0, 0
        se_sum, ae_sum = 0.0, 0.0

        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                if train:
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    task_loss = F.mse_loss(pred, y)
                    loss = task_loss

                    if self.gate_sparsity_coeff > 0:
                        loss = loss + self.gate_sparsity_coeff * gate_sparsity_penalty(self.model)
                    if self.flops_price > 0:
                        loss = loss + self.flops_price * differentiable_effective_flops(
                            self.model, self.flops_input_shape
                        )
                    loss = loss + self.phantom_manager.aux_loss()

                    loss.backward()
                    self.phantom_manager.record_gradients()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.monitor.record_loss(task_loss.item())

                    if self.structural_updates_enabled and self.structural_scheduler.should_update(self.step):
                        new_events = apply_structural_update(
                            model=self.model,
                            monitor=self.monitor,
                            step=self.step,
                            phantom_manager=self.phantom_manager,
                            **self.reg_config,
                        )
                        if new_events:
                            self.events.extend(new_events)
                            if any(e.action == "insert_layer" for e in new_events):
                                self.monitor.refresh_hooks()
                            self.phantom_manager.refresh()

                            self.optimizer = rebuild_optimizer(
                                self.model, torch.optim.Adam, self.optimizer_kwargs,
                                old_optimizer=self.optimizer,
                                gate_lr_multiplier=self.gate_lr_mult,
                                act_lr_multiplier=self.act_lr_mult,
                            )
                            self._add_phantom_group()
                            for group in self.optimizer.param_groups:
                                group.setdefault("initial_lr", group["lr"])
                            self.scheduler = self._build_scheduler(last_epoch=self.step)

                            self.structural_scheduler.record_update(
                                self.step, [e.layer_name for e in new_events]
                            )
                        else:
                            self.structural_scheduler.record_update(self.step, [])

                    self.step += 1
                    pred_detached = pred.detach()
                else:
                    with torch.no_grad():
                        pred_detached = self.model(x).detach()

                bsz = x.size(0)
                total_loss += F.mse_loss(pred_detached, y).item() * bsz
                n += bsz

                # Unstandardize for metrics
                if self.y_mean is not None and self.y_std is not None:
                    pred_orig = pred_detached * self.y_std + self.y_mean
                    y_orig = y * self.y_std + self.y_mean
                else:
                    pred_orig = pred_detached
                    y_orig = y

                se_sum += ((pred_orig - y_orig) ** 2).sum().item()
                ae_sum += (pred_orig - y_orig).abs().sum().item()

        if n == 0:
            raise RuntimeError(
                "empty loader (0 samples) — silently returning a 0.0 metric here would "
                "look like a perfect score and get selected as 'best'; failing loudly instead. "
                "Check seq_len+pred_len against the split length (e.g. Electricity's "
                "_VAL_HOURS/_TEST_HOURS vs. a long horizon)."
            )
        avg_loss = total_loss / max(n, 1)
        mse = se_sum / max(n, 1)
        mae = ae_sum / max(n, 1)
        return avg_loss, {"mse": mse, "mae": mae, "rmse": math.sqrt(mse)}

    def _hard_prune_and_finetune(self, finetune_epochs: int = 10) -> int:
        """B5: physically remove dead-gate (sigmoid(gate) < 0.5) channels, then
        fine-tune (no structural updates) so survivors adjust. Returns the
        pre-prune param count ("params_grown"); self.model ends up at the
        post-prune size ("params_pruned")."""
        params_grown = count_parameters(self.model)
        min_neurons = self.reg_config.get("min_neurons", 4)
        events = hard_prune_dead_gates(self.model, step=self.step, min_neurons=min_neurons)
        if not events:
            return params_grown

        self.events.extend(events)
        self.phantom_manager.refresh()
        self.optimizer = rebuild_optimizer(
            self.model, torch.optim.Adam, self.optimizer_kwargs,
            old_optimizer=self.optimizer,
            gate_lr_multiplier=self.gate_lr_mult, act_lr_multiplier=self.act_lr_mult,
        )
        self._add_phantom_group()

        self.structural_updates_enabled = False
        for _ in tqdm(range(finetune_epochs), desc="fine-tune (post-prune)", unit="epoch"):
            self._run_epoch(self.train_loader, train=True)
            _, val_m = self._run_epoch(self.val_loader, train=False)
            is_better = val_m["mse"] < self.best_val_mse
            if is_better:
                self.best_val_mse = val_m["mse"]
                _, self.best_test_metrics = self._run_epoch(self.test_loader, train=False)

        return params_grown

    def run(self) -> dict:
        metrics_path = self.run_dir / "metrics.jsonl"
        tag = "/".join(self.run_dir.parts[-3:])
        pbar = tqdm(range(self.max_epochs), desc=tag, unit="epoch", dynamic_ncols=True)
        for epoch in pbar:
            train_loss, train_m = self._run_epoch(self.train_loader, train=True)
            val_loss, val_m = self._run_epoch(self.val_loader, train=False)

            is_better = val_m["mse"] < self.best_val_mse
            if is_better:
                self.best_val_mse = val_m["mse"]
                _, self.best_test_metrics = self._run_epoch(self.test_loader, train=False)

            m = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mse": val_m["mse"],
                "val_mae": val_m["mae"],
                "params": count_parameters(self.model),
                "num_structural_events": len(self.events),
            }
            _write_jsonl(metrics_path, m)
            pbar.set_postfix(
                val_mse=f"{val_m['mse']:.4f}", val_mae=f"{val_m['mae']:.4f}",
                params=f"{count_parameters(self.model):,}", events=len(self.events),
            )

        params_grown = self._hard_prune_and_finetune()

        final = {
            "best_val_mse": self.best_val_mse,
            "params": count_parameters(self.model),
            "params_grown": params_grown,
            "params_pruned": count_parameters(self.model),
            "effective_params": count_effective_parameters(self.model),
            "num_structural_events": len(self.events),
            "final_topology": self.model.topology_summary(),
            "topology_state": self.model.topology_state(),
            **{f"test_{k}": v for k, v in self.best_test_metrics.items()},
        }
        with open(self.run_dir / "final.json", "w") as f:
            json.dump(final, f, indent=2)
        return final


class StaticFinalForecastingRunner:
    """Trains a TSRTCNForecaster at a fixed discovered topology (no growth)."""

    def __init__(
        self,
        model: TSRTCNForecaster,
        loaders: dict,
        run_dir: Path,
        cfg: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]
        self.run_dir = run_dir
        self.device = device
        self.y_mean = loaders.get("y_mean")
        self.y_std = loaders.get("y_std")

        train_cfg = cfg["training"]
        self.max_epochs = train_cfg.get("max_epochs", 100)

        lr = train_cfg.get("learning_rate", 0.001)
        wd = train_cfg.get("weight_decay", 0.0001)
        warmup = train_cfg.get("warmup_steps", 500)
        min_lr = train_cfg.get("min_lr", 1e-5)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        total_steps = self.max_epochs * len(self.train_loader)

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.best_val_mse = float("inf")
        self.best_test_metrics = {}

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, n = 0.0, 0
        se_sum, ae_sum = 0.0, 0.0

        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad()

                pred = self.model(x)
                loss = F.mse_loss(pred, y)

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                bsz = x.size(0)
                total_loss += loss.item() * bsz
                n += bsz

                if self.y_mean is not None and self.y_std is not None:
                    pred_orig = pred.detach() * self.y_std + self.y_mean
                    y_orig = y * self.y_std + self.y_mean
                else:
                    pred_orig = pred.detach()
                    y_orig = y

                se_sum += ((pred_orig - y_orig) ** 2).sum().item()
                ae_sum += (pred_orig - y_orig).abs().sum().item()

        if n == 0:
            raise RuntimeError(
                "empty loader (0 samples) — silently returning a 0.0 metric here would "
                "look like a perfect score and get selected as 'best'; failing loudly instead. "
                "Check seq_len+pred_len against the split length (e.g. Electricity's "
                "_VAL_HOURS/_TEST_HOURS vs. a long horizon)."
            )
        avg_loss = total_loss / max(n, 1)
        mse = se_sum / max(n, 1)
        mae = ae_sum / max(n, 1)
        return avg_loss, {"mse": mse, "mae": mae, "rmse": math.sqrt(mse)}

    def run(self) -> dict:
        tag = "/".join(self.run_dir.parts[-3:])
        pbar = tqdm(range(self.max_epochs), desc=f"[static] {tag}", unit="epoch", dynamic_ncols=True)
        for epoch in pbar:
            self._run_epoch(self.train_loader, train=True)
            val_loss, val_m = self._run_epoch(self.val_loader, train=False)

            is_better = val_m["mse"] < self.best_val_mse
            if is_better:
                self.best_val_mse = val_m["mse"]
                _, self.best_test_metrics = self._run_epoch(self.test_loader, train=False)

            pbar.set_postfix(val_mse=f"{val_m['mse']:.4f}", val_mae=f"{val_m['mae']:.4f}")

        final = {
            "best_val_mse": self.best_val_mse,
            "params": count_parameters(self.model),
            **{f"test_{k}": v for k, v in self.best_test_metrics.items()},
        }
        with open(self.run_dir / "final.json", "w") as f:
            json.dump(final, f, indent=2)
        return final


def run_tsr_forecast(
    seed: int,
    cfg: dict,
    dataset_name: str,
    pred_len: int,
    results_root: Path,
    device: torch.device,
) -> dict:
    run_dir = results_root / "tsr" / f"pred{pred_len}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "final.json").exists():
        logger.info(f"  [SKIP] tsr/pred{pred_len}/seed{seed} already complete")
        with open(run_dir / "final.json") as f:
            return json.load(f)

    set_seed(seed)
    logger.info(f">>> tsr pred_len={pred_len} seed={seed} dir={run_dir}")

    raw_loaders = get_loaders(dataset_name, cfg["data"], cfg["training"]["batch_size"], pred_len)
    loaders = _unpack_loaders(raw_loaders, cfg["training"]["batch_size"])
    input_size = loaders["input_size"]
    seq_len = cfg["data"]["seq_len"]

    tsr_cfg = cfg.get("tsr", {})
    model = TSRTCNForecaster(
        input_size=input_size,
        pred_len=pred_len,
        seed_channels=tsr_cfg.get("seed_channels", 32),
        num_levels=tsr_cfg.get("num_levels", 2),
        kernel_size=tsr_cfg.get("kernel_size", 3),
        gate_init=tsr_cfg.get("gate_init", 3.0),
        act_init=tsr_cfg.get("act_init", "relu"),
        norm_group_size=tsr_cfg.get("norm_group_size", 8),
        channel_independent=tsr_cfg.get("channel_independent", True),
    )
    runner = TSRForecastingRunner(model, loaders, run_dir, cfg, device, pred_len)
    return runner.run()


def run_static_final_forecast(
    seed: int,
    cfg: int,
    dataset_name: str,
    pred_len: int,
    results_root: Path,
    device: torch.device,
    block_conv_widths: List[tuple],
    channel_independent: bool = True,
) -> dict:
    run_dir = results_root / "tsr_static_final" / f"pred{pred_len}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "final.json").exists():
        logger.info(f"  [SKIP] tsr_static_final/pred{pred_len}/seed{seed} already complete")
        with open(run_dir / "final.json") as f:
            return json.load(f)

    set_seed(seed)
    logger.info(f">>> tsr_static_final pred_len={pred_len} seed={seed} widths={block_conv_widths} dir={run_dir}")

    raw_loaders = get_loaders(dataset_name, cfg["data"], cfg["training"]["batch_size"], pred_len)
    loaders = _unpack_loaders(raw_loaders, cfg["training"]["batch_size"])
    input_size = loaders["input_size"]

    tsr_cfg = cfg.get("tsr", {})
    model = TSRTCNForecaster(
        input_size=input_size,
        pred_len=pred_len,
        kernel_size=tsr_cfg.get("kernel_size", 3),
        gate_init=tsr_cfg.get("gate_init", 3.0),
        act_init=tsr_cfg.get("act_init", "relu"),
        norm_group_size=tsr_cfg.get("norm_group_size", 8),
        block_conv_widths=block_conv_widths,
        channel_independent=channel_independent,
    )
    runner = StaticFinalForecastingRunner(model, loaders, run_dir, cfg, device)
    return runner.run()


def _run_forecast_worker(
    variant: str,
    seed: int,
    cfg: dict,
    dataset_name: str,
    pred_len: int,
    results_root_str: str,
    device_str: str,
    block_conv_widths: Optional[List[tuple]] = None,
    channel_independent: Optional[bool] = None,
) -> dict:
    """Worker function for parallel execution."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] {variant}/pred{pred_len}/seed{seed}: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    device = torch.device(device_str)
    results_root = Path(results_root_str)

    if variant == "tsr":
        return run_tsr_forecast(seed, cfg, dataset_name, pred_len, results_root, device)
    elif variant == "tsr_static_final":
        tsr_cfg = cfg.get("tsr", {})
        if block_conv_widths is None:
            ch = tsr_cfg.get("seed_channels", 32)
            block_conv_widths = [(ch, ch)] * tsr_cfg.get("num_levels", 2)
        if channel_independent is None:
            channel_independent = tsr_cfg.get("channel_independent", True)
        return run_static_final_forecast(
            seed, cfg, dataset_name, pred_len, results_root, device,
            block_conv_widths=block_conv_widths,
            channel_independent=channel_independent,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


def aggregate_results(results_root: Path, seeds: List[int], horizons: List[int]) -> dict:
    summary = {}
    for variant in ["tsr", "tsr_static_final"]:
        for horizon in horizons:
            params_list = []
            mse_list = []
            mae_list = []
            for seed in seeds:
                final_path = results_root / variant / f"pred{horizon}" / f"seed{seed}" / "final.json"
                if not final_path.exists():
                    continue
                with open(final_path) as f:
                    d = json.load(f)
                params_list.append(d.get("params", 0))
                mse_list.append(d.get("test_mse", d.get("best_val_mse", float("inf"))))
                mae_list.append(d.get("test_mae", float("inf")))

            if not params_list:
                continue

            key = f"{variant}/pred{horizon}"
            summary[key] = {
                "variant": variant,
                "horizon": horizon,
                "params": {"mean": float(np.mean(params_list)), "std": float(np.std(params_list))},
                "test_mse": {"mean": float(np.mean(mse_list)), "std": float(np.std(mse_list))},
                "test_mae": {"mean": float(np.mean(mae_list)), "std": float(np.std(mae_list))},
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="TSR Time-series Forecasting benchmark")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["etth1", "etth2", "electricity", "weather"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=None,
                        help="Forecast horizons (default: from config)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--skip-static-final", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["training"]["max_epochs"] = args.epochs

    seeds = args.seeds or cfg["experiment"]["seeds"]
    horizons = args.horizons or cfg["experiment"]["horizons"]
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    logger.info(f"Device: {device}")

    # Phase 1: Run TSR for each horizon
    for horizon in horizons:
        logger.info(f"\n{'='*60}\nHorizon: {horizon}\n{'='*60}")

        tsr_results = {}
        tasks = []
        for seed in seeds:
            tsr_dir = results_root / "tsr" / f"pred{horizon}" / f"seed{seed}"
            if (tsr_dir / "final.json").exists():
                logger.info(f"  [SKIP] tsr/pred{horizon}/seed{seed} already complete")
                with open(tsr_dir / "final.json") as f:
                    tsr_results[seed] = json.load(f)
            else:
                tasks.append(("tsr", seed))

        if tasks:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.max_parallel, mp_context=ctx) as executor:
                futures = {
                    executor.submit(
                        _run_forecast_worker, variant, seed, cfg, args.dataset,
                        horizon, str(results_root), str(device)
                    ): (variant, seed)
                    for variant, seed in tasks
                }
                for future in as_completed(futures):
                    variant, seed = futures[future]
                    try:
                        result = future.result()
                        tsr_results[seed] = result
                    except Exception:
                        logger.exception(f"FAILED {variant}/pred{horizon}/seed{seed}")

        # Phase 2: Run static_final ablation
        if not args.skip_static_final:
            static_tasks = []
            for seed in seeds:
                static_dir = results_root / "tsr_static_final" / f"pred{horizon}" / f"seed{seed}"
                if (static_dir / "final.json").exists():
                    logger.info(f"  [SKIP] tsr_static_final/pred{horizon}/seed{seed} already complete")
                    continue

                # Get discovered topology from TSR run
                tsr_cfg = cfg.get("tsr", {})
                if seed in tsr_results:
                    topo = tsr_results[seed].get("topology_state", {})
                    block_conv_widths = topo.get("block_conv_widths")
                    channel_independent = topo.get("channel_independent", tsr_cfg.get("channel_independent", True))
                    if block_conv_widths is None:
                        ch = tsr_cfg.get("seed_channels", 32)
                        block_conv_widths = [(ch, ch)] * tsr_cfg.get("num_levels", 2)
                    else:
                        block_conv_widths = [tuple(w) for w in block_conv_widths]
                else:
                    ch = tsr_cfg.get("seed_channels", 32)
                    block_conv_widths = [(ch, ch)] * tsr_cfg.get("num_levels", 2)
                    channel_independent = tsr_cfg.get("channel_independent", True)

                static_tasks.append(("tsr_static_final", seed, block_conv_widths, channel_independent))

            if static_tasks:
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=args.max_parallel, mp_context=ctx) as executor:
                    futures = {
                        executor.submit(
                            _run_forecast_worker, variant, seed, cfg, args.dataset,
                            horizon, str(results_root), str(device),
                            block_conv_widths=block_conv_widths,
                            channel_independent=channel_independent,
                        ): (variant, seed)
                        for variant, seed, block_conv_widths, channel_independent in static_tasks
                    }
                    for future in as_completed(futures):
                        variant, seed = futures[future]
                        try:
                            future.result()
                        except Exception:
                            logger.exception(f"FAILED {variant}/pred{horizon}/seed{seed}")

    # Aggregate
    summary = aggregate_results(results_root, seeds, horizons)
    summary_path = results_root / "tsr_forecasting_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")

    # Print table
    print("\n" + "=" * 80)
    print(f"TSR FORECASTING SUMMARY — {args.dataset}")
    print(f"{'VARIANT':<20}{'HORIZON':>10}{'PARAMS':>12}{'TEST_MSE':>14}{'TEST_MAE':>14}")
    print("-" * 80)
    for key in sorted(summary.keys()):
        r = summary[key]
        print(
            f"{r['variant']:<20}{r['horizon']:>10}{int(r['params']['mean']):>12,}"
            f"{r['test_mse']['mean']:>14.4f}{r['test_mae']['mean']:>14.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
