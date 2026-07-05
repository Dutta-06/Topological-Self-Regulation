"""
Time-series forecasting benchmark runner.

Trains LSTM, GRU, TCN, PatchTST, and Mamba baselines on a forecasting dataset
(ETTh1, ETTh2, or Electricity) under identical conditions, across multiple
seeds. Mirrors scripts/gate_experiment.py's runner pattern for consistency
with the rest of the repo.

Usage:
    python benchmarks/timeseries/run_forecasting.py --dataset etth1 \
        --config benchmarks/timeseries/configs/etth1.yaml \
        --results-dir benchmarks/timeseries/results/etth1
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.timeseries.data.etth import get_ett_loaders
from benchmarks.timeseries.data.electricity import get_electricity_loaders
from benchmarks.timeseries.models.rnn import RNNForecaster
from benchmarks.timeseries.models.tcn import TCNForecaster
from benchmarks.timeseries.models.patchtst import PatchTSTForecaster
from benchmarks.timeseries.models.mamba import MambaForecaster

logger = logging.getLogger("run_forecasting")

ALL_MODELS = ["lstm", "gru", "tcn", "patchtst", "mamba"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def build_model(model_name: str, cfg: dict, input_size: int, seq_len: int) -> nn.Module:
    mcfg = cfg["models"][model_name]
    if model_name == "lstm":
        return RNNForecaster(input_size, cell_type="lstm", **mcfg)
    elif model_name == "gru":
        return RNNForecaster(input_size, cell_type="gru", **mcfg)
    elif model_name == "tcn":
        return TCNForecaster(input_size, **mcfg)
    elif model_name == "patchtst":
        return PatchTSTForecaster(input_size, seq_len=seq_len, **mcfg)
    elif model_name == "mamba":
        return MambaForecaster(input_size, **mcfg)
    raise ValueError(f"Unknown model {model_name}")


def get_loaders(dataset_name: str, data_cfg: dict, batch_size: int):
    if dataset_name in ("etth1", "etth2"):
        return get_ett_loaders(
            name=dataset_name, batch_size=batch_size,
            seq_len=data_cfg["seq_len"], pred_len=data_cfg["pred_len"],
            root=data_cfg["root"], num_workers=data_cfg.get("num_workers", 0),
        )
    elif dataset_name == "electricity":
        return get_electricity_loaders(
            batch_size=batch_size, seq_len=data_cfg["seq_len"],
            pred_len=data_cfg["pred_len"], root=data_cfg["root"],
            num_workers=data_cfg.get("num_workers", 0),
            num_clients=data_cfg.get("num_clients", 16),
        )
    raise ValueError(f"Unknown forecasting dataset {dataset_name}")


class ForecastRunner:
    def __init__(self, model, train_loader, val_loader, test_loader, run_dir, train_cfg, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.run_dir = run_dir
        self.device = device
        self.max_epochs = train_cfg["max_epochs"]
        self.grad_clip = train_cfg.get("grad_clip", 0.0)

        lr = train_cfg["learning_rate"]
        wd = train_cfg.get("weight_decay", 0.0)
        warmup = train_cfg.get("warmup_steps", 0)
        min_lr = train_cfg.get("min_lr", 1e-5)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        total_steps = max(1, self.max_epochs * len(train_loader))

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.best_val_mse = float("inf")
        self.best_test_mse = float("inf")

    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total_loss, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                pred = self.model(x)
                loss = F.mse_loss(pred, y)
                if train:
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
        return total_loss / max(n, 1)

    def run(self) -> dict:
        metrics_path = self.run_dir / "metrics.jsonl"
        for epoch in range(self.max_epochs):
            train_mse = self._run_epoch(self.train_loader, train=True)
            val_mse = self._run_epoch(self.val_loader, train=False)
            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                self.best_test_mse = self._run_epoch(self.test_loader, train=False)
            m = {"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse}
            _write_jsonl(metrics_path, m)
            logger.info(
                f"  epoch {epoch + 1}/{self.max_epochs} "
                f"train_mse={train_mse:.4f} val_mse={val_mse:.4f}"
            )

        final = {
            "best_val_mse": self.best_val_mse,
            "test_mse": self.best_test_mse,
            "params": sum(p.numel() for p in self.model.parameters()),
        }
        with open(self.run_dir / "final.json", "w") as f:
            json.dump(final, f, indent=2)
        return final


def run_one(model_name: str, seed: int, cfg: dict, dataset_name: str, results_root: Path, device) -> dict:
    run_dir = results_root / model_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "final.json").exists():
        logger.info(f"  [SKIP] {model_name}/seed{seed} already complete")
        with open(run_dir / "final.json") as f:
            return json.load(f)

    set_seed(seed)
    logger.info(f">>> {model_name} seed={seed} dir={run_dir}")

    train_loader, val_loader, test_loader = get_loaders(dataset_name, cfg["data"], cfg["training"]["batch_size"])
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    seq_len = sample_x.shape[1]

    model = build_model(model_name, cfg, input_size, seq_len)
    runner = ForecastRunner(model, train_loader, val_loader, test_loader, run_dir, cfg["training"], device)
    return runner.run()


def aggregate_results(results_root: Path, models, seeds) -> dict:
    summary = {}
    for model_name in models:
        test_mses, val_mses, params_list = [], [], []
        for seed in seeds:
            final_path = results_root / model_name / f"seed{seed}" / "final.json"
            if not final_path.exists():
                continue
            with open(final_path) as f:
                d = json.load(f)
            test_mses.append(d.get("test_mse", float("nan")))
            val_mses.append(d.get("best_val_mse", float("nan")))
            params_list.append(d.get("params", 0))
        if not test_mses:
            continue

        def _stats(lst):
            a = np.array(lst, dtype=float)
            return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "runs": len(lst)}

        summary[model_name] = {
            "test_mse": _stats(test_mses),
            "val_mse": _stats(val_mses),
            "params": _stats(params_list),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Time-series forecasting benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=["etth1", "etth2", "electricity"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["all"], choices=ALL_MODELS + ["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
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

    models = ALL_MODELS if "all" in args.models else args.models
    seeds = args.seeds or cfg["experiment"]["seeds"]

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    logger.info(f"Device: {device}")

    for model_name in models:
        for seed in seeds:
            run_one(model_name, seed, cfg, args.dataset, results_root, device)

    summary = aggregate_results(results_root, models, seeds)
    with open(results_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'MODEL':<12}{'TEST_MSE':>12}{'±STD':>10}{'PARAMS':>14}")
    print("-" * 70)
    for model_name, stats in summary.items():
        print(
            f"{model_name:<12}{stats['test_mse']['mean']:>12.4f}"
            f"{stats['test_mse']['std']:>10.4f}{int(stats['params']['mean']):>14,}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
