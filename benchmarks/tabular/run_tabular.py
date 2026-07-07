"""
Tabular benchmark runner ("Table 1").

Trains MLP, ResNet-MLP, FT-Transformer, TabNet, and SAINT baselines on a
tabular dataset (Adult Income, California Housing, Forest Cover Type, or
HIGGS) under identical conditions. MLP is a single fixed-size reference
(deliberately not swept); the other four sweep model-size presets where a
dataset is large enough to warrant it (l/xl on Covertype/Higgs, for a
Pareto accuracy-vs-params comparison) and use a single fixed preset
elsewhere (Adult, California Housing) — see configs/*.yaml.

Handles both task types (config.task): classification (Adult, Covertype,
Higgs — cross-entropy, accuracy) and regression (California Housing — MSE
on a standardized target, metrics reported after unstandardizing back to
the target's original units).

Parallelism: (model, preset) combinations train concurrently on the same
GPU (--max-parallel, default 4) — each is a separate process (spawn
context). Safe to re-run after a partial run/kill: completed combinations
are skipped before any process is spawned for them.

Usage:
    python benchmarks/tabular/run_tabular.py --dataset higgs \
        --config benchmarks/tabular/configs/higgs.yaml \
        --results-dir benchmarks/tabular/results/higgs
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.tabular.data.adult import get_adult_loaders
from benchmarks.tabular.data.california_housing import get_california_housing_loaders
from benchmarks.tabular.data.covertype import get_covertype_loaders
from benchmarks.tabular.data.higgs import get_higgs_loaders
from benchmarks.tabular.data.fashion_mnist import get_fashion_mnist_loaders
from benchmarks.tabular.models.mlp import MLP
from benchmarks.tabular.models.resnet_mlp import ResNetMLP
from benchmarks.tabular.models.ft_transformer import FTTransformer
from benchmarks.tabular.models.tabnet import TabNet
from benchmarks.tabular.models.saint import SAINT

logger = logging.getLogger("run_tabular")

ALL_MODELS = ["mlp", "resnet_mlp", "ft_transformer", "tabnet", "saint"]
ALL_DATASETS = ["adult", "california_housing", "covertype", "higgs", "fashion_mnist"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def build_model(model_name: str, preset_cfg: dict, num_numeric: int, cat_cardinalities: list, num_out: int) -> nn.Module:
    if model_name == "mlp":
        return MLP(num_numeric, cat_cardinalities, num_out, **preset_cfg)
    elif model_name == "resnet_mlp":
        return ResNetMLP(num_numeric, cat_cardinalities, num_out, **preset_cfg)
    elif model_name == "ft_transformer":
        return FTTransformer(num_numeric, cat_cardinalities, num_out, **preset_cfg)
    elif model_name == "tabnet":
        return TabNet(num_numeric, cat_cardinalities, num_out, **preset_cfg)
    elif model_name == "saint":
        return SAINT(num_numeric, cat_cardinalities, num_out, **preset_cfg)
    raise ValueError(f"Unknown model {model_name}")


def get_loaders(dataset_name: str, data_cfg: dict, batch_size: int, seed: int) -> dict:
    if dataset_name == "adult":
        return get_adult_loaders(
            batch_size=batch_size, root=data_cfg["root"], num_workers=data_cfg.get("num_workers", 0),
            val_fraction=data_cfg.get("val_fraction", 0.15), seed=seed,
        )
    elif dataset_name == "california_housing":
        return get_california_housing_loaders(
            batch_size=batch_size, root=data_cfg["root"], num_workers=data_cfg.get("num_workers", 0),
            val_fraction=data_cfg.get("val_fraction", 0.1), test_fraction=data_cfg.get("test_fraction", 0.2),
            seed=seed,
        )
    elif dataset_name == "covertype":
        return get_covertype_loaders(
            batch_size=batch_size, root=data_cfg["root"], num_workers=data_cfg.get("num_workers", 0),
            val_fraction=data_cfg.get("val_fraction", 0.1), test_fraction=data_cfg.get("test_fraction", 0.1),
            seed=seed,
        )
    elif dataset_name == "higgs":
        return get_higgs_loaders(
            batch_size=batch_size, root=data_cfg["root"], num_workers=data_cfg.get("num_workers", 0),
            num_train_rows=data_cfg.get("num_train_rows", 1_000_000),
            val_fraction=data_cfg.get("val_fraction", 0.05), seed=seed,
        )
    elif dataset_name == "fashion_mnist":
        return get_fashion_mnist_loaders(
            batch_size=batch_size, root=data_cfg["root"], num_workers=data_cfg.get("num_workers", 0),
            val_fraction=data_cfg.get("val_fraction", 0.1), seed=seed,
        )
    raise ValueError(f"Unknown tabular dataset {dataset_name}")


class TabularRunner:
    """Shared train/eval loop for both task types.

    Classification: cross-entropy loss, accuracy is the model-selection and
    reported metric.
    Regression: MSE loss on a standardized target; reported metrics
    (MSE/MAE/RMSE) are computed after unstandardizing predictions back to
    the target's original units (y_mean/y_std from the data loader).
    """

    def __init__(self, model, loaders: dict, run_dir, train_cfg, device, task: str):
        self.model = model.to(device)
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]
        self.test_loader = loaders["test"]
        self.task = task
        self.y_mean = loaders.get("y_mean")
        self.y_std = loaders.get("y_std")
        self.run_dir = run_dir
        self.device = device
        self.max_epochs = train_cfg["max_epochs"]
        self.grad_clip = train_cfg.get("grad_clip", 0.0)

        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)

        lr = train_cfg["learning_rate"]
        wd = train_cfg.get("weight_decay", 0.0)
        warmup = train_cfg.get("warmup_steps", 0)
        min_lr = train_cfg.get("min_lr", 1e-5)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        total_steps = max(1, self.max_epochs * len(self.train_loader))

        def lr_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return max(min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        if task == "classification":
            self.best_val_metric = 0.0  # accuracy, higher is better
        else:
            self.best_val_metric = float("inf")  # MSE, lower is better
        self.best_test_metrics = {}

    def _loss_and_metrics(self, x_num, x_cat, y):
        logits = self.model(x_num, x_cat)
        if self.task == "classification":
            loss = F.cross_entropy(logits, y)
        else:
            loss = F.mse_loss(logits.squeeze(-1), y)
        return logits, loss

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, n = 0.0, 0
        correct = 0
        se_sum, ae_sum = 0.0, 0.0  # regression: squared/absolute error in ORIGINAL units
        with torch.set_grad_enabled(train):
            for x_num, x_cat, y in loader:
                x_num, x_cat, y = x_num.to(self.device), x_cat.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits, loss = self._loss_and_metrics(x_num, x_cat, y)
                    total = loss
                    if hasattr(self.model, "last_sparsity_loss") and train:
                        total = total + self.model.last_sparsity_loss
                if train:
                    self.scaler.scale(total).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                bsz = x_num.size(0)
                total_loss += loss.item() * bsz
                n += bsz
                if self.task == "classification":
                    correct += (logits.argmax(-1) == y).sum().item()
                else:
                    pred = logits.squeeze(-1).float() * self.y_std + self.y_mean
                    target = y.float() * self.y_std + self.y_mean
                    se_sum += ((pred - target) ** 2).sum().item()
                    ae_sum += (pred - target).abs().sum().item()

        avg_loss = total_loss / max(n, 1)
        if self.task == "classification":
            return avg_loss, {"acc": correct / max(n, 1)}
        return avg_loss, {"mse": se_sum / max(n, 1), "mae": ae_sum / max(n, 1), "rmse": math.sqrt(se_sum / max(n, 1))}

    def run(self) -> dict:
        metrics_path = self.run_dir / "metrics.jsonl"
        for epoch in range(self.max_epochs):
            train_loss, train_m = self._run_epoch(self.train_loader, train=True)
            val_loss, val_m = self._run_epoch(self.val_loader, train=False)

            if self.task == "classification":
                val_metric, is_better = val_m["acc"], val_m["acc"] > self.best_val_metric
            else:
                val_metric, is_better = val_m["mse"], val_m["mse"] < self.best_val_metric

            if is_better:
                self.best_val_metric = val_metric
                _, self.best_test_metrics = self._run_epoch(self.test_loader, train=False)

            m = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_m.items()}}
            _write_jsonl(metrics_path, m)
            logger.info(f"  epoch {epoch + 1}/{self.max_epochs} " + " ".join(f"{k}={v:.4f}" for k, v in val_m.items()))

        final = {
            "best_val_metric": self.best_val_metric,
            "params": sum(p.numel() for p in self.model.parameters()),
            **{f"test_{k}": v for k, v in self.best_test_metrics.items()},
        }
        with open(self.run_dir / "final.json", "w") as f:
            json.dump(final, f, indent=2)
        return final


def run_one(model_name: str, preset_name: str, seed: int, cfg: dict, dataset_name: str, results_root: Path, device) -> dict:
    run_dir = results_root / model_name / preset_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "final.json").exists():
        logger.info(f"  [SKIP] {model_name}/{preset_name}/seed{seed} already complete")
        with open(run_dir / "final.json") as f:
            return json.load(f)

    set_seed(seed)
    logger.info(f">>> {model_name}/{preset_name} seed={seed} dir={run_dir}")

    task = cfg["task"]
    loaders = get_loaders(dataset_name, cfg["data"], cfg["training"]["batch_size"], seed)
    num_out = loaders["num_classes"] if task == "classification" else 1

    preset_cfg = cfg["models"][model_name]["presets"][preset_name]
    model = build_model(model_name, preset_cfg, loaders["num_numeric"], loaders["cat_cardinalities"], num_out)
    runner = TabularRunner(model, loaders, run_dir, cfg["training"], device, task)
    return runner.run()


def _preset_fully_done(results_root: Path, model_name: str, preset_name: str, seeds) -> bool:
    return all((results_root / model_name / preset_name / f"seed{s}" / "final.json").exists() for s in seeds)


def _run_preset_worker(model_name: str, preset_name: str, seeds, cfg: dict, dataset_name: str, results_root_str: str, device_str: str) -> None:
    """Runs one (model, preset) combination's full seed grid, sequentially — see
    benchmarks/timeseries/run_forecasting.py's identical worker pattern."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] {model_name}/{preset_name}: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    device = torch.device(device_str)
    results_root = Path(results_root_str)
    for seed in seeds:
        run_one(model_name, preset_name, seed, cfg, dataset_name, results_root, device)


def aggregate_results(results_root: Path, task: str, models, presets_by_model, seeds) -> dict:
    summary = {}
    for model_name in models:
        for preset_name in presets_by_model[model_name]:
            key = f"{model_name}/{preset_name}"
            params_list = []
            metric_lists = {}
            for seed in seeds:
                final_path = results_root / model_name / preset_name / f"seed{seed}" / "final.json"
                if not final_path.exists():
                    continue
                with open(final_path) as f:
                    d = json.load(f)
                params_list.append(d.get("params", 0))
                for k, v in d.items():
                    if k.startswith("test_"):
                        metric_lists.setdefault(k, []).append(v)
            if not params_list:
                continue

            def _stats(lst):
                a = np.array(lst, dtype=float)
                return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "runs": len(lst)}

            summary[key] = {
                "model": model_name, "preset": preset_name,
                "params": _stats(params_list),
                **{k: _stats(v) for k, v in metric_lists.items()},
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Tabular benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=ALL_DATASETS)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["all"], choices=ALL_MODELS + ["all"])
    parser.add_argument("--presets", nargs="+", default=None,
                         help="Model-size presets to sweep (default: all presets in config)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-parallel", type=int, default=4,
                         help="Number of (model, preset) combinations to train concurrently on the GPU")
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

    presets_by_model = {
        m: (args.presets or list(cfg["models"][m]["presets"].keys())) for m in models
    }

    tasks = []
    for model_name in models:
        for preset_name in presets_by_model[model_name]:
            if preset_name not in cfg["models"][model_name]["presets"]:
                logger.info(f"  [SKIP] {model_name}/{preset_name} — not defined in config for this model")
                continue
            if _preset_fully_done(results_root, model_name, preset_name, seeds):
                logger.info(f"  [SKIP] {model_name}/{preset_name} — all seeds already complete")
                continue
            tasks.append((model_name, preset_name))

    if tasks:
        device_str = str(device)
        ctx = mp.get_context("spawn")  # required for CUDA safety across processes
        failed = []
        with ProcessPoolExecutor(max_workers=args.max_parallel, mp_context=ctx) as executor:
            futures = {
                executor.submit(_run_preset_worker, m, p, seeds, cfg, args.dataset, str(results_root), device_str): (m, p)
                for m, p in tasks
            }
            for future in as_completed(futures):
                m, p = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(f"FAILED {m}/{p}")
                    failed.append((m, p))
        if failed:
            logger.warning(f"{len(failed)} (model, preset) combination(s) failed: {failed}")

    # Only presets actually defined per model (avoids KeyError when a model
    # doesn't have every preset name, e.g. mlp's single "m" vs others' l/xl).
    valid_presets_by_model = {
        m: [p for p in presets_by_model[m] if p in cfg["models"][m]["presets"]] for m in models
    }
    summary = aggregate_results(results_root, cfg["task"], models, valid_presets_by_model, seeds)
    with open(results_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    rows = sorted(summary.values(), key=lambda r: r["params"]["mean"])
    metric_keys = sorted({k for r in rows for k in r if k.startswith("test_")})
    print("\n" + "=" * 90)
    print(f"TABULAR SUMMARY — {args.dataset}")
    header = f"{'MODEL':<16}{'PRESET':<8}{'PARAMS':>12}" + "".join(f"{k.upper():>14}" for k in metric_keys)
    print(header)
    print("-" * 90)
    for r in rows:
        row = f"{r['model']:<16}{r['preset']:<8}{int(r['params']['mean']):>12,}"
        row += "".join(f"{r[k]['mean']:>14.4f}" if k in r else " " * 14 for k in metric_keys)
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    main()
