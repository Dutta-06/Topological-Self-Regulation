"""
Time-series classification benchmark runner.

Trains LSTM, GRU, TCN, PatchTST, and Mamba baselines on a classification
dataset (HAR, or one/more UCR/UEA archive datasets) under identical
conditions, across multiple seeds and multiple model-size presets (for a
Pareto accuracy-vs-params comparison rather than one arbitrary size per
model).

Usage:
    python benchmarks/timeseries/run_classification.py --dataset har \
        --config benchmarks/timeseries/configs/har.yaml \
        --results-dir benchmarks/timeseries/results/har

    python benchmarks/timeseries/run_classification.py --dataset ucr_uea \
        --config benchmarks/timeseries/configs/ucr_uea.yaml \
        --results-dir benchmarks/timeseries/results/ucr_uea
    # (loops over every dataset name in config.data.datasets)
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

from benchmarks.timeseries.data.har import get_har_loaders
from benchmarks.timeseries.data.ucr_uea import get_ucr_uea_loaders
from benchmarks.timeseries.models.rnn import RNNClassifier
from benchmarks.timeseries.models.tcn import TCNClassifier
from benchmarks.timeseries.models.patchtst import PatchTSTClassifier
from benchmarks.timeseries.models.mamba import MambaClassifier

logger = logging.getLogger("run_classification")

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


def build_model(model_name: str, preset_cfg: dict, input_size: int, seq_len: int, num_classes: int) -> nn.Module:
    if model_name == "lstm":
        return RNNClassifier(input_size, num_classes, cell_type="lstm", **preset_cfg)
    elif model_name == "gru":
        return RNNClassifier(input_size, num_classes, cell_type="gru", **preset_cfg)
    elif model_name == "tcn":
        return TCNClassifier(input_size, num_classes, **preset_cfg)
    elif model_name == "patchtst":
        return PatchTSTClassifier(input_size, seq_len=seq_len, num_classes=num_classes, **preset_cfg)
    elif model_name == "mamba":
        return MambaClassifier(input_size, num_classes, **preset_cfg)
    raise ValueError(f"Unknown model {model_name}")


def get_loaders(dataset_name: str, data_cfg: dict, batch_size: int, seed: int):
    if dataset_name == "har":
        return get_har_loaders(
            batch_size=batch_size, root=data_cfg["root"],
            num_workers=data_cfg.get("num_workers", 0),
            val_fraction=data_cfg.get("val_fraction", 0.15), seed=seed,
        )
    # UCR/UEA: dataset_name here is a specific archive dataset name (e.g. "ECG200")
    train_loader, val_loader, test_loader, seq_len, num_channels, num_classes = get_ucr_uea_loaders(
        dataset_name, batch_size=batch_size, root=data_cfg["root"],
        num_workers=data_cfg.get("num_workers", 0),
        val_fraction=data_cfg.get("val_fraction", 0.2), seed=seed,
    )
    return train_loader, val_loader, test_loader, num_channels, num_classes


class ClassifyRunner:
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
        self.best_val_acc = 0.0
        self.best_test_acc = 0.0

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                if train:
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                total_loss += loss.item() * x.size(0)
                correct += (logits.argmax(-1) == y).sum().item()
                n += x.size(0)
        return total_loss / max(n, 1), correct / max(n, 1)

    def run(self) -> dict:
        metrics_path = self.run_dir / "metrics.jsonl"
        for epoch in range(self.max_epochs):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                _, self.best_test_acc = self._run_epoch(self.test_loader, train=False)
            m = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc}
            _write_jsonl(metrics_path, m)
            logger.info(
                f"  epoch {epoch + 1}/{self.max_epochs} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )

        final = {
            "best_val_acc": self.best_val_acc,
            "test_acc": self.best_test_acc,
            "params": sum(p.numel() for p in self.model.parameters()),
        }
        with open(self.run_dir / "final.json", "w") as f:
            json.dump(final, f, indent=2)
        return final


def run_one(
    model_name: str, preset_name: str, seed: int, cfg: dict, dataset_name: str,
    results_root: Path, device,
) -> dict:
    run_dir = results_root / dataset_name / model_name / preset_name / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "final.json").exists():
        logger.info(f"  [SKIP] {dataset_name}/{model_name}/{preset_name}/seed{seed} already complete")
        with open(run_dir / "final.json") as f:
            return json.load(f)

    set_seed(seed)
    logger.info(f">>> {dataset_name}/{model_name}/{preset_name} seed={seed} dir={run_dir}")

    train_loader, val_loader, test_loader, num_channels, num_classes = get_loaders(
        dataset_name, cfg["data"], cfg["training"]["batch_size"], seed
    )
    sample_x, _ = next(iter(train_loader))
    seq_len = sample_x.shape[1]

    preset_cfg = cfg["models"][model_name]["presets"][preset_name]
    model = build_model(model_name, preset_cfg, num_channels, seq_len, num_classes)
    runner = ClassifyRunner(model, train_loader, val_loader, test_loader, run_dir, cfg["training"], device)
    return runner.run()


def aggregate_results(results_root: Path, dataset_name: str, models, presets, seeds) -> dict:
    summary = {}
    for model_name in models:
        for preset_name in presets:
            key = f"{model_name}/{preset_name}"
            test_accs, val_accs, params_list = [], [], []
            for seed in seeds:
                final_path = results_root / dataset_name / model_name / preset_name / f"seed{seed}" / "final.json"
                if not final_path.exists():
                    continue
                with open(final_path) as f:
                    d = json.load(f)
                test_accs.append(d.get("test_acc", float("nan")))
                val_accs.append(d.get("best_val_acc", float("nan")))
                params_list.append(d.get("params", 0))
            if not test_accs:
                continue

            def _stats(lst):
                a = np.array(lst, dtype=float)
                return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "runs": len(lst)}

            summary[key] = {
                "model": model_name,
                "preset": preset_name,
                "test_acc": _stats(test_accs),
                "val_acc": _stats(val_accs),
                "params": _stats(params_list),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Time-series classification benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=["har", "ucr_uea"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["all"], choices=ALL_MODELS + ["all"])
    parser.add_argument("--presets", nargs="+", default=None,
                         help="Model-size presets to sweep (default: all presets in config, for a Pareto curve)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ucr-datasets", nargs="+", default=None,
                         help="Override the UCR/UEA dataset list from config.data.datasets")
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

    if args.dataset == "har":
        dataset_names = ["har"]
    else:
        dataset_names = args.ucr_datasets or cfg["data"]["datasets"]

    all_summaries = {}
    for dataset_name in dataset_names:
        for model_name in models:
            presets = args.presets or list(cfg["models"][model_name]["presets"].keys())
            for preset_name in presets:
                for seed in seeds:
                    run_one(model_name, preset_name, seed, cfg, dataset_name, results_root, device)

        all_presets = sorted({p for m in models for p in (args.presets or cfg["models"][m]["presets"].keys())})
        summary = aggregate_results(results_root, dataset_name, models, all_presets, seeds)
        all_summaries[dataset_name] = summary
        with open(results_root / dataset_name / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        rows = sorted(summary.values(), key=lambda r: r["params"]["mean"])
        print("\n" + "=" * 70)
        print(f"DATASET: {dataset_name}")
        print(f"{'MODEL':<10}{'PRESET':<8}{'PARAMS':>12}{'TEST_ACC':>12}{'±STD':>10}")
        print("-" * 70)
        for r in rows:
            print(
                f"{r['model']:<10}{r['preset']:<8}{int(r['params']['mean']):>12,}"
                f"{r['test_acc']['mean']:>12.4f}{r['test_acc']['std']:>10.4f}"
            )
        print("=" * 70)

    with open(results_root / "summary_all.json", "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
