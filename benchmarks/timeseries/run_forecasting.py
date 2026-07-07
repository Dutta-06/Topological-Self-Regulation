"""
Time-series forecasting benchmark runner.

Trains LSTM, GRU, TCN, PatchTST, and Mamba baselines on a forecasting dataset
(ETTh1, ETTh2, or Electricity) under identical conditions, across multiple
seeds, multiple forecast horizons, and multiple model-size presets (for a
Pareto accuracy-vs-params comparison rather than one arbitrary size per
model). Mirrors scripts/gate_experiment.py's runner pattern for consistency
with the rest of the repo.

Protocol: standard multivariate long-horizon forecasting (Informer/Autoformer/
PatchTST convention) — given a seq_len window of ALL channels, predict the
next pred_len steps of ALL channels; report MSE and MAE averaged over the
full (pred_len, channels) target. Horizons default to the standard ETT set
{96, 192, 336, 720}.

Parallelism: (model, preset, horizon) combinations train concurrently on the
same GPU (--max-parallel, default 4) — each is a separate process (spawn
context) that runs its own seed grid sequentially. Safe to re-run after a
partial run/kill: already-completed (model, preset, horizon) triples are
skipped before any process is spawned for them, and individual seed runs
within an incomplete triple are skipped too (existing final.json).

Usage:
    python benchmarks/timeseries/run_forecasting.py --dataset etth1 \
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
from benchmarks.timeseries.data.weather import get_weather_loaders
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


def build_model(model_name: str, preset_cfg: dict, input_size: int, seq_len: int, pred_len: int) -> nn.Module:
    if model_name == "lstm":
        return RNNForecaster(input_size, pred_len, cell_type="lstm", **preset_cfg)
    elif model_name == "gru":
        return RNNForecaster(input_size, pred_len, cell_type="gru", **preset_cfg)
    elif model_name == "tcn":
        return TCNForecaster(input_size, pred_len, **preset_cfg)
    elif model_name == "patchtst":
        return PatchTSTForecaster(input_size, seq_len=seq_len, pred_len=pred_len, **preset_cfg)
    elif model_name == "mamba":
        return MambaForecaster(input_size, pred_len, **preset_cfg)
    raise ValueError(f"Unknown model {model_name}")


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
            num_clients=data_cfg.get("num_clients", 321),
        )
    elif dataset_name == "weather":
        return get_weather_loaders(
            batch_size=batch_size, seq_len=data_cfg["seq_len"],
            pred_len=pred_len, root=data_cfg["root"],
            num_workers=data_cfg.get("num_workers", 0),
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

        # Mixed precision: only meaningful on CUDA (uses Tensor Cores where
        # available). GradScaler(enabled=False) is a documented no-op, so this
        # is always safe to call even on CPU.
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.use_amp)

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
        self.best_test_mae = float("inf")

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_mse, total_mae, n = 0.0, 0.0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    pred = self.model(x)
                    loss = F.mse_loss(pred, y)
                if train:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                with torch.no_grad():
                    mae = F.l1_loss(pred.float(), y.float())
                total_mse += loss.item() * x.size(0)
                total_mae += mae.item() * x.size(0)
                n += x.size(0)
        return total_mse / max(n, 1), total_mae / max(n, 1)

    def run(self) -> dict:
        metrics_path = self.run_dir / "metrics.jsonl"
        for epoch in range(self.max_epochs):
            train_mse, train_mae = self._run_epoch(self.train_loader, train=True)
            val_mse, val_mae = self._run_epoch(self.val_loader, train=False)
            if val_mse < self.best_val_mse:
                self.best_val_mse = val_mse
                self.best_test_mse, self.best_test_mae = self._run_epoch(self.test_loader, train=False)
            m = {"epoch": epoch, "train_mse": train_mse, "train_mae": train_mae, "val_mse": val_mse}
            _write_jsonl(metrics_path, m)
            logger.info(
                f"  epoch {epoch + 1}/{self.max_epochs} "
                f"train_mse={train_mse:.4f} val_mse={val_mse:.4f}"
            )

        final = {
            "best_val_mse": self.best_val_mse,
            "test_mse": self.best_test_mse,
            "test_mae": self.best_test_mae,
            "params": sum(p.numel() for p in self.model.parameters()),
        }
        with open(self.run_dir / "final.json", "w") as f:
            json.dump(final, f, indent=2)
        return final


def run_one(
    model_name: str, preset_name: str, pred_len: int, seed: int, cfg: dict,
    dataset_name: str, results_root: Path, device,
) -> dict:
    run_dir = results_root / model_name / preset_name / f"h{pred_len}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "final.json").exists():
        logger.info(f"  [SKIP] {model_name}/{preset_name}/h{pred_len}/seed{seed} already complete")
        with open(run_dir / "final.json") as f:
            return json.load(f)

    set_seed(seed)
    logger.info(f">>> {model_name}/{preset_name} h={pred_len} seed={seed} dir={run_dir}")

    train_loader, val_loader, test_loader = get_loaders(
        dataset_name, cfg["data"], cfg["training"]["batch_size"], pred_len
    )
    sample_x, _ = next(iter(train_loader))
    input_size = sample_x.shape[-1]
    seq_len = sample_x.shape[1]

    preset_cfg = cfg["models"][model_name]["presets"][preset_name]
    model = build_model(model_name, preset_cfg, input_size, seq_len, pred_len)
    runner = ForecastRunner(model, train_loader, val_loader, test_loader, run_dir, cfg["training"], device)
    return runner.run()


def _task_fully_done(results_root: Path, model_name: str, preset_name: str, pred_len: int, seeds) -> bool:
    return all(
        (results_root / model_name / preset_name / f"h{pred_len}" / f"seed{s}" / "final.json").exists()
        for s in seeds
    )


def _run_task_worker(
    model_name: str, preset_name: str, pred_len: int, seeds, cfg: dict,
    dataset_name: str, results_root_str: str, device_str: str,
) -> None:
    """Runs one (model, preset, horizon) combination's full seed grid, sequentially.

    This is the unit of work dispatched to a separate process so multiple
    (model, preset, horizon) combinations train concurrently on the same GPU —
    each is a fresh Python interpreter (spawn context, required for CUDA) with
    its own CUDA context; the driver time-slices them across the same physical
    device. None of our models saturate a modern GPU alone, so this gets real
    speedup rather than just serializing through a lock.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] {model_name}/{preset_name}/h{pred_len}: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    device = torch.device(device_str)
    results_root = Path(results_root_str)
    for seed in seeds:
        run_one(model_name, preset_name, pred_len, seed, cfg, dataset_name, results_root, device)


def aggregate_results(results_root: Path, models, presets, pred_lens, seeds) -> dict:
    summary = {}
    for model_name in models:
        for preset_name in presets:
            for pred_len in pred_lens:
                key = f"{model_name}/{preset_name}/h{pred_len}"
                test_mses, test_maes, val_mses, params_list = [], [], [], []
                for seed in seeds:
                    final_path = results_root / model_name / preset_name / f"h{pred_len}" / f"seed{seed}" / "final.json"
                    if not final_path.exists():
                        continue
                    with open(final_path) as f:
                        d = json.load(f)
                    test_mses.append(d.get("test_mse", float("nan")))
                    test_maes.append(d.get("test_mae", float("nan")))
                    val_mses.append(d.get("best_val_mse", float("nan")))
                    params_list.append(d.get("params", 0))
                if not test_mses:
                    continue

                def _stats(lst):
                    a = np.array(lst, dtype=float)
                    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a)), "runs": len(lst)}

                summary[key] = {
                    "model": model_name,
                    "preset": preset_name,
                    "pred_len": pred_len,
                    "test_mse": _stats(test_mses),
                    "test_mae": _stats(test_maes),
                    "val_mse": _stats(val_mses),
                    "params": _stats(params_list),
                }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Time-series forecasting benchmark")
    parser.add_argument("--dataset", type=str, required=True, choices=["etth1", "etth2", "electricity", "weather"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=["all"], choices=ALL_MODELS + ["all"])
    parser.add_argument("--presets", nargs="+", default=None,
                         help="Model-size presets to sweep (default: all presets in config, for a Pareto curve)")
    parser.add_argument("--pred-lens", nargs="+", type=int, default=None,
                         help="Forecast horizons to sweep (default: config.data.pred_lens)")
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
    pred_lens = args.pred_lens or cfg["data"]["pred_lens"]

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    logger.info(f"Device: {device}")

    # Flatten to (model, preset) tasks — each trains its full pred_len x seed
    # grid sequentially inside its own process, but different (model, preset,
    # horizon) combinations all run concurrently on the same GPU.
    tasks = []
    for model_name in models:
        presets = args.presets or list(cfg["models"][model_name]["presets"].keys())
        for preset_name in presets:
            for pred_len in pred_lens:
                if _task_fully_done(results_root, model_name, preset_name, pred_len, seeds):
                    logger.info(f"  [SKIP] {model_name}/{preset_name}/h{pred_len} — all seeds already complete")
                    continue
                tasks.append((model_name, preset_name, pred_len))

    if tasks:
        device_str = str(device)
        ctx = mp.get_context("spawn")  # required for CUDA safety across processes
        failed = []
        with ProcessPoolExecutor(max_workers=args.max_parallel, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _run_task_worker, m, p, h, seeds, cfg, args.dataset, str(results_root), device_str
                ): (m, p, h)
                for m, p, h in tasks
            }
            for future in as_completed(futures):
                m, p, h = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception(f"FAILED {m}/{p}/h{h}")
                    failed.append((m, p, h))
        if failed:
            logger.warning(f"{len(failed)} (model, preset, horizon) combination(s) failed: {failed}")

    # presets may differ per model; for aggregation, take the union actually run
    all_presets = sorted({p for m in models for p in (args.presets or cfg["models"][m]["presets"].keys())})
    summary = aggregate_results(results_root, models, all_presets, pred_lens, seeds)
    with open(results_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pareto-sorted printout: params ascending within each horizon
    print("\n" + "=" * 90)
    print(f"FORECASTING SUMMARY — {args.dataset}")
    print("=" * 90)
    for pred_len in pred_lens:
        rows = [v for v in summary.values() if v["pred_len"] == pred_len]
        rows.sort(key=lambda r: r["params"]["mean"])
        print(f"\n--- horizon={pred_len} ---")
        print(f"{'MODEL':<10}{'PRESET':<8}{'PARAMS':>12}{'TEST_MSE':>12}{'TEST_MAE':>12}")
        print("-" * 90)
        for r in rows:
            print(
                f"{r['model']:<10}{r['preset']:<8}{int(r['params']['mean']):>12,}"
                f"{r['test_mse']['mean']:>12.4f}{r['test_mae']['mean']:>12.4f}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
