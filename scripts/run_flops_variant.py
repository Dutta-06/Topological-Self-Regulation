"""FLOP-budgeted discovery and its C2 retrain, for chosen vision cells.

Same recipe as the archived parameter-budgeted campaign (85% budget, k=4,
100-step window, 100 epochs, seed 42) with `--cost flops`, so the controller
ranks proposals by kappa_flops and the annealed ceiling is 85% of the dense
reference's forward FLOPs. The exported width vector is then retrained from
scratch exactly as C2 is. Cells run concurrently; each writes
results/tsrx_flops/<cell>.pt, results/static_matched_flops/<cell>.pt, and a
JSON record with parameters, FLOPs and accuracies of both arms next to the
archived reference values; scripts/eval_flops_variant.py tabulates.

Usage:
    python scripts/run_flops_variant.py                                  # EfficientNet-B0 x CIFARs
    python scripts/run_flops_variant.py --cells efficientnet_b0/cifar10 vgg16_bn/cifar100
"""

import argparse
import json
import subprocess
import threading
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ARCH_KEY = {"resnet18": "resnet18", "vgg16_bn": "vgg16bn",
            "mobilenet_v2": "mobilenetv2", "efficientnet_b0": "efficientnet_b0"}


def done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("epoch", 0)) >= 90
    except Exception:
        return False


def run(cmd, logfile: Path) -> bool:
    with logfile.open("w", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode == 0


# torch.fx symbolic tracing keeps global state, so two cells measuring at once corrupt
# each other's trace ("module is not installed as a submodule"). Measuring is seconds;
# serialising it costs nothing and the trainings still run concurrently.
_MEASURE_LOCK = threading.Lock()


def measure(arch: str, dataset: str, tsrx_ckpt: Path, c2_ckpt: Path) -> dict:
    from bench.models import build_model
    from bench.train_static_matched import resize_model_to_widths
    from tsrx.alloc.cost import model_flops
    hw = {"cifar10": 32, "cifar100": 32, "tiny_imagenet": 64, "imagenet100": 224}[dataset]
    ncls = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200, "imagenet100": 100}[dataset]
    ex = torch.zeros(2, 3, hw, hw)
    ref = build_model(arch, ncls, cifar_stem=hw != 224)
    tsr = torch.load(tsrx_ckpt, map_location="cpu", weights_only=False)
    c2 = torch.load(c2_ckpt, map_location="cpu", weights_only=False)
    disc = resize_model_to_widths(build_model(arch, ncls, cifar_stem=hw != 224), tsr["discovered_widths"], ex)
    table = json.loads((ROOT / "bench" / "pruning_targets.json").read_text(encoding="utf-8"))["cells"] \
        if (ROOT / "bench" / "pruning_targets.json").exists() else {}
    cell = table.get(f"{arch}/{dataset}", {})
    return {
        "arch": arch, "dataset": dataset, "cost_mode": "flops",
        "reference_params": sum(p.numel() for p in ref.parameters()), "reference_flops": model_flops(ref, ex),
        "discovered_params": sum(p.numel() for p in disc.parameters()), "discovered_flops": model_flops(disc, ex),
        "discovered_widths": tsr["discovered_widths"],
        "tsr_best_top1": float(tsr["best_val_acc"]), "c2_best_top1": float(c2["best_val_acc"]),
        "archived": {k: cell[k] for k in ("reference_top1", "c2_top1", "tsr_top1", "target_params")} if cell else None,
        "checkpoints": [tsrx_ckpt.name, c2_ckpt.name],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def pipeline(arch: str, dataset: str, num_workers: int, dry_run: bool) -> str:
    key = ARCH_KEY[arch]
    tsrx_out = ROOT / "results" / "tsrx_flops" / f"{key}_{dataset}_flops.pt"
    c2_out = ROOT / "results" / "static_matched_flops" / f"{key}_{dataset}_flops.pt"
    for p in (tsrx_out, c2_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not done(tsrx_out):
        cmd = [sys.executable, "-m", "bench.train_tsrx", "--arch", arch, "--dataset", dataset,
               "--epochs", "100", "--batch-size", "128", "--budget-ratio", "0.85", "--cost", "flops",
               "--seed", "42", "--num-workers", str(num_workers), "--out", str(tsrx_out)]
        print(f"[{arch}/{dataset} discovery] $ {' '.join(cmd)}", flush=True)
        if not dry_run and not run(cmd, tsrx_out.with_suffix(".log")):
            return f"FAILED discovery for {arch}/{dataset} (see {tsrx_out.with_suffix('.log').name})"
    if not done(c2_out):
        cmd = [sys.executable, "-m", "bench.train_static_matched", "--tsrx-checkpoint", str(tsrx_out),
               "--epochs", "100", "--batch-size", "128", "--seed", "42",
               "--num-workers", str(num_workers), "--out", str(c2_out)]
        print(f"[{arch}/{dataset} C2] $ {' '.join(cmd)}", flush=True)
        if not dry_run and not run(cmd, c2_out.with_suffix(".log")):
            return f"FAILED C2 for {arch}/{dataset} (see {c2_out.with_suffix('.log').name})"
    if dry_run:
        return f"{arch}/{dataset}: dry run"
    with _MEASURE_LOCK:
        rec = measure(arch, dataset, tsrx_out, c2_out)
    rec["seconds"] = time.time() - t0
    tsrx_out.with_suffix(".json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return (f"{arch}/{dataset}: FLOPs {rec['reference_flops']:,} -> {rec['discovered_flops']:,} "
            f"({100 * (rec['discovered_flops'] / rec['reference_flops'] - 1):+.1f}%), params "
            f"{100 * (rec['discovered_params'] / rec['reference_params'] - 1):+.1f}%, "
            f"TSR {100 * rec['tsr_best_top1']:.2f}%, C2 {100 * rec['c2_best_top1']:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["efficientnet_b0/cifar10", "efficientnet_b0/cifar100"])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--concurrent", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cells = [tuple(c.split("/")) for c in args.cells]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrent)) as pool:
        for f in [pool.submit(pipeline, a, d, args.num_workers, args.dry_run) for a, d in cells]:
            print(f.result(), flush=True)
    if not args.dry_run:
        print("\nTable: python scripts/eval_flops_variant.py")


if __name__ == "__main__":
    main()
