"""Run the structured-pruning baselines over every vision cell that has a C2 arm.

Each cell x criterion x mode is one call to bench/prune_baseline.py, writing
results/pruned/<arch>_<dataset>_<criterion>_<mode>.pt. Cells whose output
already exists (and is marked complete) are skipped, so the sweep can be
resumed. Cells without a C2 checkpoint are listed and skipped: the baseline is
defined at C2's deployed count and cannot run before C2 exists.

Usage:
    python scripts/run_pruning_baselines.py                       # everything
    python scripts/run_pruning_baselines.py --criteria l1 --modes finetune
    python scripts/run_pruning_baselines.py --cells resnet18/cifar100 vgg16_bn/cifar100
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.prune_baseline import ARCH_KEY, CRITERIA, MODES, find_match  # noqa: E402

ARCHS = ["resnet18", "vgg16_bn", "mobilenet_v2", "efficientnet_b0"]
DATASETS = ["cifar10", "cifar100", "tiny_imagenet", "imagenet100"]


def output_path(arch: str, dataset: str, criterion: str, mode: str) -> Path:
    return ROOT / "results" / "pruned" / f"{ARCH_KEY[arch]}_{dataset}_{criterion}_{mode}.pt"


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(torch.load(path, map_location="cpu", weights_only=False).get("complete"))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None, help="arch/dataset pairs; default: all with a C2 arm")
    ap.add_argument("--criteria", nargs="*", choices=CRITERIA, default=["l1", "bnscale"])
    ap.add_argument("--modes", nargs="*", choices=MODES, default=["finetune"])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra flags passed through to prune_baseline.py")
    args = ap.parse_args()

    cells = ([tuple(c.split("/")) for c in args.cells] if args.cells
             else [(a, d) for a in ARCHS for d in DATASETS])
    missing = [(a, d) for a, d in cells if find_match(a, d) is None]
    cells = [c for c in cells if c not in missing]
    if missing:
        print("no C2 arm yet, skipped: " + ", ".join(f"{a}/{d}" for a, d in missing))

    jobs = [(a, d, c, m) for a, d in cells for c in args.criteria for m in args.modes]
    todo = [j for j in jobs if not is_complete(output_path(*j))]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already complete, {len(todo)} to run")
    for arch, dataset, criterion, mode in todo:
        out = output_path(arch, dataset, criterion, mode)
        cmd = [sys.executable, "-m", "bench.prune_baseline", "--arch", arch, "--dataset", dataset,
               "--criterion", criterion, "--mode", mode, "--num-workers", str(args.num_workers),
               "--out", str(out), *args.extra]
        print("\n$ " + " ".join(cmd))
        if args.dry_run:
            continue
        res = subprocess.run(cmd, cwd=ROOT)
        if res.returncode != 0:
            print(f"FAILED ({res.returncode}): {arch}/{dataset} {criterion} {mode}; continuing")


if __name__ == "__main__":
    main()
