"""Run the structured-pruning baselines over the vision cells, self-contained.

Designed for a machine that has only this branch and the datasets: no archived
checkpoint is required. For each cell the sweep

  1. trains the dense reference with the shared recipe if none is present under
     results/reference/ (bench/train_reference.py, 100 epochs, seed 42), and
     stamps it complete;
  2. prunes it to the exact deployed count of the archived C2 model, read from
     bench/pruning_targets.json (or from the C2 checkpoint if one is present);
  3. fine-tunes (or retrains from scratch) and records the result as
     results/pruned/<arch>_<dataset>_<criterion>_<mode>.json next to the
     checkpoint. The JSON is what gets committed; the .pt files are ignored.

Every step is resumable: existing complete outputs are skipped.

Usage:
    python scripts/run_pruning_baselines.py --num-workers 8              # everything
    python scripts/run_pruning_baselines.py --criteria l1 --modes scratch --cells resnet18/cifar100
    python scripts/run_pruning_baselines.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.prune_baseline import (  # noqa: E402
    ARCH_KEY, CRITERIA, MODES, find_reference, load_targets, reference_name,
)

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


def ensure_reference(arch: str, dataset: str, num_workers: int, dry_run: bool) -> bool:
    """Train the dense reference here if none is present; returns True when one exists."""
    cifar_stem = dataset not in ("imagenet100", "imagenet-100")
    try:
        find_reference(arch, dataset, cifar_stem)
        return True
    except SystemExit:
        pass
    out = ROOT / "results" / "reference" / reference_name(arch, dataset)
    cmd = [sys.executable, "-m", "bench.train_reference", "--arch", arch, "--dataset", dataset,
           "--epochs", "100", "--seed", "42", "--num-workers", str(num_workers),
           "--batch-size", "64" if not cifar_stem else "128", "--out", str(out)]
    print(f"\n[reference missing] $ {' '.join(cmd)}")
    if dry_run:
        return True
    res = subprocess.run(cmd, cwd=ROOT)
    if res.returncode != 0 or not out.exists():
        print(f"FAILED to train the reference for {arch}/{dataset}; skipping its cells")
        return False
    ck = torch.load(out, map_location="cpu", weights_only=False)
    ck["complete"] = True
    ck["total_epochs"] = 100
    ck["trained_for"] = "pruning-baselines"
    torch.save(ck, out)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None, help="arch/dataset pairs; default: every cell with a target")
    ap.add_argument("--criteria", nargs="*", choices=CRITERIA, default=["l1", "bnscale"])
    ap.add_argument("--modes", nargs="*", choices=MODES, default=["finetune"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-train-reference", action="store_true",
                    help="skip cells whose reference is missing instead of training it")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra flags passed through to prune_baseline.py")
    args = ap.parse_args()

    targets = load_targets()
    cells = ([tuple(c.split("/")) for c in args.cells] if args.cells
             else [(a, d) for a in ARCHS for d in DATASETS])
    missing = [(a, d) for a, d in cells if f"{a}/{d}" not in targets]
    cells = [c for c in cells if c not in missing]
    if missing:
        print("no target count recorded (no archived C2 yet), skipped: "
              + ", ".join(f"{a}/{d}" for a, d in missing))

    jobs = [(a, d, c, m) for a, d in cells for c in args.criteria for m in args.modes]
    todo = [j for j in jobs if not is_complete(output_path(*j))]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already complete, {len(todo)} to run")

    ready = {}
    for arch, dataset, criterion, mode in todo:
        if (arch, dataset) not in ready:
            if args.no_train_reference:
                cifar_stem = dataset not in ("imagenet100", "imagenet-100")
                try:
                    find_reference(arch, dataset, cifar_stem)
                    ready[(arch, dataset)] = True
                except SystemExit:
                    print(f"no reference for {arch}/{dataset}; skipped (--no-train-reference)")
                    ready[(arch, dataset)] = False
            else:
                ready[(arch, dataset)] = ensure_reference(arch, dataset, args.num_workers, args.dry_run)
        if not ready[(arch, dataset)]:
            continue
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

    if not args.dry_run:
        print("\nDone. Results to commit: results/pruned/*.json  (checkpoints stay local)")
        print("Table: python scripts/eval_pruning_baselines.py")


if __name__ == "__main__":
    main()
