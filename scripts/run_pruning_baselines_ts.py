"""Run the forecasting pruning baselines for every archived C2 checkpoint.

Each results/ts_static_matched/*.pt (one per backbone x dataset x horizon x
budget, as scripts/overnight_ts.sh names them) defines one target count; the
baseline prunes that cell's reference to it. Outputs go to
results/ts_pruned/<C2 stem>_<criterion>_<mode>.pt and are skipped when complete.

Usage:
    python scripts/run_pruning_baselines_ts.py                       # everything
    python scripts/run_pruning_baselines_ts.py --criteria l1 taylor --modes finetune scratch
    python scripts/run_pruning_baselines_ts.py --only weather_h96   # substring filter on the C2 stem
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CRITERIA = ("l1", "bnscale", "taylor")
MODES = ("finetune", "scratch")


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(torch.load(path, map_location="cpu", weights_only=False).get("complete"))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", default="results/ts_static_matched", help="directory of C2 checkpoints")
    ap.add_argument("--only", nargs="*", default=None, help="substrings a C2 stem must contain")
    ap.add_argument("--criteria", nargs="*", choices=CRITERIA, default=["l1", "taylor"])
    ap.add_argument("--modes", nargs="*", choices=MODES, default=["finetune"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra flags passed through to prune_baseline_ts.py")
    args = ap.parse_args()

    matches = sorted((ROOT / args.matches).glob("*.pt"))
    if args.only:
        matches = [m for m in matches if any(s in m.stem for s in args.only)]
    jobs = [(m, c, mode) for m in matches for c in args.criteria for mode in args.modes]
    out_dir = ROOT / "results" / "ts_pruned"
    todo = [(m, c, mode) for m, c, mode in jobs
            if not is_complete(out_dir / f"{m.stem}_{c}_{mode}.pt")]
    print(f"{len(matches)} C2 checkpoints, {len(jobs)} jobs, {len(jobs) - len(todo)} complete, {len(todo)} to run")
    for match, criterion, mode in todo:
        out = out_dir / f"{match.stem}_{criterion}_{mode}.pt"
        cmd = [sys.executable, "-m", "bench.prune_baseline_ts", "--match", str(match),
               "--criterion", criterion, "--mode", mode, "--out", str(out), *args.extra]
        print("\n$ " + " ".join(cmd))
        if args.dry_run:
            continue
        res = subprocess.run(cmd, cwd=ROOT)
        if res.returncode != 0:
            print(f"FAILED ({res.returncode}): {match.stem} {criterion} {mode}; continuing")


if __name__ == "__main__":
    main()
