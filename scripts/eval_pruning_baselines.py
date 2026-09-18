"""Tabulate the pruning baselines against the three archived arms.

For every results/pruned/*.pt, reads its recorded best top-1 and deployed count
and lines it up with the cell's dense reference (the stamped re-measured value
where present), C2 and online TSR-X. Writes results/pruning_baselines.csv and
prints a Markdown table.

Columns: Ref, C2, TSR, then one column per (criterion, mode) present, all
top-1 in percent at the same deployed count, and C2 - pruned for each.

Usage:
    python scripts/eval_pruning_baselines.py
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.prune_baseline import ARCH_KEY, find_match, find_reference  # noqa: E402

KEY_ARCH = {v: k for k, v in ARCH_KEY.items()}


def top1(ck: dict) -> float:
    if "verified_top1" in ck:
        return float(ck["verified_top1"]) / 100.0
    return float(ck.get("best_val_acc", ck.get("val_acc")))


def main() -> None:
    rows = defaultdict(dict)
    for path in sorted((ROOT / "results" / "pruned").glob("*.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if not ck.get("complete"):
            print(f"skipping incomplete {path.name}")
            continue
        cell = (ck["arch"], ck["dataset"])
        col = f"{ck['criterion']}/{ck['mode']}"
        rows[cell][col] = (top1(ck), int(ck["params"]))
        rows[cell]["pruned_params"] = int(ck["params"])

    if not rows:
        raise SystemExit("no complete pruning baselines under results/pruned/")

    baseline_cols = sorted({c for r in rows.values() for c in r if "/" in c})
    out_rows = []
    for (arch, dataset), r in sorted(rows.items()):
        cifar_stem = dataset not in ("imagenet100", "imagenet-100")
        ref = torch.load(find_reference(arch, dataset, cifar_stem), map_location="cpu", weights_only=False)
        match_path = find_match(arch, dataset)
        c2 = torch.load(match_path, map_location="cpu", weights_only=False) if match_path else None
        tsr_path = ROOT / "results" / "tsrx" / f"{ARCH_KEY[arch]}_{dataset}_two_regime.pt"
        tsr = torch.load(tsr_path, map_location="cpu", weights_only=False) if tsr_path.exists() else None
        row = {"arch": arch, "dataset": dataset,
               "ref_top1": 100 * top1(ref),
               "c2_top1": 100 * top1(c2) if c2 else None,
               "tsr_top1": 100 * top1(tsr) if tsr else None,
               "c2_params": int(c2["params"]) if c2 else None,
               "pruned_params": r["pruned_params"]}
        for col in baseline_cols:
            if col in r:
                acc, _params = r[col]
                row[col] = 100 * acc
                row[f"c2_minus_{col}"] = (row["c2_top1"] - 100 * acc) if c2 else None
            else:
                row[col] = row[f"c2_minus_{col}"] = None
        out_rows.append(row)

    fields = list(out_rows[0].keys())
    csv_path = ROOT / "results" / "pruning_baselines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    def fmt(v):
        return "--" if v is None else (f"{v:+.2f}" if isinstance(v, float) and abs(v) < 20 else f"{v:.2f}" if isinstance(v, float) else f"{v:,}")

    head = ["arch", "dataset", "ref", "C2", "TSR"] + baseline_cols + [f"C2 - {c}" for c in baseline_cols]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in out_rows:
        vals = [row["arch"], row["dataset"], fmt(row["ref_top1"]), fmt(row["c2_top1"]), fmt(row["tsr_top1"])]
        vals += [fmt(row[c]) for c in baseline_cols] + [fmt(row[f"c2_minus_{c}"]) for c in baseline_cols]
        print("| " + " | ".join(str(v) for v in vals) + " |")
    print(f"\nwrote {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
