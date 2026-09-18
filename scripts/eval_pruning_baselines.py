"""Tabulate the pruning baselines against Ref / C2 / TSR-X.

Reads the JSON records the runner writes (results/pruned/*.json) -- no
checkpoint is needed -- and lines each pruned result up with the cell's three
archived arms. The arms' accuracies come from bench/pruning_targets.json (the
paper's re-measured values), so the table can be built on any machine; the
reference that was actually pruned on the run machine is reported as its own
column, since a reference trained there differs from the archived one by seed
and hardware.

Writes results/pruning_baselines.csv and prints a Markdown table. "C2 - <col>"
is positive when the discovered allocation is better (higher top-1).

Usage:
    python scripts/eval_pruning_baselines.py [--latex]
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.prune_baseline import load_targets  # noqa: E402

ARCH_LABEL = {"resnet18": "ResNet-18", "vgg16_bn": "VGG-16-BN", "mobilenet_v2": "MobileNetV2",
              "efficientnet_b0": "EfficientNet-B0"}
DATA_LABEL = {"cifar10": "C-10", "cifar100": "C-100", "tiny_imagenet": "Tiny-IN", "imagenet100": "IN-100"}
ORDER = ["resnet18", "vgg16_bn", "mobilenet_v2", "efficientnet_b0"]
DORDER = ["cifar10", "cifar100", "tiny_imagenet", "imagenet100"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="also print table rows in the paper's format")
    args = ap.parse_args()

    targets = load_targets()
    cells = defaultdict(dict)
    for path in sorted((ROOT / "results" / "pruned").glob("*.json")):
        if path.name.endswith(".events.json"):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        if not rec.get("complete"):
            print(f"skipping incomplete {path.name}")
            continue
        cell = f"{rec['arch']}/{rec['dataset']}"
        col = f"{rec['criterion']}/{rec['mode']}"
        cells[cell][col] = 100 * rec["best_top1"]
        cells[cell]["_ref_here"] = 100 * rec["reference_top1"]
        cells[cell]["_params"] = rec["params"]
        cells[cell]["_before"] = cells[cell].get("_before", {})
        cells[cell]["_before"][col] = 100 * rec["pruned_top1_before_training"]
    if not cells:
        raise SystemExit("no complete pruning records under results/pruned/*.json")

    cols = sorted({c for r in cells.values() for c in r if "/" in c})
    rows = []
    for arch in ORDER:
        for dataset in DORDER:
            cell = f"{arch}/{dataset}"
            if cell not in cells:
                continue
            r, t = cells[cell], targets.get(cell, {})
            row = {"arch": arch, "dataset": dataset,
                   "ref_params": t.get("reference_params"), "params": r["_params"],
                   "ref_top1": t.get("reference_top1"), "c2_top1": t.get("c2_top1"),
                   "tsr_top1": t.get("tsr_top1"), "ref_top1_here": r["_ref_here"]}
            best = None
            for col in cols:
                row[col] = r.get(col)
                row[f"before_{col}"] = r["_before"].get(col)
                if row[col] is not None and (best is None or row[col] > best):
                    best = row[col]
            row["best_pruned"] = best
            row["c2_minus_ref"] = (row["c2_top1"] - row["ref_top1"]) if t else None
            row["c2_minus_best_pruned"] = (row["c2_top1"] - best) if (t and best is not None) else None
            rows.append(row)

    csv_path = ROOT / "results" / "pruning_baselines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def f(v, signed=False):
        if v is None:
            return "--"
        return f"{v:+.2f}" if signed else f"{v:.2f}"

    head = ["Backbone", "Data", "Ref", "Ref (here)", "C2", "TSR"] + cols + ["C2-Ref", "C2-best P"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        vals = [ARCH_LABEL[row["arch"]], DATA_LABEL[row["dataset"]], f(row["ref_top1"]),
                f(row["ref_top1_here"]), f(row["c2_top1"]), f(row["tsr_top1"])]
        vals += [f(row[c]) for c in cols]
        vals += [f(row["c2_minus_ref"], True), f(row["c2_minus_best_pruned"], True)]
        print("| " + " | ".join(vals) + " |")
    print(f"\nwrote {csv_path.relative_to(ROOT)}")

    if args.latex:
        print("\n% rows for tab:vision with the pruned columns; \\Delta P from the archived counts")
        for row in rows:
            dp = 100 * (row["params"] / row["ref_params"] - 1) if row["ref_params"] else float("nan")
            cells_tex = [f"${dp:+.2f}$", f(row["ref_top1"]), f(row["c2_top1"]), f(row["tsr_top1"])]
            cells_tex += [f(row[c]) for c in cols]
            cells_tex += [f"${row['c2_minus_ref']:+.2f}$" if row["c2_minus_ref"] is not None else "--",
                          f"${row['c2_minus_best_pruned']:+.2f}$" if row["c2_minus_best_pruned"] is not None else "--"]
            print(f"{ARCH_LABEL[row['arch']]} & {DATA_LABEL[row['dataset']]} & " + " & ".join(cells_tex) + r" \\")


if __name__ == "__main__":
    main()
