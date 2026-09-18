"""Tabulate the forecasting pruning baselines against Ref / C2 / TSR-X / C3.

For every results/ts_pruned/*.pt, lines up its test MSE with the cell's dense
reference, the C2 it was matched to, the TSR-X run behind that C2 and the C3
random control of the same stem, all at the same deployed count. Writes
results/ts_pruning_baselines.csv and prints a Markdown table; "C2 - pruned"
is negative when the discovered allocation is better (lower MSE).

Usage:
    python scripts/eval_pruning_baselines_ts.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False) if path and path.exists() else None


def main() -> None:
    cells = defaultdict(dict)
    for path in sorted((ROOT / "results" / "ts_pruned").glob("*.pt")):
        ck = load(path)
        if not ck or not ck.get("complete"):
            print(f"skipping incomplete {path.name}")
            continue
        stem = Path(ck["match"]).stem
        cells[stem][f"{ck['criterion']}/{ck['mode']}"] = ck["test_mse"]
        cells[stem]["_ck"] = ck
    if not cells:
        raise SystemExit("no complete pruning baselines under results/ts_pruned/")

    cols = sorted({c for r in cells.values() for c in r if "/" in c})
    rows = []
    for stem, r in sorted(cells.items()):
        ck = r["_ck"]
        c2 = load(ROOT / "results" / "ts_static_matched" / f"{stem}.pt")
        c3 = load(ROOT / "results" / "ts_c3" / f"{stem}.pt")
        tsr = load(ROOT / "results" / "ts_tsrx" / f"{stem}.pt")
        row = {"cell": stem, "arch": ck["arch"], "dataset": ck["dataset"], "horizon": ck["pred_len"],
               "ref_params": ck["reference_params"], "params": ck["params"],
               "ref_test_mse": ck["reference_test_mse"],
               "c2_test_mse": c2["test_mse"] if c2 else None,
               "tsr_test_mse": (tsr or {}).get("test_mse"),
               "c3_test_mse": c3["test_mse"] if c3 else None}
        for col in cols:
            row[col] = r.get(col)
            row[f"c2_minus_{col}"] = (row["c2_test_mse"] - r[col]) if (c2 and col in r) else None
        rows.append(row)

    csv_path = ROOT / "results" / "ts_pruning_baselines.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(v):
        return "--" if v is None else (f"{v:+.4f}" if "minus" in str(v) else f"{v:.4f}") if isinstance(v, float) else str(v)

    head = ["cell", "dP (%)", "Ref", "C2", "TSR", "C3"] + cols + [f"C2 - {c}" for c in cols]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        vals = [row["cell"], f"{100 * (row['params'] / row['ref_params'] - 1):+.1f}",
                fmt(row["ref_test_mse"]), fmt(row["c2_test_mse"]), fmt(row["tsr_test_mse"]), fmt(row["c3_test_mse"])]
        vals += [fmt(row[c]) for c in cols]
        vals += ["--" if row[f"c2_minus_{c}"] is None else f"{row[f'c2_minus_{c}']:+.4f}" for c in cols]
        print("| " + " | ".join(vals) + " |")
    print(f"\nwrote {csv_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
