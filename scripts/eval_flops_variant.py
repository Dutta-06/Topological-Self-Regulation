"""Tabulate the FLOP-budgeted cells against the parameter-budgeted ones.

Reads results/tsrx_flops/*.json (written by scripts/run_flops_variant.py) and
prints, per cell, the reference, the archived parameter-budget arms and the
FLOP-budget arms with their parameter and FLOP changes. Writes
results/flops_variant.csv.

Usage:
    python scripts/eval_flops_variant.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    rows = []
    for path in sorted((ROOT / "results" / "tsrx_flops").glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        a = r.get("archived") or {}
        rows.append({
            "arch": r["arch"], "dataset": r["dataset"],
            "ref_top1": a.get("reference_top1"),
            "param_budget_c2": a.get("c2_top1"), "param_budget_tsr": a.get("tsr_top1"),
            "param_budget_dP": (100 * (a["target_params"] / r["reference_params"] - 1)) if a else None,
            "flop_budget_c2": 100 * r["c2_best_top1"], "flop_budget_tsr": 100 * r["tsr_best_top1"],
            "flop_budget_dP": 100 * (r["discovered_params"] / r["reference_params"] - 1),
            "flop_budget_dF": 100 * (r["discovered_flops"] / r["reference_flops"] - 1),
            "reference_flops": r["reference_flops"], "discovered_flops": r["discovered_flops"],
        })
    if not rows:
        raise SystemExit("no records under results/tsrx_flops/*.json")

    def f(v, s=False):
        return "--" if v is None else (f"{v:+.2f}" if s else f"{v:.2f}")

    print("| cell | Ref | param-budget: ΔP | C2 | TSR | FLOP-budget: ΔP | ΔF | C2 | TSR |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['arch']}/{r['dataset']} | {f(r['ref_top1'])} | {f(r['param_budget_dP'], True)} | "
              f"{f(r['param_budget_c2'])} | {f(r['param_budget_tsr'])} | {f(r['flop_budget_dP'], True)} | "
              f"{f(r['flop_budget_dF'], True)} | {f(r['flop_budget_c2'])} | {f(r['flop_budget_tsr'])} |")
    out = ROOT / "results" / "flops_variant.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
