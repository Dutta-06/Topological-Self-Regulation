"""Aggregate the sensitivity sweep (scripts/run_sensitivity.py) against its
centre point, the ablation campaign's `released` run of the same cell/seed.

For each setting: online top-1 (best and last epoch), the change from the
centre, the exported parameter count, births, and how similar the exported
allocation is to the centre's (cosine of log2(width/ref) vectors).

    python scripts/eval_sensitivity.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.eval_ablations import canonical, cosine, cost_equivalent_orbits, growth_stats, widths_log_ratio  # noqa: E402
from scripts.run_sensitivity import SETTINGS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="resnet18_cifar100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=str(ROOT / "results" / "sensitivity"))
    ap.add_argument("--centre-root", default=str(ROOT / "results" / "ablations"))
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    def rec(path: Path):
        j = path.with_suffix(".json")
        return json.loads(j.read_text(encoding="utf-8")) if j.exists() else None

    centre_ck = Path(args.centre_root) / args.cell / f"released_s{args.seed}.pt"
    centre = rec(centre_ck)
    rows = [("centre (H~=1, k=4, K=100, α=0.5)", centre_ck, centre)]
    for name in SETTINGS:
        ck = Path(args.root) / args.cell / f"{name}_s{args.seed}.pt"
        rows.append((name + "  " + " ".join(SETTINGS[name]), ck, rec(ck)))

    arch, ds = args.cell.split("_", 1)
    arch = {"vgg16bn": "vgg16_bn", "mobilenetv2": "mobilenet_v2"}.get(arch, arch)
    orbits = cost_equivalent_orbits(arch, ds)
    cw = widths_log_ratio(centre_ck) if centre else None
    lines = [f"# Sensitivity, {args.cell} seed {args.seed} (online arm)\n",
             "| setting | best | last | Δlast vs centre | params | births | top-group share | cos vs centre | modulo swaps |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, ck, r in rows:
        if r is None:
            lines.append(f"| {label} | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        best, last = 100 * r["best_val_acc"], 100 * (r.get("final_val_acc") or float("nan"))
        d = last - 100 * centre["final_val_acc"] if centre and centre.get("final_val_acc") else float("nan")
        b, top, _, _, _ = growth_stats(ck.with_name(f"{ck.stem}_decisions.jsonl"))
        w = widths_log_ratio(ck)
        c = cosine(w, cw) if cw else float("nan")
        cc = cosine(canonical(w, orbits), canonical(cw, orbits)) if cw else float("nan")
        lines.append(f"| {label} | {best:.2f} | {last:.2f} | {d:+.2f} | {r['params']:,} | {b} | {top:.0%} "
                     f"| {c:.3f} | {cc:.3f} |")
    lines.append(f"\nInterchangeable (cost-free swap) groups: {orbits or 'none'}.")
    print("\n".join(lines))
    if args.markdown:
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
