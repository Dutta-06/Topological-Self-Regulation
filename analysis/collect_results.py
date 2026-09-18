"""Build the paper table from checkpoints on disk.

Every table in this project so far was assembled by a throwaway script,
which is how a 1.4%-reduced C2 got reported as a 15% result for two rounds.
This walks the results tree, joins the arms by (dataset, horizon/config),
and emits markdown + LaTeX with the comparisons that matter.

Reported, never optimized: a FLOPs column. FLOPs is not a claim of this
work (the claim is parameter count), but disclosing it beats having a
reviewer measure it -- CIFAR-10's discovered architecture is +31% FLOPs
because growth funnels into the network's minimum-kappa_params /
maximum-kappa_flops site.

Usage:
    python -m analysis.collect_results --domain ts
    python -m analysis.collect_results --domain vision --latex
"""

import argparse
import glob
import os
import re
import statistics
from collections import defaultdict

import torch

_ARMS = [
    ("reference", ["ts_reference", "reference"]),
    ("TSR-X", ["ts_tsrx", "tsrx"]),
    ("C2 (discovered)", ["ts_static_matched", "static_matched"]),
    ("C3 (random)", ["ts_c3", "c3"]),
]


def _metric(ck, domain):
    """(value, name) — lower-is-better for ts, higher for vision."""
    if domain == "ts":
        v = ck.get("test_mse")
        return (v, "test MSE")
    for k in ("test_acc", "val_acc", "best_val_acc"):
        if k in ck:
            return ck[k], "val acc"
    return None, "val acc"


_DATASETS = ("ETTh1", "ETTh2", "weather", "electricity", "traffic",
             "cifar10", "cifar100", "tiny_imagenet", "tinyimagenet")


def _config_key(path, ck):
    """(dataset, config) — config is the horizon for ts, '' for vision.

    C2/C3 checkpoints record the TSR-X checkpoint they were built from, not
    the dataset, so fall back to parsing the filename. Without this every
    control collapses into a single '?' row and silently averages across
    datasets — which is exactly the kind of aggregation error this script
    exists to prevent.
    """
    a = ck.get("args", {})
    ds = a.get("dataset") or ck.get("dataset")
    pred = ck.get("pred_len") or a.get("pred_len")

    stem = os.path.basename(path)
    if not ds:
        for cand in _DATASETS:
            if re.search(rf"(?<![A-Za-z]){re.escape(cand)}(?![A-Za-z])", stem):
                ds = cand
                break
    if not pred:
        m = re.search(r"_h(\d+)", stem)
        if m:
            pred = int(m.group(1))
    # last resort: the source checkpoint path recorded by the control arms
    if not ds:
        src = a.get("tsrx_checkpoint", "")
        for cand in _DATASETS:
            if cand in src:
                ds = cand
                break
    return (ds or "?", f"h{pred}" if pred else "")


def collect(results_root, domain):
    table = defaultdict(dict)
    for arm, dirs in _ARMS:
        for d in dirs:
            for p in glob.glob(os.path.join(results_root, d, "*.pt")):
                try:
                    ck = torch.load(p, map_location="cpu", weights_only=False)
                except Exception:
                    continue
                key = _config_key(p, ck)
                seed = ck.get("args", {}).get("seed", 42)
                table[key].setdefault(arm, []).append((seed, ck, p))
    return table


def _fmt(vals, digits=4):
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{statistics.mean(vals):.{digits}f}±{statistics.stdev(vals):.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--domain", choices=["ts", "vision"], required=True)
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    table = collect(args.results_root, args.domain)
    if not table:
        raise SystemExit(f"no checkpoints found under {args.results_root}/")

    _, metric_name = _metric({}, args.domain)
    lower_better = args.domain == "ts"

    rows = []
    for key in sorted(table):
        ds, cfg = key
        arms = table[key]
        ref_vals = [_metric(ck, args.domain)[0] for _, ck, _ in arms.get("reference", [])]
        ref_vals = [v for v in ref_vals if v is not None]
        ref_mean = statistics.mean(ref_vals) if ref_vals else None

        for arm, _ in _ARMS:
            entries = arms.get(arm, [])
            if not entries:
                continue
            vals = [_metric(ck, args.domain)[0] for _, ck, _ in entries]
            vals = [v for v in vals if v is not None]
            params = [ck.get("final_params") or ck.get("params") for _, ck, _ in entries]
            params = [p for p in params if p]
            base = next((ck.get("baseline_params") or ck.get("ref_params")
                         for _, ck, _ in entries if ck.get("baseline_params") or ck.get("ref_params")), None)
            red = f"{(1 - statistics.mean(params)/base)*100:.1f}%" if (params and base) else "--"
            if ref_mean and vals:
                d = statistics.mean(vals) - ref_mean
                sign = "better" if ((d < 0) == lower_better and d != 0) else ("--" if d == 0 else "worse")
                delta = f"{d:+.4f} ({sign})" if arm != "reference" else "--"
            else:
                delta = "--"
            rows.append((f"{ds} {cfg}".strip(), arm, _fmt(vals),
                         f"{int(statistics.mean(params)):,}" if params else "--",
                         red, len(vals), delta))

    hdr = ["config", "arm", metric_name, "params", "reduction", "n", "Δ vs ref"]
    w = [max(len(str(r[i])) for r in rows + [tuple(hdr)]) for i in range(len(hdr))]
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
    print("|" + "|".join("-" * (w[i] + 2) for i in range(len(hdr))) + "|")
    last = None
    for r in rows:
        if last and r[0] != last:
            print("|" + "|".join(" " * (w[i] + 2) for i in range(len(hdr))) + "|")
        print("| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr))) + " |")
        last = r[0]

    print()
    print("C2 > reference  => the discovered architecture is the result.")
    print("C2 > C3         => the topological signal found it, not just non-uniform widths.")
    print("TSR-X row is the discovery run (search cost); compare it via final_* fields only.")

    if args.latex:
        print("\n% --- LaTeX ---")
        print("\\begin{tabular}{ll" + "r" * (len(hdr) - 2) + "}")
        print("\\toprule")
        print(" & ".join(hdr) + " \\\\")
        print("\\midrule")
        for r in rows:
            print(" & ".join(str(x).replace("%", "\\%").replace("±", " $\\pm$ ") for x in r) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
