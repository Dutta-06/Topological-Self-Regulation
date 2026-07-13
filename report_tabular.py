#!/usr/bin/env python3
"""Neat summary table for TSR tabular benchmark runs.

Reads benchmarks/tabular/results/{dataset}/tsr_summary.json (tsr vs
tsr_static_final) and summary.json (external baselines), prints one aligned
table comparing TSR against its own matched-param control and against the
best external baseline, per dataset.

Usage: python3 report_tabular.py
"""

import json
import os

RESULTS_DIR = "benchmarks/tabular/results"
DATASETS = ["adult", "california_housing", "covertype", "higgs", "fashion_mnist"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def metric_key_of(entry):
    return "test_mse" if "test_mse" in entry else "test_acc"


def higher_is_better(metric_key):
    return metric_key == "test_acc"


def best_baseline(baselines, metric_key):
    best_name, best_val, best_params = None, None, None
    for name, entry in baselines.items():
        if metric_key not in entry:
            continue
        val = entry[metric_key]["mean"]
        params = entry.get("params", {}).get("mean")
        if best_val is None:
            best_name, best_val, best_params = name, val, params
            continue
        better = val > best_val if higher_is_better(metric_key) else val < best_val
        if better:
            best_name, best_val, best_params = name, val, params
    return best_name, best_val, best_params


def fmt_metric(val, metric_key):
    if val is None:
        return "  n/a  "
    return f"{val:.4f}" if metric_key == "test_acc" else f"{val:.4f}"


def fmt_params(p):
    return "n/a" if p is None else f"{int(p):,}"


def main():
    rows = []
    for ds in DATASETS:
        ds_dir = os.path.join(RESULTS_DIR, ds)
        tsr_summary = load(os.path.join(ds_dir, "tsr_summary.json"))
        baselines = load(os.path.join(ds_dir, "summary.json")) or {}

        if not tsr_summary or "tsr" not in tsr_summary:
            rows.append((ds, None))
            continue

        tsr = tsr_summary["tsr"]
        static = tsr_summary.get("tsr_static_final", {})
        metric_key = metric_key_of(tsr)
        hib = higher_is_better(metric_key)

        tsr_val = tsr[metric_key]["mean"]
        tsr_params = tsr.get("params", {}).get("mean")
        static_val = static.get(metric_key, {}).get("mean")
        static_params = static.get("params", {}).get("mean")

        base_name, base_val, base_params = best_baseline(baselines, metric_key)

        vs_static = None
        if static_val is not None:
            vs_static = "TSR" if (tsr_val > static_val) == hib else "static"
        vs_baseline = None
        if base_val is not None:
            vs_baseline = "TSR" if (tsr_val > base_val) == hib else "baseline"

        rows.append((ds, {
            "metric_key": metric_key,
            "tsr_val": tsr_val, "tsr_params": tsr_params,
            "static_val": static_val, "static_params": static_params,
            "base_name": base_name, "base_val": base_val, "base_params": base_params,
            "vs_static": vs_static, "vs_baseline": vs_baseline,
        }))

    col = "{:<20}{:<8}{:>12}{:>12}{:>12}{:>12}{:>20}{:>12}{:>10}{:>10}"
    header = col.format(
        "dataset", "metric", "tsr", "tsr_params", "static", "static_p",
        "best_base", "base_val", "vs_stat", "vs_base",
    )
    print(header)
    print("-" * len(header))

    for ds, d in rows:
        if d is None:
            print(f"{ds:<20} MISSING (no tsr_summary.json)")
            continue
        print(col.format(
            ds,
            d["metric_key"].replace("test_", ""),
            fmt_metric(d["tsr_val"], d["metric_key"]),
            fmt_params(d["tsr_params"]),
            fmt_metric(d["static_val"], d["metric_key"]),
            fmt_params(d["static_params"]),
            f"{d['base_name']}" if d["base_name"] else "n/a",
            fmt_metric(d["base_val"], d["metric_key"]),
            d["vs_static"] or "n/a",
            d["vs_baseline"] or "n/a",
        ))
        if d["base_params"] is not None:
            print(f"{'':<20}{'':<8}{'':>12}{'':>12}{'':>12}{'':>12}{'(params='+format(int(d['base_params']),',')+')':>20}")

    print()
    print("vs_stat/vs_base = which side won on the metric (params comparison shown separately above).")


if __name__ == "__main__":
    main()
