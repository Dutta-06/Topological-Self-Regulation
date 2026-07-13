#!/usr/bin/env python3
"""Neat summary tables for TSR timeseries benchmark runs (forecasting + classification).

Forecasting: reads benchmarks/timeseries/results/{ds}/tsr_forecasting_summary.json
(tsr vs tsr_static_final, per horizon) and summary.json (external baselines,
keyed by model/preset/h{horizon}).

Classification: reads benchmarks/timeseries/results/{ds}/tsr_classification_summary.json
and the matching baseline summary.json (handles har's nested results/har/har/summary.json
layout as well as ucr_uea/{DS}/summary.json).

Usage: python3 report_timeseries.py
"""

import json
import os

RESULTS_DIR = "benchmarks/timeseries/results"
FORECAST_DATASETS = ["etth1", "etth2", "weather", "electricity"]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, nd=4):
    return "n/a" if val is None else f"{val:.{nd}f}"


def fmt_params(p):
    return "n/a" if p is None else f"{int(p):,}"


def discover_classification_datasets():
    datasets = []
    if os.path.isdir(os.path.join(RESULTS_DIR, "har")):
        datasets.append(("har", os.path.join(RESULTS_DIR, "har")))
    ucr_dir = os.path.join(RESULTS_DIR, "ucr_uea")
    if os.path.isdir(ucr_dir):
        for name in sorted(os.listdir(ucr_dir)):
            path = os.path.join(ucr_dir, name)
            if os.path.isdir(path):
                datasets.append((name, path))
    return datasets


def baseline_summary_for(ds_dir, ds_name):
    # har's baseline runner nests its output one extra level (results/har/har/summary.json);
    # ucr_uea datasets do not. Try flat first, then the nested form.
    flat = load(os.path.join(ds_dir, "summary.json"))
    if flat:
        return flat
    nested = load(os.path.join(ds_dir, ds_name, "summary.json"))
    return nested or {}


def print_forecasting():
    print("=" * 100)
    print("FORECASTING")
    print("=" * 100)
    col = "{:<14}{:>6}{:>12}{:>14}{:>12}{:>22}{:>12}{:>10}{:>10}"
    header = col.format(
        "dataset", "h", "tsr_mse", "tsr_params", "static_mse",
        "best_baseline", "base_mse", "vs_stat", "vs_base",
    )
    print(header)
    print("-" * len(header))

    for ds in FORECAST_DATASETS:
        ds_dir = os.path.join(RESULTS_DIR, ds)
        tsr_summary = load(os.path.join(ds_dir, "tsr_forecasting_summary.json"))
        baselines = load(os.path.join(ds_dir, "summary.json")) or {}

        if not tsr_summary:
            print(f"{ds:<14} MISSING (no tsr_forecasting_summary.json)")
            continue

        horizons = sorted({v["horizon"] for v in tsr_summary.values()})
        for h in horizons:
            tsr = tsr_summary.get(f"tsr/pred{h}")
            static = tsr_summary.get(f"tsr_static_final/pred{h}")
            if tsr is None:
                continue

            tsr_mse = tsr["test_mse"]["mean"]
            tsr_params = tsr.get("params", {}).get("mean")
            static_mse = static["test_mse"]["mean"] if static else None

            best_name, best_mse, best_params = None, None, None
            for name, entry in baselines.items():
                if entry.get("pred_len") != h or "test_mse" not in entry:
                    continue
                val = entry["test_mse"]["mean"]
                if best_mse is None or val < best_mse:
                    best_name, best_mse = name, val
                    best_params = entry.get("params", {}).get("mean")

            vs_static = "n/a" if static_mse is None else ("TSR" if tsr_mse < static_mse else "static")
            vs_base = "n/a" if best_mse is None else ("TSR" if tsr_mse < best_mse else "baseline")

            print(col.format(
                ds, h, fmt(tsr_mse), fmt_params(tsr_params), fmt(static_mse),
                best_name or "n/a", fmt(best_mse), vs_static, vs_base,
            ))
            if best_params is not None:
                print(f"{'':<14}{'':>6}{'':>12}{'':>14}{'':>12}{'(params='+format(int(best_params), ',')+')':>22}")


def print_classification():
    print()
    print("=" * 100)
    print("CLASSIFICATION")
    print("=" * 100)
    col = "{:<16}{:>10}{:>12}{:>10}{:>22}{:>10}{:>10}{:>10}"
    header = col.format(
        "dataset", "tsr_acc", "tsr_params", "static", "best_baseline", "base_acc", "vs_stat", "vs_base",
    )
    print(header)
    print("-" * len(header))

    for ds_name, ds_dir in discover_classification_datasets():
        tsr_summary = load(os.path.join(ds_dir, "tsr_classification_summary.json"))
        if not tsr_summary or "tsr" not in tsr_summary:
            print(f"{ds_name:<16} MISSING (no tsr_classification_summary.json)")
            continue

        baselines = baseline_summary_for(ds_dir, ds_name)

        tsr = tsr_summary["tsr"]
        static = tsr_summary.get("tsr_static_final", {})
        tsr_acc = tsr["test_acc"]["mean"]
        tsr_params = tsr.get("params", {}).get("mean")
        static_acc = static.get("test_acc", {}).get("mean")

        best_name, best_acc, best_params = None, None, None
        for name, entry in baselines.items():
            if "test_acc" not in entry:
                continue
            val = entry["test_acc"]["mean"]
            if best_acc is None or val > best_acc:
                best_name, best_acc = name, val
                best_params = entry.get("params", {}).get("mean")

        vs_static = "n/a" if static_acc is None else ("TSR" if tsr_acc > static_acc else "static")
        vs_base = "n/a" if best_acc is None else ("TSR" if tsr_acc > best_acc else "baseline")

        print(col.format(
            ds_name, fmt(tsr_acc), fmt_params(tsr_params), fmt(static_acc),
            best_name or "n/a", fmt(best_acc), vs_static, vs_base,
        ))
        if best_params is not None:
            print(f"{'':<16}{'':>10}{'':>12}{'':>10}{'(params='+format(int(best_params), ',')+')':>22}")


def main():
    print_forecasting()
    print_classification()


if __name__ == "__main__":
    main()
