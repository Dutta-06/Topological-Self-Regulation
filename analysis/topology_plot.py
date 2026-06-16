"""
Topology evolution plots for TSR runs.

Reads the per-run topology.jsonl written by gate_experiment.py and produces:

  1. width_evolution.png  — per-layer channel/neuron count vs. training step
     (one line per TSR layer, coloured by layer index)

  2. gate_evolution.png   — mean gate value per layer vs. step
     (shows how open the gating is over time)

  3. activation_evolution.png — dominant activation per layer vs. step
     (heat-map: each row = a layer, each column = a step)

  4. events_timeline.png  — vertical tick marks for every structural event
     (prune / grow / insert_layer), overlaid on the val-accuracy curve from
     metrics.jsonl

Usage:
  python analysis/topology_plot.py \\
      --run-dir results/gate/tsr_phantom/seed42 \\
      [--out-dir results/gate/tsr_phantom/seed42/plots]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Guard matplotlib import so the module can be imported without display.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


ACT_COLOURS = {
    "relu": "#e74c3c",
    "tanh": "#3498db",
    "gelu": "#2ecc71",
    "silu": "#f39c12",
    "unknown": "#95a5a6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_run(run_dir: Path) -> dict:
    return {
        "topology": _load_jsonl(run_dir / "topology.jsonl"),
        "metrics":  _load_jsonl(run_dir / "metrics.jsonl"),
        "events":   _load_jsonl(run_dir / "events.jsonl"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _layer_series(topology_records: List[dict]) -> Dict[str, dict]:
    """Build per-layer time series from topology snapshots.

    Returns {layer_name: {"steps": [...], "out_size": [...], "gate_mean": [...],
                          "dominant_activation": [...]}}.
    """
    series: Dict[str, dict] = {}
    for snap in topology_records:
        step = snap.get("step", 0)
        for layer in snap.get("layers", []):
            name = layer.get("name", "?")
            if name not in series:
                series[name] = {"steps": [], "out_size": [], "gate_mean": [],
                                 "dominant_activation": []}
            series[name]["steps"].append(step)
            series[name]["out_size"].append(layer.get("out_size", 0))
            series[name]["gate_mean"].append(layer.get("gate_mean", 1.0))
            series[name]["dominant_activation"].append(
                layer.get("dominant_activation", "relu")
            )
    return series


def plot_width_evolution(topology_records, out_path: Path) -> None:
    if not _HAVE_MPL:
        return
    series = _layer_series(topology_records)
    if not series:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    colours = plt.cm.tab10.colors
    for i, (name, data) in enumerate(series.items()):
        ax.step(data["steps"], data["out_size"],
                where="post", color=colours[i % 10], label=name, linewidth=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer width (neurons / channels)")
    ax.set_title("TSR Width Evolution")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gate_evolution(topology_records, out_path: Path) -> None:
    if not _HAVE_MPL:
        return
    series = _layer_series(topology_records)
    if not series:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    colours = plt.cm.tab10.colors
    for i, (name, data) in enumerate(series.items()):
        ax.plot(data["steps"], data["gate_mean"],
                color=colours[i % 10], label=name, linewidth=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean gate value (sigmoid)")
    ax.set_title("TSR Gate Evolution")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_activation_evolution(topology_records, out_path: Path) -> None:
    if not _HAVE_MPL:
        return
    series = _layer_series(topology_records)
    if not series:
        return

    act_to_int = {"relu": 0, "tanh": 1, "gelu": 2, "silu": 3}
    layer_names = list(series.keys())
    # Find common step grid (union of all steps)
    all_steps = sorted({s for d in series.values() for s in d["steps"]})
    if not all_steps:
        return

    # Build matrix: rows=layers, cols=steps
    mat = []
    for name in layer_names:
        d = series[name]
        step_to_act = dict(zip(d["steps"], d["dominant_activation"]))
        row = [act_to_int.get(step_to_act.get(s, "relu"), 0) for s in all_steps]
        mat.append(row)

    import numpy as np
    mat = np.array(mat, dtype=float)

    cmap = matplotlib.colors.ListedColormap(
        [ACT_COLOURS["relu"], ACT_COLOURS["tanh"],
         ACT_COLOURS["gelu"], ACT_COLOURS["silu"]]
    )

    fig, ax = plt.subplots(figsize=(12, max(2, len(layer_names) * 0.5 + 1)))
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=3,
              extent=[all_steps[0], all_steps[-1], len(layer_names) - 0.5, -0.5])
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.set_xlabel("Training step")
    ax.set_title("Dominant Activation per Layer over Training")

    patches = [mpatches.Patch(color=v, label=k) for k, v in ACT_COLOURS.items()
               if k != "unknown"]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_events_timeline(metrics_records, events_records, out_path: Path) -> None:
    if not _HAVE_MPL:
        return
    if not metrics_records:
        return

    steps = [m.get("step", 0) for m in metrics_records]
    accs  = [m.get("val_accuracy", 0.0) for m in metrics_records]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps, accs, color="#2c3e50", linewidth=2, label="Val Accuracy")

    event_colours = {"prune": "#e74c3c", "grow": "#27ae60", "insert_layer": "#8e44ad"}
    seen = set()
    for ev in events_records:
        action = ev.get("action", "unknown")
        colour = event_colours.get(action, "#7f8c8d")
        label = action if action not in seen else None
        seen.add(action)
        ax.axvline(x=ev.get("step", 0), color=colour, alpha=0.6,
                   linewidth=1, linestyle="--", label=label)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Structural Events Timeline")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def make_plots(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_run(run_dir)

    if not data["topology"] and not data["metrics"]:
        print(f"No data found in {run_dir}. Skipping.")
        return

    plot_width_evolution(data["topology"],      out_dir / "width_evolution.png")
    plot_gate_evolution(data["topology"],        out_dir / "gate_evolution.png")
    plot_activation_evolution(data["topology"],  out_dir / "activation_evolution.png")
    plot_events_timeline(data["metrics"],
                         data["events"],         out_dir / "events_timeline.png")

    print(f"Plots written to {out_dir}/")


def main():
    if not _HAVE_MPL:
        print("matplotlib not available — install it to produce plots.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="TSR topology evolution plots")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Directory containing topology.jsonl, metrics.jsonl, events.jsonl")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for PNG files (default: <run-dir>/plots)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    make_plots(run_dir, out_dir)


if __name__ == "__main__":
    main()
