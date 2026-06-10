"""
Ablation / comparison table generator.

Reads results/gate/summary.json (produced by gate_experiment.py) and renders:

  1. Console: an ASCII table with mean ± std for each variant.

  2. results/gate/ablation_table.csv  — machine-readable, for spreadsheets.

  3. results/gate/ablation_table.tex  — LaTeX booktabs table, ready to paste
     into the AAAI paper.

  4. results/gate/pareto_frontier.png — accuracy vs. inference MFLOPs scatter
     with TSR variants highlighted (extends the existing pareto_plot.py to
     include both TSR modes alongside the VGG baselines).

Usage:
  python analysis/ablation_table.py [--summary results/gate/summary.json]
                                    [--out-dir results/gate]
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAVE_MPL = True
except ImportError:
    _HAVE_MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# Display names and column ordering
# ─────────────────────────────────────────────────────────────────────────────

DISPLAY = {
    "tsr_phantom":   "TSR (phantom, ours)",
    "tsr_heuristic": "TSR (heuristic)",
    "vgg_tiny":      "VGG-Tiny   [8,8,16]",
    "vgg_small":     "VGG-Small  [16,16,32]",
    "static_final":  "Static-Final (TSR shape)",
}

VARIANT_ORDER = [
    "tsr_phantom",
    "tsr_heuristic",
    "vgg_tiny",
    "vgg_small",
    "static_final",
]

# Marker styles for the Pareto plot
MARKERS = {
    "tsr_phantom":   ("*", "#2980b9", 300),
    "tsr_heuristic": ("D", "#8e44ad", 150),
    "vgg_tiny":      ("o", "#7f8c8d", 100),
    "vgg_small":     ("o", "#95a5a6", 100),
    "static_final":  ("s", "#bdc3c7", 100),
}


# ─────────────────────────────────────────────────────────────────────────────
# Table rendering
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(mean: float, std: float, pct: bool = False) -> str:
    scale = 100.0 if pct else 1.0
    return f"{mean * scale:.2f} ± {std * scale:.2f}"


def _acc(v: dict) -> tuple:
    a = v.get("val_accuracy", {})
    return a.get("mean", 0.0), a.get("std", 0.0)


def _inf(v: dict) -> tuple:
    a = v.get("inference_flops", {})
    return a.get("mean", 0.0) / 1e6, a.get("std", 0.0) / 1e6


def _par(v: dict) -> tuple:
    a = v.get("params", {})
    return a.get("mean", 0.0), a.get("std", 0.0)


def _train(v: dict) -> tuple:
    a = v.get("total_train_flops", {})
    return a.get("mean", 0.0) / 1e9, a.get("std", 0.0) / 1e9


def print_ascii(summary: dict) -> None:
    header = f"{'Model':<28} {'Val Acc (%)':>18} {'Inf MFLOPs':>18} {'Params':>16} {'Train GFLOPs':>16}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for key in VARIANT_ORDER:
        if key not in summary:
            continue
        v = summary[key]
        m_acc, s_acc = _acc(v)
        m_inf, s_inf = _inf(v)
        m_par, s_par = _par(v)
        m_tr,  s_tr  = _train(v)
        name = DISPLAY.get(key, key)
        print(
            f"{name:<28} "
            f"{_fmt(m_acc, s_acc, pct=True):>18} "
            f"{m_inf:>9.2f} ± {s_inf:.2f}  "
            f"{m_par:>8.0f} ± {s_par:.0f}  "
            f"{m_tr:>7.3f} ± {s_tr:.3f}"
        )
    print(sep + "\n")


def write_csv(summary: dict, out_path: Path) -> None:
    rows = []
    for key in VARIANT_ORDER:
        if key not in summary:
            continue
        v = summary[key]
        m_acc, s_acc = _acc(v)
        m_inf, s_inf = _inf(v)
        m_par, s_par = _par(v)
        m_tr,  s_tr  = _train(v)
        rows.append({
            "variant":           key,
            "display_name":      DISPLAY.get(key, key),
            "val_acc_mean":      f"{m_acc * 100:.4f}",
            "val_acc_std":       f"{s_acc * 100:.4f}",
            "inf_mflops_mean":   f"{m_inf:.4f}",
            "inf_mflops_std":    f"{s_inf:.4f}",
            "params_mean":       f"{m_par:.1f}",
            "params_std":        f"{s_par:.1f}",
            "train_gflops_mean": f"{m_tr:.4f}",
            "train_gflops_std":  f"{s_tr:.4f}",
        })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV written to {out_path}")


def write_latex(summary: dict, out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Gate Experiment: CIFAR-10, 3 seeds (mean ± std). "
        r"\textbf{TSR (phantom)} is the proposed system; "
        r"TSR (heuristic) uses the original bottleneck-score growth signal; "
        r"Static-Final matches TSR's discovered architecture trained from scratch.}",
        r"\label{tab:gate_experiment}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Val Acc (\%) & Inf.\ MFLOPs & Params & Train GFLOPs \\",
        r"\midrule",
    ]
    for key in VARIANT_ORDER:
        if key not in summary:
            continue
        v = summary[key]
        m_acc, s_acc = _acc(v)
        m_inf, s_inf = _inf(v)
        m_par, s_par = _par(v)
        m_tr,  s_tr  = _train(v)
        name_tex = DISPLAY.get(key, key).replace("_", r"\_")
        if key == "tsr_phantom":
            name_tex = r"\textbf{" + name_tex + r"}"
        lines.append(
            f"{name_tex} & "
            f"${m_acc*100:.2f} \\pm {s_acc*100:.2f}$ & "
            f"${m_inf:.2f} \\pm {s_inf:.2f}$ & "
            f"${m_par:.0f} \\pm {s_par:.0f}$ & "
            f"${m_tr:.3f} \\pm {s_tr:.3f}$ \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX written to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pareto plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto(summary: dict, out_path: Path) -> None:
    if not _HAVE_MPL:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    xs, ys = [], []
    for key in VARIANT_ORDER:
        if key not in summary:
            continue
        v = summary[key]
        m_acc, _ = _acc(v)
        m_inf, s_inf = _inf(v)
        marker, colour, size = MARKERS.get(key, ("o", "#7f8c8d", 80))
        ax.scatter(m_inf, m_acc * 100, marker=marker, color=colour,
                   s=size, zorder=5, label=DISPLAY.get(key, key))
        if marker == "o":
            xs.append(m_inf)
            ys.append(m_acc * 100)

    # Static Pareto frontier through VGG baselines
    if xs:
        pts = sorted(zip(xs, ys))
        frontier_x, frontier_y = [pts[0][0]], [pts[0][1]]
        for x, y in pts[1:]:
            if y >= frontier_y[-1]:
                frontier_x.append(x)
                frontier_y.append(y)
        ax.plot(frontier_x, frontier_y, "k--", alpha=0.4, label="Static Pareto")

    ax.set_xlabel("Inference Compute (MFLOPs)")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("CIFAR-10 Accuracy vs. Inference Cost")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Pareto plot written to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ROOT = Path(__file__).resolve().parent.parent
    default_summary = str(ROOT / "results" / "gate" / "summary.json")

    parser = argparse.ArgumentParser(description="Render ablation table from gate experiment results")
    parser.add_argument("--summary", type=str, default=default_summary,
                        help="Path to summary.json produced by gate_experiment.py")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory (default: directory of summary.json)")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"summary.json not found at {summary_path}")
        print("Run gate_experiment.py first.")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print_ascii(summary)
    write_csv(summary, out_dir / "ablation_table.csv")
    write_latex(summary, out_dir / "ablation_table.tex")
    plot_pareto(summary, out_dir / "pareto_frontier.png")


if __name__ == "__main__":
    main()
