"""Backfill `final_widths` into TSR-X timeseries checkpoints produced
BEFORE train_ts_tsrx.py recorded it.

Why this exists: the checkpoint is written on best-validation-MSE, but LTSF
models bottom out at epoch 1-9 while the budget anneal is still ramping
(end_frac=0.5 => epoch 25 of 50). So `discovered_widths` can capture a
barely-pruned model: measured on the first full sweep, weather_h96 reached
16.1% parameter reduction by the end of training but its best-val
checkpoint recorded only 1.4%. A C2 control built from that snapshot
trains the wrong architecture and the run's headline reduction is fiction.

The decision trace (`*_decisions.jsonl`) records every applied edit, so the
converged architecture is recoverable by replaying prune/grow events from
the baseline widths. The replay is checked against the `deployed_after`
figure the trainer logged and refuses to write on any mismatch, so a
silently wrong reconstruction cannot be committed to a checkpoint.

Checkpoints are gitignored; decision traces are tracked. Run this on any
machine that has the original .pt files but not the backfill:

    python -m bench.backfill_final_widths results/ts_tsrx
"""

import argparse
import glob
import json
import os

import torch

from bench.resize import resize_model_to_widths
from bench.ts_models import build_ts_model
from data.ltsf import n_channels

# Coupling-group taps for the 4-block TCN: 4 per-block bottlenecks
# (conv1->conv2) plus the shared residual-stream group. Every one starts at
# `hidden`. Verified against discover_groups() in tests_tsrx/test_timeseries.py.
_TCN_TAPS = (0, 3, 5, 7, 8)


def replay_final_widths(ckpt_path: str, decisions_path: str):
    """Return (widths, params) for the converged architecture, or raise."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    hidden, pred_len, seq_len, ds = a["hidden"], a["pred_len"], a["seq_len"], a["dataset"]

    with open(decisions_path) as f:
        recs = [json.loads(line) for line in f]

    widths = {t: hidden for t in _TCN_TAPS}
    for r in recs:
        if r["action"] == "none":
            continue
        if r["prune_tap"] is not None:
            widths[int(r["prune_tap"])] -= 1
        if r["grow_tap"] is not None:
            widths[int(r["grow_tap"])] += 1

    n_vars = n_channels(ds)
    model = build_ts_model("tcn", n_vars, pred_len, hidden=hidden)
    model = resize_model_to_widths(
        model, {str(k): v for k, v in widths.items()}, torch.zeros(2, seq_len, n_vars)
    )
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logged = recs[-1]["deployed_after"]
    if params != logged:
        raise ValueError(
            f"replay mismatch: reconstructed {params:,} but the trainer logged "
            f"{logged:,} deployed params. Refusing to write a checkpoint whose "
            f"architecture cannot be reproduced from the decision trace."
        )
    return ck, {str(k): v for k, v in widths.items()}, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default="results/ts_tsrx")
    ap.add_argument("--force", action="store_true",
                    help="overwrite final_widths even if already present")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.pt")))
    if not paths:
        raise SystemExit(f"no .pt checkpoints under {args.results_dir}")

    for p in paths:
        tag = os.path.basename(p)[4:-3]
        dec = p.replace(".pt", "_decisions.jsonl")
        if not os.path.exists(dec):
            print(f"{tag:<18} SKIP  no decision trace at {os.path.basename(dec)}")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if "final_widths" in ck and not args.force:
            print(f"{tag:<18} skip  already has final_widths ({ck.get('final_params', 0):,})")
            continue
        try:
            ck, widths, params = replay_final_widths(p, dec)
        except ValueError as e:
            print(f"{tag:<18} FAIL  {e}")
            continue
        base = ck["baseline_params"]
        ck["final_widths"] = widths
        ck["final_params"] = params
        ck["final_param_saving_pct"] = (1 - params / base) * 100
        ck["final_widths_provenance"] = (
            "replayed from decisions.jsonl (verified exact vs logged deployed_after)"
        )
        torch.save(ck, p)
        print(f"{tag:<18} OK    {params:>12,}  ({ck['final_param_saving_pct']:.1f}% reduction)  "
              f"best-val snapshot was {ck['params']:,} ({(1-ck['params']/base)*100:.1f}%)")


if __name__ == "__main__":
    main()
