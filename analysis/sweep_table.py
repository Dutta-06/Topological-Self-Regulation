"""Compression-vs-accuracy frontier from a budget sweep, plus the validity
checks that decide whether the numbers mean anything.

`analysis/collect_results.py` globs the standard arm directories; the sweep
writes to results/ts_sweep/ with budget-tagged filenames, so it needs its
own reader. Expected layout:

    results/ts_sweep/tsrx_<ds>_h<H>_br<BR>.pt
    results/ts_sweep/c2_<ds>_h<H>_br<BR>.pt
    results/ts_sweep/c3_<ds>_h<H>_br<BR>.pt

VALIDITY comes first and is not optional. Two failure modes have already
silently invalidated whole campaigns in this project:
  * C2 built from the best-val snapshot instead of the converged
    architecture -- a run reported as 15% was really 1.4%.
  * C3 not matched to TSR-X's final parameter count, which makes it not a
    control at all.
Both are checked here and printed before any accuracy number.

Usage:
    python -m analysis.sweep_table --dataset weather --horizon 96 \
        --reference results/ablate/mixing_w96.pt
"""

import argparse
import glob
import os
import re

import torch


def _load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results/ts_sweep")
    ap.add_argument("--dataset", default="weather")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--reference", default=None,
                    help="reference checkpoint to compare against (e.g. "
                         "results/ablate/mixing_w96.pt or results/ts_reference/tcn_weather_h96.pt)")
    args = ap.parse_args()

    pat = os.path.join(args.sweep_dir, f"tsrx_{args.dataset}_h{args.horizon}_br*.pt")
    budgets = []
    for p in sorted(glob.glob(pat)):
        m = re.search(r"_br([0-9.]+)\.pt$", p)
        if m:
            budgets.append(m.group(1))
    if not budgets:
        raise SystemExit(f"no sweep checkpoints matching {pat}")

    ref = _load(args.reference) if args.reference else None
    ref_val = ref.get("val_mse") if ref else None
    ref_test = ref.get("test_mse") if ref else None
    ref_params = ref.get("params") if ref else None

    print("=" * 92)
    print("VALIDITY CHECKS  (if any of these fail, the accuracy table below is meaningless)")
    print("=" * 92)
    ok_all = True
    for br in budgets:
        tag = f"{args.dataset}_h{args.horizon}_br{br}"
        tx = _load(f"{args.sweep_dir}/tsrx_{tag}.pt")
        c2 = _load(f"{args.sweep_dir}/c2_{tag}.pt")
        c3 = _load(f"{args.sweep_dir}/c3_{tag}.pt")
        if tx is None:
            print(f"  br={br}: MISSING tsrx checkpoint"); ok_all = False; continue

        final_p = tx.get("final_params")
        issues = []
        if final_p is None:
            issues.append("tsrx has no final_params (run bench.backfill_final_widths)")
        if c2 is None:
            issues.append("C2 missing")
        elif final_p and c2.get("params") != final_p:
            issues.append(f"C2 params {c2.get('params'):,} != tsrx final {final_p:,} "
                          f"(C2 used the WRONG architecture)")
        if c3 is None:
            issues.append("C3 missing")
        else:
            rel = c3.get("param_match_rel_error")
            if rel is None or rel > 0.01:
                issues.append(f"C3 off target by {('n/a' if rel is None else f'{rel*100:.2f}%')} (>1% = not a control)")
        dorm = tx.get("max_port_magnitude")
        if dorm not in (None, 0.0):
            issues.append(f"dormancy leaked: max_port_magnitude={dorm}")

        if issues:
            ok_all = False
            print(f"  br={br}: FAIL")
            for i in issues:
                print(f"      - {i}")
        else:
            print(f"  br={br}: ok   (C2 and C3 both matched to tsrx final_params = {final_p:,})")
    print()

    print("=" * 92)
    print(f"COMPRESSION FRONTIER  —  {args.dataset} h{args.horizon}")
    if ref is not None:
        print(f"reference: {ref_params:,} params   val {ref_val:.4f}   test {ref_test:.4f}")
    print("=" * 92)
    hdr = f"{'budget':>7}{'params':>10}{'reduct':>8} | {'TSRX val':>9}{'C2 val':>9}{'C3 val':>9} | {'C2 test':>9}{'vs ref':>10}{'C2vC3':>9}"
    print(hdr)
    print("-" * len(hdr))
    for br in budgets:
        tag = f"{args.dataset}_h{args.horizon}_br{br}"
        tx, c2, c3 = (_load(f"{args.sweep_dir}/{a}_{tag}.pt") for a in ("tsrx", "c2", "c3"))
        if tx is None:
            continue
        base = tx.get("baseline_params")
        fp = tx.get("final_params") or tx.get("params")
        red = (1 - fp / base) * 100 if base and fp else float("nan")
        tv = tx.get("final_val_mse", tx.get("best_val_mse"))
        c2v = c2.get("val_mse") if c2 else None
        c3v = c3.get("val_mse") if c3 else None
        c2t = c2.get("test_mse") if c2 else None

        vs = f"{c2t - ref_test:+.4f}" if (c2t is not None and ref_test is not None) else "--"
        if c2v is not None and c3v is not None:
            d = c3v - c2v            # positive => C2 lower => signal wins
            c2vc3 = f"{d:+.4f}{'*' if d > 0 else ''}"
        else:
            c2vc3 = "--"
        f = lambda v: f"{v:.4f}" if isinstance(v, float) else "--"
        print(f"{br:>7}{fp:>10,}{red:>7.1f}% | {f(tv):>9}{f(c2v):>9}{f(c3v):>9} | "
              f"{f(c2t):>9}{vs:>10}{c2vc3:>9}")

    print()
    print("C2vC3 positive (*) = the topological signal beat random allocation at the same budget.")
    print("vs ref: negative = the compressed model is BETTER than the full reference.")
    if not ok_all:
        print("\n!! validity checks failed above — fix those before reading anything into this table.")


if __name__ == "__main__":
    main()
