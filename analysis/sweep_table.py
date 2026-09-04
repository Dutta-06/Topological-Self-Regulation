"""Compression-vs-accuracy frontier from a budget sweep, plus the validity
checks that decide whether the numbers mean anything.

`analysis/collect_results.py` globs the standard arm directories; the sweep
writes to results/ts_sweep/ with budget-tagged filenames, so it needs its
own reader. Expected layout, all produced by scripts/campaign.sh:

    results/ts_sweep/tsrx_<ds>_h<H>_br<BR>.pt              seed 42 (Stage 1)
    results/ts_sweep/tsrx_<ds>_h<H>_br<BR>_s<SEED>.pt       seeds 43/44 (Stage 2)
    results/ts_sweep/c2_<ds>_h<H>_br<BR>[_s<SEED>].pt
    results/ts_sweep/c3_<ds>_h<H>_br<BR>[_s<SEED>].pt       seed-42 random draw
    results/ts_sweep/c3_<ds>_h<H>_br<BR>_c<N>.pt            extra random draws (Stage 3)

Multiple seeds/draws at one budget are aggregated as mean +/- std (std omitted
at n=1, which is reported as a bare number so it is never mistaken for an
error bar).

VALIDITY comes first and is not optional. Two failure modes have already
silently invalidated whole campaigns in this project:
  * C2 built from the best-val snapshot instead of the converged
    architecture -- a run reported as 15% was really 1.4%.
  * C3 not matched to TSR-X's final parameter count, which makes it not a
    control at all.
Both are checked here, per checkpoint, and printed before any accuracy number.

Usage:
    python -m analysis.sweep_table --dataset weather --horizon 96 \
        --reference results/ts_reference/tcn_weather_h96.pt
"""

import argparse
import glob
import os
import re
import statistics

import torch


def _load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _fmt(vals, digits=4):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{statistics.mean(vals):.{digits}f}±{statistics.stdev(vals):.{digits}f}"


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _tag(arch, ds, h):
    """Filename tag for one (arch, dataset, horizon). `arch=""` matches the
    original untagged naming (tsrx_<ds>_h<H>_br<BR>.pt) used by the first
    TCN sweep, so that old data stays readable without a rename."""
    return f"{arch}_{ds}_h{h}" if arch else f"{ds}_h{h}"


def _find_budgets(sweep_dir, arch, ds, h):
    pat = os.path.join(sweep_dir, f"tsrx_{_tag(arch, ds, h)}_br*.pt")
    budgets = set()
    for p in glob.glob(pat):
        m = re.match(rf"tsrx_{re.escape(_tag(arch, ds, h))}_br([0-9.]+)(?:_s\d+)?\.pt$",
                     os.path.basename(p))
        if m:
            budgets.add(m.group(1))
    return sorted(budgets, key=float, reverse=True)


def _variants(sweep_dir, arm, arch, ds, h, br):
    """Every checkpoint for this arm/budget across seeds (_s<N>) and extra
    C3 draws (_c<N>). Base file (no suffix) is seed 42 / the default draw."""
    base = f"{arm}_{_tag(arch, ds, h)}_br{br}"
    paths = [os.path.join(sweep_dir, base + ".pt")]
    paths += sorted(glob.glob(os.path.join(sweep_dir, base + "_s*.pt")))
    paths += sorted(glob.glob(os.path.join(sweep_dir, base + "_c*.pt")))
    return [p for p in paths if os.path.exists(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results/ts_sweep")
    ap.add_argument("--dataset", default="weather")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--reference", default=None,
                    help="reference checkpoint to compare against, e.g. "
                         "results/ts_reference/patchtst_weather_h96.pt")
    ap.add_argument("--arch", default="",
                    help="arch tag in the sweep filenames (tsrx_<arch>_<ds>_h<H>_br<BR>.pt). "
                         "Empty (default) reads the original untagged TCN sweep.")
    args = ap.parse_args()

    budgets = _find_budgets(args.sweep_dir, args.arch, args.dataset, args.horizon)
    if not budgets:
        raise SystemExit(f"no sweep checkpoints for arch={args.arch!r} {args.dataset} h{args.horizon} under {args.sweep_dir}")

    ref = _load(args.reference) if args.reference else None
    ref_test = ref.get("test_mse") if ref else None
    ref_params = ref.get("params") if ref else None

    print("=" * 100)
    print("VALIDITY CHECKS  (if any of these fail, the accuracy table below is meaningless)")
    print("=" * 100)
    ok_all = True
    for br in budgets:
        tx_paths = _variants(args.sweep_dir, "tsrx", args.arch, args.dataset, args.horizon, br)
        issues, n_checked = [], 0
        final_ps = set()
        for txp in tx_paths:
            tx = _load(txp)
            if tx is None:
                issues.append(f"unreadable: {os.path.basename(txp)}"); continue
            n_checked += 1
            suffix = os.path.basename(txp)[len(f"tsrx_{_tag(args.arch, args.dataset, args.horizon)}_br{br}"):-3]
            final_p = tx.get("final_params")
            if final_p is None:
                issues.append(f"{suffix or '(seed42)'}: no final_params"); continue
            final_ps.add(final_p)
            dorm = tx.get("max_port_magnitude")
            if dorm not in (None, 0.0):
                issues.append(f"{suffix or '(seed42)'}: dormancy leaked (max_port_magnitude={dorm})")

            c2 = _load(os.path.join(args.sweep_dir, f"c2_{_tag(args.arch, args.dataset, args.horizon)}_br{br}{suffix}.pt"))
            if c2 is None:
                issues.append(f"{suffix or '(seed42)'}: C2 missing")
            elif c2.get("params") != final_p:
                issues.append(f"{suffix or '(seed42)'}: C2 params {c2.get('params'):,} != "
                              f"tsrx final {final_p:,} (C2 used the WRONG architecture)")

        for c3p in _variants(args.sweep_dir, "c3", args.arch, args.dataset, args.horizon, br):
            c3 = _load(c3p)
            suffix = os.path.basename(c3p)[len(f"c3_{_tag(args.arch, args.dataset, args.horizon)}_br{br}"):-3]
            rel = c3.get("param_match_rel_error") if c3 else None
            if rel is None or rel > 0.01:
                issues.append(f"C3{suffix or '(seed42)'}: off target by "
                              f"{('n/a' if rel is None else f'{rel*100:.2f}%')} (>1% = not a control)")

        if n_checked == 0:
            print(f"  br={br}: MISSING tsrx checkpoint(s)"); ok_all = False; continue
        if issues:
            ok_all = False
            print(f"  br={br}: FAIL  ({n_checked} seed/draw file(s) found)")
            for i in issues:
                print(f"      - {i}")
        else:
            fp = final_ps.pop() if len(final_ps) == 1 else f"varies: {sorted(final_ps)}"
            print(f"  br={br}: ok   (n={n_checked}, final_params={fp})")
    print()

    print("=" * 100)
    print(f"COMPRESSION FRONTIER  —  {args.arch or 'tcn(untagged)'} / {args.dataset} h{args.horizon}   (mean±std where n>1)")
    if ref is not None:
        print(f"reference: {ref_params:,} params   test {ref_test:.4f}")
    print("=" * 100)
    hdr = (f"{'budget':>7}{'params':>10}{'reduct':>8} | {'n(seed)':>8}{'C2 test':>16} | "
           f"{'vs ref':>10} | {'n(C3)':>7}{'C2 val':>10}{'C3 val':>16}{'C2-C3':>10}")
    print(hdr)
    print("-" * len(hdr))
    for br in budgets:
        tx_paths = _variants(args.sweep_dir, "tsrx", args.arch, args.dataset, args.horizon, br)
        if not tx_paths:
            continue
        tx0 = _load(tx_paths[0])
        base, fp = tx0.get("baseline_params"), tx0.get("final_params") or tx0.get("params")
        red = (1 - fp / base) * 100 if base and fp else float("nan")

        c2_tests, c2_vals = [], []
        for txp in tx_paths:
            suffix = os.path.basename(txp)[len(f"tsrx_{_tag(args.arch, args.dataset, args.horizon)}_br{br}"):-3]
            c2 = _load(os.path.join(args.sweep_dir, f"c2_{_tag(args.arch, args.dataset, args.horizon)}_br{br}{suffix}.pt"))
            if c2:
                c2_tests.append(c2.get("test_mse"))
                c2_vals.append(c2.get("val_mse"))
        n_seed = len([v for v in c2_tests if v is not None])

        c3_vals = [_load(p).get("val_mse") for p in
                   _variants(args.sweep_dir, "c3", args.arch, args.dataset, args.horizon, br)]
        c3_vals = [v for v in c3_vals if v is not None]
        n_c3 = len(c3_vals)

        c2t_mean = _mean(c2_tests)
        vs = f"{c2t_mean - ref_test:+.4f}" if (c2t_mean is not None and ref_test is not None) else "--"
        c2v_mean, c3v_mean = _mean(c2_vals), _mean(c3_vals)
        if c2v_mean is not None and c3v_mean is not None:
            d = c3v_mean - c2v_mean  # positive => C2 lower => signal wins
            gap = f"{d:+.4f}{'*' if d > 0 else ''}"
        else:
            gap = "--"

        print(f"{br:>7}{fp:>10,}{red:>7.1f}% | {n_seed:>8}{_fmt(c2_tests):>16} | "
              f"{vs:>10} | {n_c3:>7}{_fmt(c2_vals):>10}{_fmt(c3_vals):>16}{gap:>10}")

    print()
    print("C2-C3 positive (*) = the topological signal beat random allocation at the same budget.")
    print("vs ref: negative = the compressed model is BETTER than the full reference.")
    print("n(seed)/n(C3) = 1 means no error bar yet -- do not read a single point as a trend.")
    if not ok_all:
        print("\n!! validity checks failed above — fix those before reading anything into this table.")


if __name__ == "__main__":
    main()
