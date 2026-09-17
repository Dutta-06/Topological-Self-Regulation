"""Paper-ready time-series results tables, built from checkpoints on disk.

Mirrors the vision tables' 3-arm structure (reference / TSR-X / C2) plus the
C3 random-reallocation control, one row per (architecture, dataset) at that
cell's KNEE, and a full per-budget frontier for the appendix.

Knee rule (identical to scripts/campaign.sh::detect_knee, so the rows land
exactly where the Stage-2 seeds were run): the tightest budget whose seed-42
C2 test MSE is within TOL of the seed-42 reference. A fixed tolerance, declared
up front -- never argmin, which would be selecting on noise.

Conventions that must survive into the paper's caption:
  * TSR-X is reported at its FINAL (search-converged) architecture
    (`final_test_mse`), not its best-val checkpoint, which sits at a partial,
    larger architecture and is not comparable to C2/C3.
  * mean +/- std across whatever seeds exist; n is printed.
  * a row failing a validity check is marked with a dagger, never dropped
    silently (C2 params != TSR-X final params, C3 off target by >1%, or
    dormancy leaked).

Usage:
    python -m analysis.paper_tables                      # markdown to stdout
    python -m analysis.paper_tables --latex-dir results  # also writes .tex (booktabs, graphicx)
"""

import argparse
import glob
import os
import re
import statistics

import torch

ARCHS = [("patchtst", "PatchTST"), ("itransformer", "iTransformer"),
         ("tsmixer", "TSMixer"), ("tcn", "TCN")]
DATASETS = [("weather", "Weather"), ("electricity", "Electricity"), ("traffic", "Traffic")]


# ----------------------------------------------------------------- loading

def _load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _tag(arch, ds, h):
    # TCN kept the original untagged sweep naming (see scripts/campaign.sh).
    return f"{ds}_h{h}" if arch == "tcn" else f"{arch}_{ds}_h{h}"


def _budgets(sweep, arch, ds, h):
    t = _tag(arch, ds, h)
    out = set()
    for p in glob.glob(os.path.join(sweep, f"tsrx_{t}_br*.pt")):
        m = re.match(rf"^tsrx_{re.escape(t)}_br([0-9.]+)(?:_s\d+)?\.pt$", os.path.basename(p))
        if m:
            out.add(m.group(1))
    return sorted(out, key=float)


def _variants(sweep, arm, arch, ds, h, br, draws=False):
    """(suffix, path) for the base run, every _s<seed> run and, for C3, every
    _c<draw> run. All are samples of the same arm at the same budget."""
    base = f"{arm}_{_tag(arch, ds, h)}_br{br}"
    found = [("", os.path.join(sweep, base + ".pt"))]
    found += [(os.path.basename(p)[len(base):-3], p)
              for p in sorted(glob.glob(os.path.join(sweep, base + "_s*.pt")))]
    if draws:
        found += [(os.path.basename(p)[len(base):-3], p)
                  for p in sorted(glob.glob(os.path.join(sweep, base + "_c*.pt")))]
    return [(s, p) for s, p in found if os.path.exists(p)]


def _references(ref_dir, arch, ds, h):
    base = os.path.join(ref_dir, f"{arch}_{ds}_h{h}")
    paths = [base + ".pt"] + sorted(glob.glob(base + "_s*.pt"))
    return [ck for ck in (_load(p) for p in paths if os.path.exists(p)) if ck]


# --------------------------------------------------------------- statistics

def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else None, len(vals))


def _human(n):
    if n is None:
        return "--"
    return f"{n/1e6:.2f}M" if n >= 999_500 else f"{n/1e3:.1f}K"


def _fmt(st, latex=False, bold=False):
    if st is None:
        return "--"
    mean, std, _ = st
    if latex:
        core = f"{mean:.4f}" if std is None else f"{mean:.4f}_{{\\pm {std:.4f}}}"
        return f"$\\mathbf{{{core}}}$" if bold else f"${core}$"
    core = f"{mean:.4f}" if std is None else f"{mean:.4f}±{std:.4f}"
    return f"**{core}**" if bold else core


# ------------------------------------------------------------------ one cell

def _cell(sweep, ref_dir, arch, ds, h, br, metric):
    """Every arm's statistic for one (arch, dataset, budget)."""
    refs = _references(ref_dir, arch, ds, h)
    ref_params = refs[0].get("params") if refs else None
    issues, tx_vals, tx_params, fallback = [], [], [], False
    c2_vals, c2_params = [], []

    for suffix, p in _variants(sweep, "tsrx", arch, ds, h, br):
        tx = _load(p)
        if tx is None:
            issues.append(f"unreadable tsrx{suffix}")
            continue
        v = tx.get(f"final_test_{metric}")
        if v is None:
            v, fallback = tx.get(f"test_{metric}"), True
        tx_vals.append(v)
        fp = tx.get("final_params")
        tx_params.append(fp)
        if tx.get("max_port_magnitude") not in (None, 0.0):
            issues.append(f"dormancy leaked{suffix}")

        c2 = _load(os.path.join(sweep, f"c2_{_tag(arch, ds, h)}_br{br}{suffix}.pt"))
        if c2 is not None:
            c2_vals.append(c2.get(f"test_{metric}"))
            c2_params.append(c2.get("params"))
            if fp is not None and c2.get("params") != fp:
                issues.append(f"C2 built wrong architecture{suffix}")

    c3_vals = []
    for suffix, p in _variants(sweep, "c3", arch, ds, h, br, draws=True):
        c3 = _load(p)
        if c3 is None:
            continue
        c3_vals.append(c3.get(f"test_{metric}"))
        rel = c3.get("param_match_rel_error")
        if rel is None or rel > 0.01:
            issues.append(f"C3 off target{suffix}")

    # TSR-X's final_params is the source of truth for the discovered size; a
    # C2 that disagrees with it is the validity failure, not the reference.
    params = (statistics.mean([x for x in tx_params if x]) if any(tx_params)
              else statistics.mean([x for x in c2_params if x]) if any(c2_params) else None)
    return {
        "ref": _stat([r.get(f"test_{metric}") for r in refs]),
        "tsrx": _stat(tx_vals),
        "c2": _stat(c2_vals),
        "c3": _stat(c3_vals),
        "ref_params": ref_params,
        "params": params,
        "reduction": (1 - params / ref_params) * 100 if (params and ref_params) else None,
        "issues": issues,
        "tsrx_fallback": fallback,
    }


def _knee(sweep, ref_dir, arch, ds, h, tol):
    refs = _references(ref_dir, arch, ds, h)
    if not refs or refs[0].get("test_mse") is None:
        return None
    ref = refs[0]["test_mse"]  # seed-42 reference, as the campaign used
    best = None
    for br in _budgets(sweep, arch, ds, h):
        c2 = _load(os.path.join(sweep, f"c2_{_tag(arch, ds, h)}_br{br}.pt"))
        t = c2.get("test_mse") if c2 else None
        if t is not None and (t - ref) / ref <= tol:
            if best is None or float(br) < float(best):
                best = br
    return best


# -------------------------------------------------------------------- tables

def _best_arm(cell):
    means = {k: cell[k][0] for k in ("ref", "tsrx", "c2", "c3") if cell[k]}
    return min(means, key=means.get) if means else None


def main_rows(sweep, ref_dir, h, tol, metric):
    rows = []
    for arch, aname in ARCHS:
        for ds, dname in DATASETS:
            br = _knee(sweep, ref_dir, arch, ds, h, tol)
            if br is None:
                continue
            cell = _cell(sweep, ref_dir, arch, ds, h, br, metric)
            rows.append((aname, dname, br, cell))
    return rows


def render_main_md(rows, metric):
    out = [f"| Model | Dataset | Params (ref → ours) | Reduction | Reference | TSR-X | **C2 (ours)** | C3 (random) | Δ C2 vs ref | n |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for aname, dname, br, c in rows:
        best = _best_arm(c)
        d = (f"{(c['c2'][0] - c['ref'][0]) / c['ref'][0] * 100:+.2f}%"
             if c["c2"] and c["ref"] else "--")
        flag = " †" if c["issues"] else ""
        tsrx = _fmt(c["tsrx"], bold=best == "tsrx") + (" ‡" if c["tsrx_fallback"] else "")
        n = "/".join(str(c[k][2]) if c[k] else "0" for k in ("ref", "tsrx", "c2", "c3"))
        out.append(
            f"| {aname}{flag} | {dname} | {_human(c['ref_params'])} → {_human(c['params'])} | "
            f"{c['reduction']:.1f}% | {_fmt(c['ref'], bold=best == 'ref')} | {tsrx} | "
            f"{_fmt(c['c2'], bold=best == 'c2')} | {_fmt(c['c3'], bold=best == 'c3')} | {d} | {n} |"
            if c["reduction"] is not None else
            f"| {aname}{flag} | {dname} | -- | -- | {_fmt(c['ref'])} | {tsrx} | {_fmt(c['c2'])} | "
            f"{_fmt(c['c3'])} | {d} | {n} |")
    return "\n".join(out)


def render_main_tex(rows, metric, tol, h):
    M = metric.upper()
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llrrccccr@{}}",
        r"\toprule",
        rf"& & \multicolumn{{2}}{{c}}{{Parameters}} & \multicolumn{{4}}{{c}}{{Test {M} ($\downarrow$)}} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}",
        r"Model & Dataset & Ref $\to$ Ours & Red. & Reference & TSR-X & \textbf{C2 (ours)} & C3 (random) & $\Delta_{\mathrm{C2}}$ \\",
        r"\midrule",
    ]
    prev = None
    for aname, dname, br, c in rows:
        if prev is not None and aname != prev:
            lines.append(r"\midrule")
        best = _best_arm(c)
        d = (f"${(c['c2'][0] - c['ref'][0]) / c['ref'][0] * 100:+.2f}\\%$"
             if c["c2"] and c["ref"] else "--")
        name = (aname + r"$^\dagger$") if c["issues"] else aname
        tsrx = _fmt(c["tsrx"], latex=True, bold=best == "tsrx") + (r"$^\ddagger$" if c["tsrx_fallback"] else "")
        red = f"{c['reduction']:.1f}\\%" if c["reduction"] is not None else "--"
        params = (f"{_human(c['ref_params'])} $\\to$ {_human(c['params'])}"
                  if c["params"] else "--")
        lines.append(
            f"{name if aname != prev else ''} & {dname} & {params} & {red} & "
            f"{_fmt(c['ref'], latex=True, bold=best == 'ref')} & {tsrx} & "
            f"{_fmt(c['c2'], latex=True, bold=best == 'c2')} & "
            f"{_fmt(c['c3'], latex=True, bold=best == 'c3')} & {d} \\\\")
        prev = aname
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        (rf"\caption{{Time-series forecasting at horizon {h}, one row per model and dataset at its "
         rf"\emph{{knee}}: the tightest parameter budget whose C2 test {M} stays within "
         rf"{tol*100:.0f}\% of the reference (declared before inspecting results). "
         rf"\textbf{{C2}} retrains the architecture discovered by TSR-X from scratch and is the claim; "
         rf"\textbf{{C3}} retrains a \emph{{random}} allocation matched to the same parameter count "
         rf"($\le$1\%). TSR-X is reported at its final, search-converged architecture. "
         rf"Values are mean$_{{\pm\mathrm{{std}}}}$ across available seeds; best mean per row in bold. "
         rf"$\Delta_{{\mathrm{{C2}}}}$ is the relative change of C2 against the reference. "
         rf"$^\dagger$Row failed a validity check. "
         rf"$^\ddagger$Final-architecture metric unavailable; best-validation value shown. "
         rf"TCN is not a competitive forecasting baseline and is included as a simple non-SOTA reference.}}"),
        rf"\label{{tab:ts-main-{metric}}}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def frontier(sweep, ref_dir, h, tol):
    blocks = []
    for arch, aname in ARCHS:
        for ds, dname in DATASETS:
            brs = _budgets(sweep, arch, ds, h)
            if not brs:
                continue
            knee = _knee(sweep, ref_dir, arch, ds, h, tol)
            refs = _references(ref_dir, arch, ds, h)
            ref = _stat([r.get("test_mse") for r in refs])
            rows = []
            for br in sorted(brs, key=float, reverse=True):
                c = _cell(sweep, ref_dir, arch, ds, h, br, "mse")
                gap = (c["c2"][0] - c["c3"][0]) if (c["c2"] and c["c3"]) else None
                rows.append((br, c, gap, br == knee))
            blocks.append((aname, dname, ref, rows))
    return blocks


def render_frontier_md(blocks):
    out = []
    for aname, dname, ref, rows in blocks:
        out.append(f"\n**{aname} — {dname}**  (reference test MSE {_fmt(ref)})\n")
        out.append("| Budget | Params | Reduction | TSR-X | C2 (ours) | C3 (random) | C2 − C3 |")
        out.append("|---|---|---|---|---|---|---|")
        for br, c, gap, is_knee in rows:
            g = f"{gap:+.4f}" if gap is not None else "--"
            red = f"{c['reduction']:.1f}%" if c["reduction"] is not None else "--"
            mark = " ★" if is_knee else ""
            dag = " †" if c["issues"] else ""
            out.append(f"| {br}{mark}{dag} | {_human(c['params'])} | {red} | {_fmt(c['tsrx'])} | "
                       f"{_fmt(c['c2'])} | {_fmt(c['c3'])} | {g} |")
    return "\n".join(out)


def render_frontier_tex(blocks, tol, h):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\begin{tabular}{@{}rrrcccr@{}}", r"\toprule",
             r"Budget & Params & Red. & TSR-X & C2 (ours) & C3 (random) & C2$-$C3 \\"]
    for aname, dname, ref, rows in blocks:
        lines += [r"\midrule",
                  rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{aname}}} --- {dname} "
                  rf"(reference {_fmt(ref, latex=True)})}} \\",
                  r"\midrule"]
        for br, c, gap, is_knee in rows:
            g = f"${gap:+.4f}$" if gap is not None else "--"
            red = f"{c['reduction']:.1f}\\%" if c["reduction"] is not None else "--"
            b = br + (r"$^\star$" if is_knee else "") + (r"$^\dagger$" if c["issues"] else "")
            lines.append(f"{b} & {_human(c['params'])} & {red} & {_fmt(c['tsrx'], latex=True)} & "
                         f"{_fmt(c['c2'], latex=True)} & {_fmt(c['c3'], latex=True)} & {g} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              (rf"\caption{{Full compression frontier at horizon {h} (test MSE, lower is better). "
               rf"$^\star$marks the knee (tightest budget with C2 within {tol*100:.0f}\% of the reference). "
               rf"C2$-$C3 $<0$ means the TSR-X-discovered allocation beat random allocation at the same "
               rf"parameter count. $^\dagger$Validity check failed.}}"),
              r"\label{tab:ts-frontier}", r"\end{table}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results/ts_sweep")
    ap.add_argument("--ref-dir", default="results/ts_reference")
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--tol", type=float, default=0.03,
                    help="knee tolerance, relative to reference test MSE (must match the campaign)")
    ap.add_argument("--latex-dir", default=None, help="write ts_table_*.tex here")
    args = ap.parse_args()

    main_mse = main_rows(args.sweep_dir, args.ref_dir, args.horizon, args.tol, "mse")
    main_mae = main_rows(args.sweep_dir, args.ref_dir, args.horizon, args.tol, "mae")
    blocks = frontier(args.sweep_dir, args.ref_dir, args.horizon, args.tol)
    if not main_mse and not blocks:
        raise SystemExit(f"no sweep checkpoints under {args.sweep_dir}")

    print(f"## Main results — test MSE at each knee (horizon {args.horizon}, tolerance {args.tol*100:.0f}%)\n")
    print(render_main_md(main_mse, "mse"))
    print(f"\n## Main results — test MAE at the same knees\n")
    print(render_main_md(main_mae, "mae"))
    print("\n## Appendix — full frontier (test MSE)")
    print(render_frontier_md(blocks))
    print("\nn = seeds for reference / TSR-X / C2 / C3 (C3 also counts extra random draws). "
          "† failed validity check. ‡ TSR-X best-val value shown (no final-architecture metric). "
          "★ knee.")

    issues = [(a, d, c["issues"]) for a, d, _, c in main_mse if c["issues"]]
    if issues:
        print("\n!! VALIDITY FAILURES — these rows must not be reported as-is:")
        for a, d, iss in issues:
            print(f"   {a}/{d}: {', '.join(sorted(set(iss)))}")

    if args.latex_dir:
        os.makedirs(args.latex_dir, exist_ok=True)
        for name, body in [("ts_table_main_mse.tex", render_main_tex(main_mse, "mse", args.tol, args.horizon)),
                           ("ts_table_main_mae.tex", render_main_tex(main_mae, "mae", args.tol, args.horizon)),
                           ("ts_table_frontier.tex", render_frontier_tex(blocks, args.tol, args.horizon))]:
            with open(os.path.join(args.latex_dir, name), "w") as f:
                f.write(body + "\n")
        print(f"\nLaTeX written to {args.latex_dir}/ts_table_{{main_mse,main_mae,frontier}}.tex "
              f"(\\input them; needs booktabs)")


if __name__ == "__main__":
    main()
