"""Aggregate the growth-ablation campaign (scripts/run_ablations.py).

Reads results/ablations/<cell>/*.json (one per finished job) and the online
arms' decision logs, and prints, per cell:

  1. every arm x {online, C2}: n seeds, accuracy mean +- s.d. (best epoch and
     last epoch -- CIFAR's "val" split is its test split, so the last-epoch
     figure is the selection-free one), deployed params, and the change vs
     the same-seed dense reference;
  2. paired same-seed differences between arms (the ablation's answer);
  3. growth mechanics from the decision logs: births, share of births in the
     single most-grown group, and how births' ports were opened.

Partial campaigns are fine: cells/arms/seeds with no record are skipped, and
a paired difference uses only seeds present in both arms.

    python scripts/eval_ablations.py
    python scripts/eval_ablations.py --metric final --markdown reports/ABLATIONS.md
"""

import argparse
import collections
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARM_ORDER = ["ref", "released", "fixed", "off", "random", "cost"]
PAIRS = [  # (a, b, question answered by a - b)
    ("released", "off", "does growth help, as published?"),
    ("fixed", "off", "does growth help, with measured-direction births?"),
    ("fixed", "released", "does keeping the measured birth direction help?"),
    ("fixed", "random", "does the gradient choose the growth site better than chance?"),
    ("fixed", "cost", "does the gradient beat 'grow the cheapest group'?"),
]


def ms(xs):
    if not xs:
        return "--"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}"


def load(root: Path):
    recs = collections.defaultdict(dict)  # cell -> (arm, kind, seed) -> record
    for f in sorted(root.glob("*/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        recs[f.parent.name][(r["arm"], r["kind"], int(r["seed"]))] = r
    return recs


def acc(r, metric):
    v = r.get("final_val_acc") if metric == "final" else r.get("best_val_acc")
    return None if v is None else 100.0 * float(v)


def widths_log_ratio(ckpt_path: Path) -> dict:
    """{group: log2(final width / reference width)} for a discovery checkpoint.
    The reference widths are the first logged decision's pre-edit state: every
    group starts at its dense width, recovered here from the run's own log."""
    import math
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    final = {int(t): v for t, v in (ck.get("final_widths") or ck["discovered_widths"]).items()}
    # Undo the logged edits to recover each group's starting width.
    start = dict(final)
    log = ckpt_path.with_name(f"{ckpt_path.stem}_decisions.jsonl")
    for line in log.open(encoding="utf-8"):
        d = json.loads(line)
        if d["action"] in ("pure_prune", "exchange"):
            start[d["prune_tap"]] += 1
        if d["action"] in ("pure_grow", "exchange"):
            start[d["grow_tap"]] -= 1
    return {t: math.log2(final[t] / start[t]) for t in final}


def cosine(a: dict, b: dict) -> float:
    keys = sorted(set(a) & set(b))
    num = sum(a[k] * b[k] for k in keys)
    den = (sum(a[k] ** 2 for k in keys) ** 0.5) * (sum(b[k] ** 2 for k in keys) ** 0.5)
    return num / den if den else float("nan")


def growth_stats(log: Path):
    births, by_tap, port = 0, collections.Counter(), collections.Counter()
    actions = collections.Counter()
    for line in log.open(encoding="utf-8"):
        d = json.loads(line)
        actions[d["action"]] += 1
        if d["action"] in ("pure_grow", "exchange"):
            births += 1
            by_tap[d["grow_tap"]] += 1
            kinds = set(((d.get("applied") or {}).get("port_init") or {}).values())
            port["grad" if kinds == {"grad"} else ("random" if kinds and "grad" not in kinds else
                                                   ("mixed" if kinds else "unlogged"))] += 1
    top = by_tap.most_common(1)[0][1] / births if births else 0.0
    return births, top, len(by_tap), port, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "results" / "ablations"))
    ap.add_argument("--metric", choices=["best", "final"], default="best")
    ap.add_argument("--markdown", default=None, help="also write the tables to this file")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # tables use Δ/±; Windows pipes default to cp1252
    root = Path(args.root)
    recs = load(root)
    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    emit(f"# Growth ablations ({args.metric}-epoch top-1, %)\n")
    for cell in sorted(recs):
        R = recs[cell]
        seeds = sorted({k[2] for k in R})
        ref = {s: acc(R[("ref", "ref", s)], args.metric) for s in seeds if ("ref", "ref", s) in R}
        emit(f"## {cell}  (seeds present: {seeds})\n")
        emit("| arm | kind | n | top-1 | Δ vs same-seed Ref | params |")
        emit("|---|---|---|---|---|---|")
        for arm in ARM_ORDER:
            for kind in (("ref",) if arm == "ref" else ("online", "c2")):
                rows = [(s, R[(arm, kind, s)]) for s in seeds if (arm, kind, s) in R]
                if not rows:
                    continue
                a = [acc(r, args.metric) for _, r in rows if acc(r, args.metric) is not None]
                d = [acc(r, args.metric) - ref[s] for s, r in rows
                     if s in ref and ref[s] is not None and acc(r, args.metric) is not None]
                p = [r["params"] for _, r in rows if r.get("params")]
                emit(f"| {arm} | {kind} | {len(rows)} | {ms(a)} | "
                     f"{'--' if arm == 'ref' else ms(d)} | {st.mean(p):,.0f} |" if p else
                     f"| {arm} | {kind} | {len(rows)} | {ms(a)} | {ms(d)} | -- |")
        emit("")
        emit("Paired same-seed differences, a − b (points); `wins` = seeds where a > b.\n")
        emit("| a − b | kind | n | mean ± s.d. | wins | question |")
        emit("|---|---|---|---|---|---|")
        for a_arm, b_arm, q in PAIRS:
            for kind in ("online", "c2"):
                common = [s for s in seeds if (a_arm, kind, s) in R and (b_arm, kind, s) in R]
                diffs = [acc(R[(a_arm, kind, s)], args.metric) - acc(R[(b_arm, kind, s)], args.metric)
                         for s in common
                         if acc(R[(a_arm, kind, s)], args.metric) is not None
                         and acc(R[(b_arm, kind, s)], args.metric) is not None]
                if diffs:
                    emit(f"| {a_arm} − {b_arm} | {kind} | {len(diffs)} | {ms(diffs)} | "
                         f"{sum(x > 0 for x in diffs)}/{len(diffs)} | {q} |")
        emit("")
        emit("Growth mechanics (online arms, from decision logs).\n")
        emit("| arm | seed | births | share in top group | groups grown | births' ports | actions |")
        emit("|---|---|---|---|---|---|---|")
        for arm in ARM_ORDER[1:]:
            for s in seeds:
                log = root / cell / f"{arm}_s{s}_decisions.jsonl"
                if (arm, "online", s) not in R or not log.exists():
                    continue
                b, top, ng, port, actions = growth_stats(log)
                emit(f"| {arm} | {s} | {b} | {top:.0%} | {ng} | {dict(port)} | {dict(actions)} |")
        emit("")

        # Discovery stability across seeds (R3): how similar are the exported
        # width vectors, as log2(width / reference width) per group?
        rows = []
        for arm in ARM_ORDER[1:]:
            vecs = {}
            for s in seeds:
                ckp = root / cell / f"{arm}_s{s}.pt"
                if (arm, "online", s) not in R or not ckp.exists():
                    continue
                vecs[s] = widths_log_ratio(ckp)
            if len(vecs) >= 2:
                ss = sorted(vecs)
                cos = [cosine(vecs[a], vecs[b]) for i, a in enumerate(ss) for b in ss[i + 1:]]
                top = [max(vecs[s], key=lambda t: vecs[s][t]) for s in ss]
                rows.append(f"| {arm} | {len(ss)} | {st.mean(cos):.3f} (min {min(cos):.3f}) | "
                            f"{'same' if len(set(top)) == 1 else 'differs'}: {top} |")
        if rows:
            emit("Discovery stability across seeds: cosine similarity of log2(width/ref) vectors "
                 "(final widths), and the most-grown group per seed.\n")
            emit("| arm | seeds | mean pairwise cosine | most-grown group |")
            emit("|---|---|---|---|")
            for r in rows:
                emit(r)
            emit("")

    if args.markdown:
        Path(args.markdown).write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
