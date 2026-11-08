"""Write width specs for C2 retrains of finished discovery runs, from their
small result files rather than their checkpoints.

A spec is {arm, arch, dataset, seed, final_widths, params}. Widths come from
the run's JSON record when it has them (bench/train_oto.py), or are replayed
from its decision log otherwise: the dense width of every coupling group
minus each logged removal plus each logged birth. Every spec is checked by
rebuilding the network at its widths and comparing the parameter count with
the one the run recorded; a mismatch aborts.

    python scripts/make_c2_specs.py --arms oto nobank --out results/c2_specs
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.models import build_model  # noqa: E402
from bench.resize import resize_model_to_widths  # noqa: E402
from tsrx.graph.bundle import build_all_bundles  # noqa: E402
from tsrx.graph.groups import discover_groups  # noqa: E402
from tsrx.graph.trace import trace_model  # noqa: E402


def dense_widths(arch, num_classes):
    m = build_model(arch, num_classes, cifar_stem=True)
    bundles = build_all_bundles(discover_groups(trace_model(m.eval(), (torch.zeros(2, 3, 32, 32),))), m)
    return {int(t): int(b.size) for t, b in bundles.items()}


def replay(log_path, start):
    w = dict(start)
    delta = collections.Counter()
    for line in open(log_path, encoding="utf-8"):
        d = json.loads(line)
        if d["action"] in ("pure_prune", "exchange"):
            delta[int(d["prune_tap"])] -= 1
        if d["action"] in ("pure_grow", "exchange"):
            delta[int(d["grow_tap"])] += 1
    for t, v in delta.items():
        w[t] += v
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["oto", "nobank"])
    ap.add_argument("--root", default=str(ROOT / "results" / "ablations"))
    ap.add_argument("--out", default=str(ROOT / "results" / "c2_specs"))
    args = ap.parse_args()
    for cell_dir in sorted(Path(args.root).iterdir()):
        if not cell_dir.is_dir():
            continue
        for arm in args.arms:
            for rec_path in sorted(cell_dir.glob(f"{arm}_s*.json")):
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                if rec.get("kind") != "online":
                    continue
                arch, ds, seed = rec["arch"], rec["dataset"], int(rec["seed"])
                ncls = 10 if ds == "cifar10" else 100
                if rec.get("final_widths"):
                    widths = {int(t): int(v) for t, v in rec["final_widths"].items()}
                    source = "record"
                else:
                    log = rec_path.with_name(f"{rec_path.stem}_decisions.jsonl")
                    widths = replay(log, dense_widths(arch, ncls))
                    source = "decision-log replay"
                m = resize_model_to_widths(build_model(arch, ncls, cifar_stem=True), widths,
                                           torch.zeros(2, 3, 32, 32))
                params = sum(p.numel() for p in m.parameters())
                if params != rec["params"]:
                    raise SystemExit(f"{rec_path}: rebuilt {params:,} != recorded {rec['params']:,}")
                spec = {"arm": arm, "arch": arch, "dataset": ds, "seed": seed, "final_widths": widths,
                        "params": params, "widths_from": source}
                out = Path(args.out) / cell_dir.name / f"{arm}_s{seed}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
                print(f"{out.relative_to(ROOT)}  {params:,} params  ({source}, verified)")


if __name__ == "__main__":
    main()
