"""Training compute of every arm, replayed from the archived decision logs.

Answers R2's Q2: what does each arm cost to TRAIN, counting the candidate
bank and the editing trajectory, not just the exported model?

For each archived vision discovery run (results/tsrx/*_two_regime.pt + its
_decisions.jsonl) this

  1. replays the width of every editable group across training from the
     decision log (each committed prune / grow / exchange moves one index);
  2. prices a training step as 3x the forward FLOPs of the network actually
     trained at that step -- for the online arm that is the deployed widths
     PLUS the k dormant candidates per editable group, which are computed in
     full even though their ports are zero;
  3. integrates over the run and compares with the other arms:
        Ref      100 epochs of the dense model
        TSR      the online discovery run (one 100-epoch run)
        C2       TSR + a 100-epoch retrain at the discovered widths
        prune/ft Ref + 40 fine-tuning epochs at the pruned widths
        prune/s  Ref + 100 epochs from scratch at the pruned widths
     (pruned widths from results/pruned/*_l1_*.json where archived).

Forward FLOPs use the paper's definition (conv and linear multiply-adds x2,
as in scripts/eval_*_triangulation.py::count_flops), computed analytically
from group widths and validated against that hook counter on rebuilt models.
Elementwise work (BN, activations, the per-step sensing reductions) and the
evaluation passes are excluded for every arm alike.

    python scripts/eval_training_compute.py
    python scripts/eval_training_compute.py --cells resnet18_cifar100 --markdown reports/TRAINING_COMPUTE.md
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.models import build_model  # noqa: E402
from bench.resize import resize_model_to_widths  # noqa: E402
from tsrx.graph.bundle import build_all_bundles  # noqa: E402
from tsrx.graph.groups import discover_groups  # noqa: E402
from tsrx.graph.trace import trace_model  # noqa: E402

DATASETS = {  # name -> (classes, input side, train images)
    "cifar10": (10, 32, 50_000),
    "cifar100": (100, 32, 50_000),
    "tiny_imagenet": (200, 64, 100_000),
    "imagenet100": (100, 224, 126_689),
}
ARCH_OF_KEY = {"resnet18": "resnet18", "vgg16bn": "vgg16_bn",
               "mobilenetv2": "mobilenet_v2", "efficientnet_b0": "efficientnet_b0"}


def count_flops_hooks(model: nn.Module, side: int) -> int:
    """The paper's counter (scripts/eval_*_triangulation.py::count_flops), CPU."""
    flops = 0

    def conv_hook(m, inp, out):
        nonlocal flops
        in_c = inp[0].shape[1]
        out_c, oh, ow = out.shape[1:]
        kh, kw = m.kernel_size
        flops += 2 * (kh * kw * (in_c // m.groups)) * out_c * oh * ow

    def linear_hook(m, inp, out):
        nonlocal flops
        flops += 2 * m.in_features * m.out_features

    hooks = [m.register_forward_hook(conv_hook if isinstance(m, nn.Conv2d) else linear_hook)
             for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(1, 3, side, side))
    for h in hooks:
        h.remove()
    return flops


class AnalyticFlops:
    """Forward FLOPs as a function of group widths.

    One forward pass of the reference records each conv/linear layer's kernel
    and output resolution; the bundles say which group indexes each layer's
    output axis (producer / self-coupled depthwise) and input axis (consumer,
    with multiplicity). Every other axis is task-fixed."""

    def __init__(self, model: nn.Module, bundles: dict, editable: set, side: int):
        self.layers = []
        geom = {}

        def rec(name):
            def hook(m, inp, out):
                if isinstance(m, nn.Conv2d):
                    geom[name] = (m.kernel_size[0] * m.kernel_size[1], out.shape[2] * out.shape[3],
                                  m.groups, inp[0].shape[1], out.shape[1])
                else:
                    geom[name] = (1, 1, 1, m.in_features, m.out_features)
            return hook

        hooks = [m.register_forward_hook(rec(n)) for n, m in model.named_modules()
                 if isinstance(m, (nn.Conv2d, nn.Linear))]
        with torch.no_grad():
            model.eval()(torch.zeros(1, 3, side, side))
        for h in hooks:
            h.remove()

        out_tap, in_tap = {}, {}
        for tap, b in bundles.items():
            if tap not in editable:
                continue
            for s in b.producer_slots:
                if s.param_name == "weight":
                    out_tap.setdefault(s.module_name, tap)
            for s in b.consumer_slots:
                if s.param_name == "weight":
                    in_tap.setdefault(s.module_name, (tap, s.multiplicity))
        for name, (kk, hw, groups, in_c, out_c) in geom.items():
            depthwise = groups > 1 and groups == in_c
            self.layers.append((name, kk, hw, groups, in_c, out_c, depthwise,
                                out_tap.get(name), in_tap.get(name)))

    def __call__(self, widths: dict) -> int:
        tot = 0
        for _, kk, hw, groups, in_c, out_c, dw, otap, itap in self.layers:
            out = widths[otap] if otap is not None else out_c
            if dw:  # one input channel per output channel, whatever the width
                tot += 2 * kk * out * hw
                continue
            inc = widths[itap[0]] * itap[1] if itap is not None else in_c
            tot += 2 * kk * (inc // groups) * out * hw
        return tot


def last_run(log: Path) -> list:
    """Decision records of the LAST run in the log. Before the driver opened
    its log with "w", an interrupted run and its restart were appended to one
    file; a step counter that goes backwards marks where a new run began."""
    records = [json.loads(line) for line in log.open(encoding="utf-8")]
    start = 0
    for i in range(1, len(records)):
        if records[i]["step"] < records[i - 1]["step"]:
            start = i
    return records[start:]


def replay(log: Path, ref_widths: dict):
    """[(step, widths-after-step, deployed)] for every committed edit of the last run."""
    w = dict(ref_widths)
    states = []
    for d in last_run(log):
        a = d["action"]
        if a in ("pure_prune", "exchange"):
            w[d["prune_tap"]] -= 1
        if a in ("pure_grow", "exchange"):
            w[d["grow_tap"]] += 1
        if a != "none":
            states.append((int(d["step"]), dict(w), int(d["deployed_after"])))
    return states


def analyse(key: str, ds: str, check: bool, verbose=print) -> dict:
    tsrx_ckpt = ROOT / "results" / "tsrx" / f"{key}_{ds}_two_regime.pt"
    log = tsrx_ckpt.with_name(f"{tsrx_ckpt.stem}_decisions.jsonl")
    ck = torch.load(tsrx_ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    classes, side, n_train = DATASETS[ds]
    model = build_model(a["arch"], classes, cifar_stem=a["cifar_stem"])
    bundles = build_all_bundles(discover_groups(trace_model(model.eval(), (torch.zeros(2, 3, side, side),))), model)
    editable = {t for t, b in bundles.items() if b.size and b.producer_slots and b.consumer_slots}
    ref_w = {t: bundles[t].size for t in editable}
    k = int(a["k"])
    F = AnalyticFlops(model, bundles, editable, side)

    spe = n_train // int(a["batch_size"])  # drop_last=True in every vision loader
    total = int(a["epochs"]) * spe
    states = replay(log, ref_w)
    run = last_run(log)
    last_step = int(run[-1]["step"]) if run else 0  # last update of any kind, no-ops included
    if last_step < total - 2 * int(a["update_interval"]):
        return {"cell": f"{key}_{ds}", "incomplete": True,
                "note": f"log ends at step {last_step} of {total}; archived run is not the full discovery run"}
    disc = {int(t): v for t, v in ck["discovered_widths"].items()}

    # Validation: the replay must land on the archived checkpoint's widths at
    # its saved (best-val) epoch, and on the logged deployed count.
    best_step_end = (int(ck["epoch"]) + 1) * spe
    at_best = dict(ref_w)
    for s, w, _ in states:
        if s < best_step_end:
            at_best = w
    replay_ok = at_best == disc
    final_w = states[-1][1] if states else dict(ref_w)

    f_ref = F(ref_w)
    f_disc = F(disc)
    f_final = F(final_w)
    inflate = lambda w: {t: v + k for t, v in w.items()}  # noqa: E731

    # Integrate the online run: widths change only after a logged decision.
    tsr_fl, tsr_deployed_fl, cur, prev = 0, 0, dict(ref_w), 0
    for s, w, _ in states:
        n = (s + 1) - prev
        tsr_fl += n * F(inflate(cur)); tsr_deployed_fl += n * F(cur)
        cur, prev = w, s + 1
    tsr_fl += (total - prev) * F(inflate(cur)); tsr_deployed_fl += (total - prev) * F(cur)

    hook_check = None
    if check:
        ex = torch.zeros(2, 3, side, side)
        pairs = {"ref": (ref_w, f_ref), "discovered": (disc, f_disc),
                 "final+k": (inflate(final_w), F(inflate(final_w)))}
        hook_check = {}
        for name, (w, fa) in pairs.items():
            m = resize_model_to_widths(build_model(a["arch"], classes, cifar_stem=a["cifar_stem"]),
                                       {str(t): v for t, v in w.items()}, ex)
            fh = count_flops_hooks(m, side)
            hook_check[name] = (fa, fh)

    out = {"cell": f"{key}_{ds}", "k": k, "steps": total, "decisions": len(states),
           "replay_matches_checkpoint": replay_ok,
           "flops_fwd": {"ref": f_ref, "discovered": f_disc, "final": f_final},
           "train_flops_units": {  # forward-FLOP-steps; x3 x batch for absolute training FLOPs
               "ref": total * f_ref,
               "tsr": tsr_fl,
               "tsr_without_candidates": tsr_deployed_fl,
               "c2": tsr_fl + total * f_disc,
           },
           "hook_check": hook_check,
           "batch_size": int(a["batch_size"])}

    pj = sorted(p for p in (ROOT / "results" / "pruned").glob(f"{key}_{ds}_l1_*.json")
                if not p.name.endswith(".events.json"))
    for p in pj:
        rec = json.loads(p.read_text(encoding="utf-8"))
        pw = {int(t): v for t, v in rec["discovered_widths"].items()}
        f_pr = F(pw)
        out["flops_fwd"]["l1_pruned"] = f_pr
        out["train_flops_units"]["l1_finetune"] = total * f_ref + int(0.4 * total) * f_pr
        out["train_flops_units"]["l1_scratch"] = total * f_ref + total * f_pr
        if rec.get("train_seconds"):
            out["l1_scratch_train_seconds"] = rec["train_seconds"]
        break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None, help="e.g. resnet18_cifar100 (default: all archived)")
    ap.add_argument("--no-check", action="store_true", help="skip the hook-counter validation")
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--json", default=str(ROOT / "results" / "training_compute.json"))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Δ/ℓ in the table; Windows pipes default to cp1252

    cells = []
    for p in sorted((ROOT / "results" / "tsrx").glob("*_two_regime.pt")):
        stem = p.stem.replace("_two_regime", "")
        for ds in DATASETS:
            if stem.endswith("_" + ds):
                key = stem[: -len(ds) - 1]
                if key in ARCH_OF_KEY and (args.cells is None or stem in args.cells):
                    cells.append((key, ds))

    results, lines = [], []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit("# Training compute by arm (forward-FLOP x steps, relative to the dense reference)\n")
    emit("Ref = 1.00. TSR counts the k dormant candidates per group; C2 = TSR + retrain; "
         "pruning pipelines include the dense reference they start from.\n")
    emit("| cell | replay ok | ΔF exported | TSR | TSR w/o cand. | C2 | ℓ1 fine-tune | ℓ1 scratch | hook check |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for key, ds in cells:
        r = analyse(key, ds, check=not args.no_check)
        results.append(r)
        if r.get("incomplete"):
            emit(f"| {r['cell']} | -- | -- | -- | -- | -- | -- | -- | {r['note']} |")
            continue
        u, f = r["train_flops_units"], r["flops_fwd"]
        rel = {n: v / u["ref"] for n, v in u.items()}
        chk = "--"
        if r["hook_check"]:
            chk = "ok" if all(a == h for a, h in r["hook_check"].values()) else \
                "MISMATCH " + str({n: (a, h) for n, (a, h) in r["hook_check"].items() if a != h})
        g = lambda n: f"{rel[n]:.2f}" if n in rel else "--"  # noqa: E731
        emit(f"| {r['cell']} | {'yes' if r['replay_matches_checkpoint'] else 'NO'} | "
             f"{100 * (f['discovered'] / f['ref'] - 1):+.1f}% | {g('tsr')} | {g('tsr_without_candidates')} | "
             f"{g('c2')} | {g('l1_finetune')} | {g('l1_scratch')} | {chk} |")

    Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    emit(f"\n(raw numbers: {args.json})")
    if args.markdown:
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
