"""Growth-ablation campaign (rebuttal): does gradient-directed growth matter?

For each cell and seed this trains, under one shared recipe:

  ref        dense reference                              (bench.train_reference)
  released   TSR-X as published: growth=grad, prune_first (bench.train_tsrx)
  fixed      growth=grad, grow_first -- births keep the measured direction
  off        growth=off: removal-only TSR-X, same schedule/scores/bank
  random     growth group drawn at random, released cadence (grow_first)
  cost       growth group = cheapest feasible one (grow_first)

and, for the arms in --c2-arms, the static-matched retrain (C2) of each
discovered architecture from its converged widths (bench.train_static_matched).

Every job is resumable: an output whose checkpoint carries `complete` is
skipped, an interrupted one is retrained from scratch. Each finished job
writes a JSON summary next to its checkpoint; aggregate with
scripts/eval_ablations.py.

Jobs are grouped into chains (discovery -> its C2; or ref) and chains run
`--concurrent` at a time, ordered seed-major so that a complete first seed
across every arm arrives before any second seed.

Examples:
    python scripts/run_ablations.py --dry-run
    python scripts/run_ablations.py --cells resnet18/cifar100 --seeds 42 --concurrent 2
    python scripts/run_ablations.py --arms released off --seeds 43 44 --out-root /kaggle/working/ablations
"""

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
ARCH_KEY = {"resnet18": "resnet18", "vgg16_bn": "vgg16bn",
            "mobilenet_v2": "mobilenetv2", "efficientnet_b0": "efficientnet_b0"}
ARMS = {  # name -> (--growth, --exchange-order)
    "released": ("grad", "prune_first"),
    "fixed": ("grad", "grow_first"),
    "off": ("off", "prune_first"),
    "random": ("random", "grow_first"),
    "cost": ("cost", "grow_first"),
}
_print_lock = threading.Lock()


def say(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(torch.load(path, map_location="cpu", weights_only=False).get("complete"))
    except Exception:
        return False


def summarize(path: Path, meta: dict, seconds: float) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = dict(meta)
    out.update({
        "best_val_acc": float(ck.get("best_val_acc", ck.get("val_acc", float("nan")))),
        "final_val_acc": ck.get("final_val_acc"),
        "best_epoch": int(ck.get("epoch", -1)),
        "params": int(ck.get("final_params", ck.get("params", 0)) or 0),
        "baseline_params": ck.get("baseline_params"),
        "port_init_counts": ck.get("port_init_counts"),
        "max_port_magnitude": ck.get("max_port_magnitude"),
        "wall_seconds": seconds,
        "train_seconds": ck.get("train_seconds"),
        "checkpoint": path.name,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })
    path.with_suffix(".json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_job(out: Path, cmd: list, meta: dict, dry_run: bool) -> bool:
    if done(out):
        say(f"skip  {out.parent.name}/{out.name} (complete)")
        return True
    say(f"start {out.parent.name}/{out.name}")
    if dry_run:
        say("      $ " + " ".join(map(str, cmd)))
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    logfile = out.with_suffix(".log")
    with logfile.open("w", encoding="utf-8") as fh:
        res = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if res.returncode != 0 or not out.exists():
        say(f"FAIL  {out.parent.name}/{out.name} (exit {res.returncode}; see {logfile})")
        return False
    ck = torch.load(out, map_location="cpu", weights_only=False)
    if not ck.get("complete"):  # C2 driver does not set it; a zero exit means it finished
        ck["complete"] = True
        torch.save(ck, out)
    s = summarize(out, meta, time.time() - t0)
    say(f"done  {out.parent.name}/{out.name} best={s['best_val_acc']:.4f} "
        f"final={s['final_val_acc'] if s['final_val_acc'] is not None else float('nan'):.4f} "
        f"params={s['params']:,} ({s['wall_seconds'] / 3600:.2f} h)")
    return True


def build_chains(args) -> list:
    """Seed-major list of chains; each chain is a list of (out, cmd, meta)."""
    py = sys.executable
    common = ["--epochs", str(args.epochs), "--batch-size", "128",
              "--num-workers", str(args.num_workers), "--data-root", args.data_root]
    chains = []
    for seed in args.seeds:
        for cell in args.cells:
            arch, ds = cell.split("/")
            cdir = Path(args.out_root) / f"{ARCH_KEY[arch]}_{ds}"
            if "ref" in args.arms:
                out = cdir / f"ref_s{seed}.pt"
                chains.append([(out,
                                [py, "-m", "bench.train_reference", "--arch", arch, "--dataset", ds,
                                 "--seed", str(seed), "--out", str(out)] + common,
                                {"arm": "ref", "arch": arch, "dataset": ds, "seed": seed, "kind": "ref"})])
            for arm in [a for a in args.arms if a in ARMS]:
                growth, order = ARMS[arm]
                disc = cdir / f"{arm}_s{seed}.pt"
                chain = [(disc,
                          [py, "-m", "bench.train_tsrx", "--arch", arch, "--dataset", ds,
                           "--seed", str(seed), "--budget-ratio", str(args.budget_ratio),
                           "--growth", growth, "--exchange-order", order, "--out", str(disc)] + common,
                          {"arm": arm, "arch": arch, "dataset": ds, "seed": seed, "kind": "online",
                           "growth": growth, "exchange_order": order})]
                if arm in args.c2_arms:
                    c2 = cdir / f"{arm}_c2_s{seed}.pt"
                    chain.append((c2,
                                  [py, "-m", "bench.train_static_matched", "--tsrx-checkpoint", str(disc),
                                   "--seed", str(seed), "--out", str(c2)] + common,
                                  {"arm": arm, "arch": arch, "dataset": ds, "seed": seed, "kind": "c2",
                                   "growth": growth, "exchange_order": order}))
                chains.append(chain)
    return chains


def run_chain(chain: list, dry_run: bool) -> None:
    for out, cmd, meta in chain:
        if not run_job(out, cmd, meta, dry_run):
            say(f"chain stopped after failure of {out.name}")
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="*", default=["resnet18/cifar100", "vgg16_bn/cifar100"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--arms", nargs="*", default=["ref", "released", "fixed", "off", "random", "cost"],
                    choices=["ref"] + list(ARMS))
    ap.add_argument("--c2-arms", nargs="*", default=["released", "fixed", "off"], choices=list(ARMS))
    ap.add_argument("--budget-ratio", type=float, default=0.85)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--concurrent", type=int, default=2, help="chains trained at the same time on this GPU")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "ablations"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    chains = build_chains(args)
    say(f"{len(chains)} chains, {sum(len(c) for c in chains)} jobs, concurrent={args.concurrent}")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrent)) as pool:
        for f in [pool.submit(run_chain, c, args.dry_run) for c in chains]:
            f.result()
    say("all chains finished; aggregate with: python scripts/eval_ablations.py")


if __name__ == "__main__":
    main()
