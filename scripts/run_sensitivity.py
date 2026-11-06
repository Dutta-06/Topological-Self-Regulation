"""Hyperparameter sensitivity of the released controller (rebuttal, R2-W4).

One-at-a-time variations around the released settings (H~ = 1, k = 4,
K = 100, alpha = 0.5) on one cell; the centre point is the ablation
campaign's `released` run of the same cell and seed. Online arm by default;
--with-c2 adds the static-matched retrain of each exported architecture.

Shares the job machinery of scripts/run_ablations.py (resumable, GPU pinning,
start deadline). Aggregate with scripts/eval_sensitivity.py.

    python scripts/run_sensitivity.py --dry-run
    python scripts/run_sensitivity.py --only hmax_0.1 hmax_10 k2 k8 --gpus 0 1 --stop-starting-after 8.5
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.run_ablations as ra  # noqa: E402

SETTINGS = {  # name -> extra train_tsrx flags (everything else = released controller)
    "hmax_0.1": ["--H-max", "0.1"],
    "hmax_10": ["--H-max", "10"],
    "k2": ["--k", "2"],
    "k8": ["--k", "8"],
    "K50": ["--update-interval", "50"],
    "K200": ["--update-interval", "200"],
    "alpha0.3": ["--budget-end-frac", "0.3"],
    "alpha0.7": ["--budget-end-frac", "0.7"],
}


def build_chains(args):
    py = sys.executable
    arch, ds = args.cell.split("/")
    cdir = Path(args.out_root) / f"{ra.ARCH_KEY[arch]}_{ds}"
    common = ["--epochs", str(args.epochs), "--batch-size", "128",
              "--num-workers", str(args.num_workers), "--data-root", args.data_root]
    chains = []
    for name in args.only or list(SETTINGS):
        out = cdir / f"{name}_s{args.seed}.pt"
        chain = [(out,
                  [py, "-m", "bench.train_tsrx", "--arch", arch, "--dataset", ds, "--seed", str(args.seed),
                   "--budget-ratio", "0.85", "--growth", "grad", "--exchange-order", "prune_first",
                   "--out", str(out)] + SETTINGS[name] + common,
                  {"arm": name, "arch": arch, "dataset": ds, "seed": args.seed, "kind": "online",
                   "setting": " ".join(SETTINGS[name])})]
        if args.with_c2:
            c2 = cdir / f"{name}_c2_s{args.seed}.pt"
            chain.append((c2, [py, "-m", "bench.train_static_matched", "--tsrx-checkpoint", str(out),
                               "--seed", str(args.seed), "--out", str(c2)] + common,
                          {"arm": name, "arch": arch, "dataset": ds, "seed": args.seed, "kind": "c2",
                           "setting": " ".join(SETTINGS[name])}))
        chains.append(chain)
    return chains


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", default="resnet18/cifar100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="*", default=None, choices=list(SETTINGS))
    ap.add_argument("--with-c2", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--gpus", nargs="*", default=None)
    ap.add_argument("--stop-starting-after", type=float, default=None, metavar="HOURS")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "sensitivity"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.stop_starting_after is not None:
        ra._stop_after_s = args.stop_starting_after * 3600
    if args.gpus:
        import queue
        ra._gpu_pool = queue.Queue()
        for g in args.gpus:
            ra._gpu_pool.put(g)
        args.concurrent = len(args.gpus)

    chains = build_chains(args)
    ra.say(f"sensitivity: {len(chains)} settings on {args.cell} seed {args.seed}, concurrent={args.concurrent}")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrent)) as pool:
        for f in [pool.submit(ra.run_chain, c, args.dry_run) for c in chains]:
            f.result()
    ra.say("done; aggregate with: python scripts/eval_sensitivity.py")


if __name__ == "__main__":
    main()
