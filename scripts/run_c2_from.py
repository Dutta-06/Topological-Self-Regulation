"""Retrain from scratch (C2) the widths exported by finished discovery runs.

Takes discovery checkpoints from any driver that records `final_widths`
(bench/train_tsrx.py, bench/train_oto.py) and trains each exported
architecture from scratch under the shared recipe with the run's own seed,
through scripts/run_ablations.py's job machinery (resumable, GPU pinning,
JSON summary per job, read by scripts/eval_ablations.py as kind "c2").

    python scripts/run_c2_from.py --arm oto --ckpts runs/oto_s42.pt runs/oto_s43.pt --gpus 0 1
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.run_ablations as ra  # noqa: E402


def run_info(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args") or ck.get("settings") or {}
    arch = a.get("arch") or ck.get("arch")
    ds = a.get("dataset") or ck.get("dataset")
    seed = a.get("seed", ck.get("seed"))
    if not (arch and ds and seed is not None and (ck.get("final_widths") or ck.get("discovered_widths"))):
        raise SystemExit(f"{ckpt}: missing arch/dataset/seed/widths")
    return arch, ds, int(seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, help="arm name the C2 summaries are filed under (e.g. oto, nobank)")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--gpus", nargs="*", default=None)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "ablations"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.gpus:
        import queue
        ra._gpu_pool = queue.Queue()
        for g in args.gpus:
            ra._gpu_pool.put(g)
        args.concurrent = len(args.gpus)

    py = sys.executable
    common = ["--epochs", str(args.epochs), "--batch-size", "128",
              "--num-workers", str(args.num_workers), "--data-root", args.data_root]
    chains = []
    for ck in args.ckpts:
        arch, ds, seed = run_info(ck)
        out = Path(args.out_root) / f"{ra.ARCH_KEY[arch]}_{ds}" / f"{args.arm}_c2_s{seed}.pt"
        cmd = [py, "-m", "bench.train_static_matched", "--tsrx-checkpoint", str(ck),
               "--seed", str(seed), "--out", str(out)] + common
        chains.append([(out, cmd, {"arm": args.arm, "arch": arch, "dataset": ds, "seed": seed, "kind": "c2",
                                   "source_checkpoint": str(ck)})])
    ra.say(f"C2 retrains: {len(chains)} jobs, concurrent={args.concurrent}")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrent)) as pool:
        for f in [pool.submit(ra.run_chain, c, args.dry_run) for c in chains]:
            f.result()
    ra.say("done; aggregate with: python scripts/eval_ablations.py")


if __name__ == "__main__":
    main()
