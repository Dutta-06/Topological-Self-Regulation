"""Retrain from scratch (C2) the widths exported by finished discovery runs.

Takes discovery checkpoints from any driver that records `final_widths`
(bench/train_tsrx.py, bench/train_oto.py) and trains each exported
architecture from scratch under the shared recipe with the run's own seed,
through scripts/run_ablations.py's job machinery (resumable, GPU pinning,
JSON summary per job, read by scripts/eval_ablations.py as kind "c2").

    python scripts/run_c2_from.py --arm oto --ckpts runs/oto_s42.pt runs/oto_s43.pt --gpus 0 1
    python scripts/run_c2_from.py --specs results/c2_specs/*/oto_s*.json --gpus 0 1

--specs takes the verified width specs of scripts/make_c2_specs.py instead of
checkpoints (the arm name is read from each spec).
"""

import argparse
import json
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
    ap.add_argument("--arm", default=None, help="arm name the C2 summaries are filed under (with --ckpts)")
    ap.add_argument("--ckpts", nargs="*", default=[])
    ap.add_argument("--specs", nargs="*", default=[], help="width specs from scripts/make_c2_specs.py")
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
    jobs = [(args.arm, ck) for ck in args.ckpts]
    for sp in args.specs:  # write a minimal checkpoint that train_static_matched reads
        spec = json.loads(Path(sp).read_text(encoding="utf-8"))
        cdir = Path(args.out_root) / f"{ra.ARCH_KEY[spec['arch']]}_{spec['dataset']}"
        cdir.mkdir(parents=True, exist_ok=True)
        ck = cdir / f"{spec['arm']}_s{spec['seed']}_widths.pt"
        torch.save({"args": {"arch": spec["arch"], "dataset": spec["dataset"], "cifar_stem": True,
                             "seed": spec["seed"]},
                    "final_widths": {int(t): int(v) for t, v in spec["final_widths"].items()},
                    "params": spec["params"]}, ck)
        jobs.append((spec["arm"], str(ck)))
    if not jobs or any(a is None for a, _ in jobs):
        raise SystemExit("give --specs, or --ckpts with --arm")
    chains = []
    for arm, ck in jobs:
        arch, ds, seed = run_info(ck)
        out = Path(args.out_root) / f"{ra.ARCH_KEY[arch]}_{ds}" / f"{arm}_c2_s{seed}.pt"
        cmd = [py, "-m", "bench.train_static_matched", "--tsrx-checkpoint", str(ck),
               "--seed", str(seed), "--out", str(out)] + common
        chains.append([(out, cmd, {"arm": arm, "arch": arch, "dataset": ds, "seed": seed, "kind": "c2",
                                   "source_checkpoint": str(ck)})])
    ra.say(f"C2 retrains: {len(chains)} jobs, concurrent={args.concurrent}")
    with ThreadPoolExecutor(max_workers=max(1, args.concurrent)) as pool:
        for f in [pool.submit(ra.run_chain, c, args.dry_run) for c in chains]:
            f.result()
    ra.say("done; aggregate with: python scripts/eval_ablations.py")


if __name__ == "__main__":
    main()
