"""C3 control arm: train a RANDOMLY reallocated architecture, matched to
the TSR-X discovery's parameter count, from scratch.

Handles both domains — vision (classification, accuracy) and timeseries
(forecasting, MSE) — because the two differ only in loader/loss/metric and
duplicating the loop is how the best-val comparison direction gets flipped
by accident.

See bench/c3_random.py for why this control exists: C2 shows the discovered
SHAPE is good, C3 shows the SIGNAL found it.

Usage:
    python -m bench.train_c3_random --tsrx-checkpoint results/ts_tsrx/tcn_ci_traffic_h96.pt \\
        --domain ts --epochs 50 --out results/ts_c3/tcn_ci_traffic_h96.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from bench.c3_random import build_c3_model


@torch.no_grad()
def _eval_vision(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return {"acc": correct / max(total, 1)}


@torch.no_grad()
def _eval_ts(model, loader, device):
    model.eval()
    mse = mae = n = 0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        mse += F.mse_loss(pred, y, reduction="sum").item()
        mae += F.l1_loss(pred, y, reduction="sum").item()
        n += y.numel()
    model.train()
    return {"mse": mse / max(n, 1), "mae": mae / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsrx-checkpoint", required=True)
    ap.add_argument("--domain", choices=["vision", "ts"], required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--c3-seed", type=int, default=None,
                    help="seed for the RANDOM width draw (defaults to --seed); vary it to "
                         "sample several random topologies at the same budget")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.tsrx_checkpoint, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})

    # Match TSR-X's FINAL (search-converged) size, not the budget target and
    # not the best-val snapshot -- see bench/backfill_final_widths.py for why
    # those differ. C2/C3/TSR-X must sit at the same parameter count or the
    # comparison is not controlled.
    target = ck.get("final_params") or ck.get("params")
    if not target:
        raise SystemExit("checkpoint has neither 'final_params' nor 'params'")

    torch.manual_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    c3_seed = args.c3_seed if args.c3_seed is not None else args.seed

    if args.domain == "ts":
        from bench.ts_models import build_ts_model, ts_model_kwargs
        from data.ltsf import get_ltsf_loaders, n_channels

        ds = ck_args["dataset"]
        arch = ck_args.get("arch", "tcn_ci")
        seq_len = ck.get("seq_len", ck_args.get("seq_len", 96))
        pred_len = ck.get("pred_len", ck_args.get("pred_len", 96))
        hidden = ck.get("hidden", ck_args.get("hidden", 64))
        n_vars = n_channels(ds)
        bs = args.batch_size or 32
        lr = args.lr if args.lr is not None else 1e-3
        wd = args.weight_decay if args.weight_decay is not None else 1e-4

        train_loader, val_loader, test_loader = get_ltsf_loaders(
            ds, seq_len=seq_len, pred_len=pred_len, batch_size=bs,
            root=args.data_root, num_workers=args.num_workers)
        class _A: pass
        _a = _A()
        for kk, vv in ck_args.items(): setattr(_a, kk, vv)
        _a.hidden, _a.seq_len = hidden, seq_len
        build_fn = lambda: build_ts_model(arch, n_vars, pred_len, **ts_model_kwargs(_a))
        example = torch.zeros(2, seq_len, n_vars)
        loss_fn = F.mse_loss
        evaluate = _eval_ts
        key, better = "mse", lambda new, best: new < best
        best_val = float("inf")
    else:
        from bench.models import build_model
        from data.cifar import get_cifar10_loaders, get_cifar100_loaders

        ds = ck_args.get("dataset", "cifar10")
        arch = ck_args.get("arch", "resnet18")
        cifar_stem = ck_args.get("cifar_stem", True)
        n_classes = 10 if ds == "cifar10" else 100
        bs = args.batch_size or 128
        lr = args.lr if args.lr is not None else 0.1
        wd = args.weight_decay if args.weight_decay is not None else 5e-4

        loader_fn = get_cifar10_loaders if ds == "cifar10" else get_cifar100_loaders
        train_loader, val_loader = loader_fn(root=args.data_root, batch_size=bs,
                                              num_workers=args.num_workers)
        test_loader = val_loader
        build_fn = lambda: build_model(arch, n_classes, cifar_stem=cifar_stem)
        example = torch.zeros(2, 3, 32, 32)
        loss_fn = F.cross_entropy
        evaluate = _eval_vision
        key, better = "acc", lambda new, best: new > best
        best_val = 0.0

    ref_params = sum(p.numel() for p in build_fn().parameters() if p.requires_grad)
    model, widths, achieved, rel = build_c3_model(
        build_fn, example, target, seed=c3_seed)
    model = model.to(args.device)

    print(f"\n{'='*70}")
    print(f"  C3 CONTROL: RANDOM reallocation at the discovered budget")
    print(f"  Domain / dataset          : {args.domain} / {ds}")
    print(f"  Reference params          : {ref_params:,}")
    print(f"  TSR-X final params        : {target:,}")
    print(f"  C3 achieved params        : {achieved:,}  ({rel*100:.2f}% off target)")
    print(f"  C3 random widths (seed {c3_seed}): {widths}")
    print(f"  => C2 > C3 is what shows the SIGNAL found the shape,")
    print(f"     rather than any non-uniform allocation of this size.")
    print(f"{'='*70}\n")

    if rel > 0.01:
        print(f"  WARNING: C3 misses the target by {rel*100:.2f}% (>1%); "
              f"this is not a matched control. Investigate before reporting.\n")

    opt = (torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd) if args.domain == "ts"
           else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                 weight_decay=wd, nesterov=True))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    for epoch in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)

        val = evaluate(model, val_loader, args.device)
        is_best = better(val[key], best_val)
        best_val = val[key] if is_best else best_val
        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={tot/max(n,1):.4f}  "
              f"val_{key}={val[key]:.4f}  best={best_val:.4f}  ({time.time()-t0:.1f}s)")

        if is_best:
            test = evaluate(model, test_loader, args.device)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "control": "C3_random_realloc",
                "params": achieved,
                "target_params": target,
                "param_match_rel_error": rel,
                "c3_widths": widths,
                "c3_seed": c3_seed,
                "ref_params": ref_params,
                **{f"val_{k}": v for k, v in val.items()},
                **{f"test_{k}": v for k, v in test.items()},
                "args": vars(args),
            }, args.out)

    print(f"\n{'='*70}")
    print(f"  C3 best val_{key} : {best_val:.4f}  @ {achieved:,} params")
    print(f"  Checkpoint         : {args.out}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
