"""Train a plain TCN reference (no TSR-X) on an LTSF forecasting dataset
and checkpoint it — the timeseries analogue of bench/train_reference.py.

Usage:
    python -m bench.train_ts_reference --dataset ETTh1 --pred-len 96 \
        --epochs 50 --out results/ts_reference/tcn_ETTh1_h96.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from bench.ts_models import build_ts_model, describe
from data.ltsf import get_ltsf_loaders, n_channels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_mse, tot_mae, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        tot_mse += F.mse_loss(pred, y, reduction="sum").item()
        tot_mae += F.l1_loss(pred, y, reduction="sum").item()
        n += y.numel()
    model.train()
    return tot_mse / max(n, 1), tot_mae / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["tcn", "tcn_ci"], default="tcn_ci",
                    help="tcn_ci (default) is channel-independent: the head is Linear(hidden, "
                         "pred_len) instead of Linear(hidden, pred_len*n_vars), so the conv body "
                         "TSR-X can reallocate is 65-93%% of params instead of 1-41%%")
    ap.add_argument("--dataset", choices=["ETTh1", "ETTh2", "weather", "electricity", "traffic"], required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_vars = n_channels(args.dataset)
    train_loader, val_loader, test_loader = get_ltsf_loaders(
        args.dataset, seq_len=args.seq_len, pred_len=args.pred_len,
        batch_size=args.batch_size, root=args.data_root, num_workers=args.num_workers,
    )

    dev_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else args.device
    model = build_ts_model(args.arch, n_vars, args.pred_len, hidden=args.hidden).to(args.device)
    baseline_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*70}")
    print(f"  Standard Static Reference Training: {args.arch.upper()} on {args.dataset}")
    print(f"  Compute Device               : {dev_name} ({args.device})")
    print(f"  Variates / seq_len / pred_len: {n_vars} / {args.seq_len} / {args.pred_len}")
    print(f"  Total Parameters             : {baseline_params:,} (Fixed Static)")
    print(f"  Block shapes                 : {describe(model, args.seq_len, n_vars)['block_shapes']}")
    print(f"  Epochs                       : {args.epochs}")
    print(f"{'='*70}\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    best_val_mse = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0, total_loss, n = time.time(), 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{opt.param_groups[0]['lr']:.2e}"})

        val_mse, val_mae = evaluate(model, val_loader, args.device)
        is_best = val_mse < best_val_mse  # LOWER is better for MSE, unlike accuracy
        best_val_mse = min(best_val_mse, val_mse)
        print(f"epoch {epoch+1:3d}/{args.epochs}  train_loss={total_loss/n:.4f}  "
              f"val_mse={val_mse:.4f}  val_mae={val_mae:.4f}  best_val_mse={best_val_mse:.4f}  ({time.time()-t0:.1f}s)")

        if is_best:
            test_mse, test_mae = evaluate(model, test_loader, args.device)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_mse": val_mse, "val_mae": val_mae,
                "test_mse": test_mse, "test_mae": test_mae,
                "arch": args.arch, "dataset": args.dataset,
                "seq_len": args.seq_len, "pred_len": args.pred_len, "hidden": args.hidden,
                "params": baseline_params,
                "args": vars(args),
            }, out_path)

    ck = torch.load(out_path, map_location="cpu", weights_only=False)
    print(f"\n{'='*70}")
    print(f"  Reference Baseline Training Complete")
    print(f"  Model / Dataset           : {args.arch.upper()} / {args.dataset}  (pred_len={args.pred_len})")
    print(f"  Fixed Parameters          : {baseline_params:,}")
    print(f"  Best Val MSE              : {ck['val_mse']:.4f}")
    print(f"  Test MSE / MAE @ best val : {ck['test_mse']:.4f} / {ck['test_mae']:.4f}")
    print(f"  Checkpoint Saved To       : {out_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
