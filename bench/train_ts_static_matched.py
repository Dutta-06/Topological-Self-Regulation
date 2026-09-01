"""C2 control for timeseries: train TSR-X's DISCOVERED TCN architecture
from scratch. Timeseries analogue of bench/train_static_matched.py — see
that file's docstring for why this control is the actual acceptance gate
for the plasticity thesis (Theorem 8.1), not TSR-X-vs-reference alone.

Usage:
    python -m bench.train_ts_static_matched \
        --tsrx-checkpoint results/ts_tsrx/tcn_ETTh1_h96.pt \
        --epochs 50 --out results/ts_static_matched/tcn_ETTh1_h96.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from bench.resize import resize_model_to_widths
from bench.ts_models import build_ts_model, ts_model_kwargs
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
    ap.add_argument("--tsrx-checkpoint", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-best-val-widths", action="store_true",
                    help="Use the best-val-epoch architecture instead of the converged final one "
                         "(only valid if the budget anneal completed before the best-val epoch)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.tsrx_checkpoint, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    arch = ck_args.get("arch", "tcn")
    dataset = ck_args.get("dataset")
    seq_len = ck.get("seq_len", ck_args.get("seq_len", 96))
    pred_len = ck.get("pred_len", ck_args.get("pred_len", 96))
    hidden = ck.get("hidden", ck_args.get("hidden", 64))
    # C2 must rebuild with the SAME RevIN setting the discovery run used,
    # or it is not a matched control.
    use_revin = ck_args.get("use_revin", True)
    # Prefer the FINAL architecture (search converged) over the best-val
    # snapshot: on LTSF the best-val epoch lands at 1-9 while the budget
    # anneal is still ramping, so `discovered_widths` can be a barely-pruned
    # model (measured: 1.4% vs the run's actual 16.1%). See the note in
    # train_ts_tsrx.py. `--use-best-val-widths` restores the old behaviour.
    widths = None if args.use_best_val_widths else ck.get("final_widths")
    width_source = "final_widths (search converged)"
    if not widths:
        widths = ck.get("discovered_widths")
        width_source = "discovered_widths (best-val snapshot)"
    if not widths:
        raise SystemExit(
            "checkpoint has no 'final_widths' or 'discovered_widths' — rerun "
            "train_ts_tsrx.py after the fix that records them, or this control "
            "cannot be built."
        )

    torch.manual_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_vars = n_channels(dataset)
    train_loader, val_loader, test_loader = get_ltsf_loaders(
        dataset, seq_len=seq_len, pred_len=pred_len,
        batch_size=args.batch_size, root=args.data_root, num_workers=args.num_workers,
    )

    # rebuild with the EXACT architecture hyperparameters of the discovery
    # run; a C2 at a different d_model/d_ff is not a matched control
    class _A: pass
    _a = _A()
    for kk, vv in ck_args.items(): setattr(_a, kk, vv)
    _a.hidden, _a.use_revin, _a.seq_len = hidden, use_revin, seq_len
    model = build_ts_model(arch, n_vars, pred_len, **ts_model_kwargs(_a))
    ref_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    example = torch.zeros(2, seq_len, n_vars)
    model = resize_model_to_widths(model, widths, example)
    model = model.to(args.device)
    matched_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*70}")
    print(f"  C2 CONTROL: discovered TCN architecture trained FROM SCRATCH")
    print(f"  Arch / dataset            : {arch} / {dataset}  (pred_len={pred_len})")
    print(f"  Reference params          : {ref_params:,}")
    print(f"  Matched (discovered)      : {matched_params:,}  ({(1-matched_params/ref_params)*100:.1f}% reduction)")
    print(f"  Width source              : {width_source}")
    print(f"  TSR-X best val MSE        : {ck.get('best_val_mse', 'n/a')}")
    print(f"  => discovered-shape MSE vs this run separates the shape from")
    print(f"     the plasticity contribution (Theorem 8.1); TSR-X is not")
    print(f"     required to beat it for the discovery result to hold.")
    print(f"{'='*70}\n")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    best_val_mse = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        val_mse, val_mae = evaluate(model, val_loader, args.device)
        is_best = val_mse < best_val_mse
        best_val_mse = min(best_val_mse, val_mse)
        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={tot/n:.4f}  val_mse={val_mse:.4f}  "
              f"val_mae={val_mae:.4f}  best_val_mse={best_val_mse:.4f}  ({time.time()-t0:.1f}s)")
        if is_best:
            test_mse, test_mae = evaluate(model, test_loader, args.device)
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_mse": val_mse, "val_mae": val_mae,
                        "test_mse": test_mse, "test_mae": test_mae,
                        "best_val_mse": best_val_mse, "params": matched_params,
                        "control": "C2_static_matched", "args": vars(args)}, args.out)

    ck_out = torch.load(args.out, map_location="cpu", weights_only=False)
    tsrx_mse = ck.get("best_val_mse")
    print(f"\n{'='*70}")
    print(f"  C2 control best val MSE : {ck_out['val_mse']:.4f}  @ {matched_params:,} params")
    print(f"  C2 test MSE / MAE       : {ck_out['test_mse']:.4f} / {ck_out['test_mae']:.4f}")
    if isinstance(tsrx_mse, float):
        print(f"  TSR-X best val MSE      : {tsrx_mse:.4f}")
        print(f"  PLASTICITY DELTA        : {ck_out['val_mse'] - tsrx_mse:+.4f} (positive = TSR-X lower/better)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
