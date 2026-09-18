"""Structured-pruning baseline on the same plastic set as TSR-X (forecasting).

The forecasting counterpart of bench/prune_baseline.py, written against the
`tsrx-time` harness (bench.ts_models, bench.resize, data.ltsf, bench.c3_random).
Takes a dense forecasting reference, removes channels from exactly the groups
TSR-X could resize -- CandidateBank eligibility AND the one-index safety probe
(`tsrx.graph.safety`), so LayerNorm-spanned residual widths and attention
projections are never touched -- until the deployed count reaches the target
C2 model's, then fine-tunes or retrains from scratch under the C2 recipe
(Adam, cosine, the discovery run's lr / weight decay / batch size).

Criteria: l1, taylor (the controller's own first-order saliency) and bnscale
(TCN only; groups without a BatchNorm scale fall back to l1).

Usage (paths follow scripts/overnight_ts.sh):
    python -m bench.prune_baseline_ts \\
        --match results/ts_static_matched/tcn_ci_weather_h96.pt \\
        --criterion l1 --mode finetune \\
        --out results/ts_pruned/tcn_ci_weather_h96_l1_finetune.pt
"""

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

from bench.c3_random import attachable_taps
from bench.resize import resize_model_to_widths
from bench.ts_models import build_ts_model, ts_model_kwargs
from data.ltsf import get_ltsf_loaders, n_channels
from tsrx.edit.pruning import (
    deployed_params, editable_bundles, importance_bnscale, importance_l1,
    make_importance_taylor, prune_to_target,
)

ROOT = Path(__file__).resolve().parent.parent
CRITERIA = ("l1", "bnscale", "taylor")
MODES = ("finetune", "scratch")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    se = ae = n = 0.0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        se += F.mse_loss(pred, y, reduction="sum").item()
        ae += F.l1_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return se / max(n, 1), ae / max(n, 1)


def reference_for(match_path: Path, match_ck: dict) -> Path:
    """The dense reference behind a C2 checkpoint: the TSR-X run it was built
    from names the cell, and the campaign stores its reference under the same
    tag in results/ts_reference/."""
    src = match_ck.get("args", {}).get("tsrx_checkpoint")
    stem = Path(src).stem if src else match_path.stem
    # sweep names are <kind>_<tag>_br<budget>[_s<seed>|_c<draw>]; the reference is just <tag>
    tag = re.sub(r"_br[0-9.]+(_s\d+|_c\d+)?$", "", re.sub(r"^(tsrx|c2|c3)_", "", stem))
    # the campaign leaves the TCN untagged, but its references are stored as tcn_<cell>
    for t in (tag, f"tcn_{tag}"):
        for cand in (ROOT / "results" / "ts_reference" / f"{t}.pt",
                     match_path.parent.parent / "ts_reference" / f"{t}.pt"):
            if cand.exists():
                return cand
    raise SystemExit(f"no dense reference found for {match_path.name} (looked for ts_reference/{tag}.pt)")


def train(model, loaders, device, epochs, lr, weight_decay, record, out: Path) -> dict:
    train_loader, val_loader, test_loader = loaders
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        val_mse, val_mae = evaluate(model, val_loader, device)
        is_best = val_mse < best_val
        best_val = min(best_val, val_mse)
        print(f"epoch {epoch + 1:3d}/{epochs}  loss={tot / n:.4f}  val_mse={val_mse:.4f}  "
              f"val_mae={val_mae:.4f}  best_val_mse={best_val:.4f}  ({time.time() - t0:.1f}s)")
        if is_best:
            test_mse, test_mae = evaluate(model, test_loader, device)
            torch.save({**record, "model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_mse": val_mse, "val_mae": val_mae,
                        "test_mse": test_mse, "test_mae": test_mae,
                        "best_val_mse": best_val, "complete": False}, out)
    ck = torch.load(out, map_location="cpu", weights_only=False)
    ck["complete"] = True
    ck["total_epochs"] = epochs
    torch.save(ck, out)
    return ck


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--match", required=True,
                    help="C2 checkpoint (results/ts_static_matched/*.pt) whose count is the target")
    ap.add_argument("--reference", default=None, help="dense reference (default: the C2's cell in ts_reference/)")
    ap.add_argument("--target-params", type=int, default=None, help="override the C2 count")
    ap.add_argument("--criterion", choices=CRITERIA, required=True)
    ap.add_argument("--mode", choices=MODES, default="finetune")
    ap.add_argument("--min-width", type=int, default=8, help="width floor, as in TSR-X")
    ap.add_argument("--taylor-batches", type=int, default=50)
    ap.add_argument("--round-fraction", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=None, help="default 20 (finetune) / 50 (scratch, = C2)")
    ap.add_argument("--lr", type=float, default=None, help="default: the discovery run's lr")
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- the cell, from the C2 checkpoint and the TSR-X run behind it -------
    match_path = Path(args.match)
    match_ck = torch.load(match_path, map_location="cpu", weights_only=False)
    tsrx_path = match_ck.get("args", {}).get("tsrx_checkpoint")
    run_args = {}
    if tsrx_path and Path(tsrx_path).exists():
        run_args = torch.load(tsrx_path, map_location="cpu", weights_only=False).get("args", {})
    ref_path = Path(args.reference) if args.reference else reference_for(match_path, match_ck)
    ref_ck = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_args = ref_ck.get("args", {})
    run_args = {**ref_args, **run_args}          # architecture from the reference, recipe from the run

    arch = run_args.get("arch", ref_ck.get("arch", "tcn_ci"))
    dataset = run_args.get("dataset", ref_ck.get("dataset"))
    seq_len = ref_ck.get("seq_len", run_args.get("seq_len", 96))
    pred_len = ref_ck.get("pred_len", run_args.get("pred_len", 96))
    hidden = ref_ck.get("hidden", run_args.get("hidden", 64))
    n_vars = n_channels(dataset)
    hp = SimpleNamespace(**run_args)
    hp.hidden, hp.seq_len = hidden, seq_len
    build_fn = lambda: build_ts_model(arch, n_vars, pred_len, **ts_model_kwargs(hp))  # noqa: E731
    example = torch.zeros(2, seq_len, n_vars)

    batch_size = args.batch_size or run_args.get("batch_size", 32)
    lr = args.lr if args.lr is not None else run_args.get("lr", 1e-3)
    weight_decay = args.weight_decay if args.weight_decay is not None else run_args.get("weight_decay", 1e-4)
    epochs = args.epochs or (20 if args.mode == "finetune" else 50)
    target = args.target_params or int(match_ck["params"])

    loaders = get_ltsf_loaders(dataset, seq_len=seq_len, pred_len=pred_len, batch_size=batch_size,
                               root=args.data_root, num_workers=args.num_workers)
    train_loader, val_loader, test_loader = loaders

    model = build_fn()
    model.load_state_dict(ref_ck["model_state_dict"])
    ref_params = deployed_params(model)
    model = model.to(args.device)
    ref_val = evaluate(model, val_loader, args.device)
    ref_test = evaluate(model, test_loader, args.device)
    print(f"\n{'=' * 70}\n  PRUNING BASELINE (forecasting)  {args.criterion} / {args.mode}\n"
          f"  Arch / dataset / horizon : {arch} / {dataset} / h{pred_len}\n"
          f"  Reference                : {ref_path.name}  test MSE {ref_test[0]:.4f} at {ref_params:,} params\n"
          f"  Target                   : {target:,} params ({100 * (target / ref_params - 1):+.1f}%) = {match_path.name}\n"
          f"  lr / wd / batch / epochs : {lr} / {weight_decay} / {batch_size} / {epochs}\n{'=' * 70}\n")

    # --- the plastic set: eligibility + the harness's safety probe -----------
    allowed = attachable_taps(build_fn(), example, build_fn=build_fn)
    bundles = editable_bundles(model.cpu(), example, allowed_taps=allowed)
    if not bundles:
        raise SystemExit("no resizable group survives the safety probe; nothing to prune")
    reference_widths = {t: b.size for t, b in bundles.items()}
    model.to(args.device)

    if args.criterion == "taylor":
        def loss_of(m, batch):
            x, y = batch
            return F.mse_loss(m(x.to(args.device)), y.to(args.device))
        importance_fn = make_importance_taylor(lambda: iter(train_loader), loss_of, args.taylor_batches)
    else:
        importance_fn = {"l1": importance_l1, "bnscale": importance_bnscale}[args.criterion]
    t0 = time.time()
    events = prune_to_target(model, bundles, target, importance_fn, min_width=args.min_width,
                             round_fraction=args.round_fraction, log=print)
    pruned_params = deployed_params(model)
    widths = {str(t): b.size for t, b in bundles.items()}
    with torch.no_grad():
        model.cpu().eval()(example)
    print(f"pruning took {time.time() - t0:.1f}s over {len(bundles)} groups "
          f"({len(events)} channels removed)")
    if pruned_params > target:
        raise SystemExit(f"pruned model has {pruned_params:,} > target {target:,}")
    if target - pruned_params > 0.01 * target:
        raise SystemExit(f"undershot the target by more than 1%: {pruned_params:,} vs {target:,}")

    model = model.to(args.device)
    pruned_val = evaluate(model, val_loader, args.device)
    print(f"pruned, before any training: val MSE {pruned_val[0]:.4f}")

    if args.mode == "scratch":
        torch.manual_seed(args.seed)
        model = resize_model_to_widths(build_fn(), widths, example).to(args.device)
        assert deployed_params(model) == pruned_params, "rebuilt width vector has a different count"

    record = {
        "control": f"pruned_{args.criterion}_{args.mode}",
        "arch": arch, "dataset": dataset, "seq_len": seq_len, "pred_len": pred_len, "hidden": hidden,
        "criterion": args.criterion, "mode": args.mode,
        "reference": ref_path.name, "reference_params": ref_params,
        "reference_val_mse": ref_val[0], "reference_test_mse": ref_test[0], "reference_test_mae": ref_test[1],
        "match": match_path.name, "target_params": target, "params": pruned_params,
        "pruned_val_mse_before_training": pruned_val[0],
        "reference_widths": {str(t): w for t, w in reference_widths.items()},
        "discovered_widths": widths, "removed": len(events),
        "epochs": epochs, "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        "args": {**vars(args), "tsrx_checkpoint": tsrx_path},
    }
    ck = train(model, loaders, args.device, epochs, lr, weight_decay, record, out)
    out.with_suffix(".events.json").write_text(json.dumps(events), encoding="utf-8")
    print(f"\n{'=' * 70}\n  {record['control']}  test MSE {ck['test_mse']:.4f} at {pruned_params:,} params "
          f"(reference {ref_test[0]:.4f} at {ref_params:,}; C2 {match_ck.get('test_mse', float('nan')):.4f})\n{'=' * 70}")


if __name__ == "__main__":
    main()
