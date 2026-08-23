"""TSR-X Training Harness for Coupled Reference Architectures (ResNet-18, VGG-16).

Dynamically reallocates channel capacity across coupled groups under a fixed or
reduced parameter budget B <= C(n_A) to achieve strict Pareto dominance
(higher accuracy with fewer parameters and FLOPs).

Usage:
    python -m bench.train_tsrx --arch resnet18 --dataset cifar100 --epochs 100 \\
        --budget-ratio 0.85 --out results/tsrx/resnet18_cifar100.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from bench.models import build_model, describe
from data.cifar import get_cifar10_loaders, get_cifar100_loaders
from tsrx.alloc.cost import kappa_params
from tsrx.alloc.exchange import evaluate_structural_update, apply_exchange
from tsrx.alloc.schedule import budget_at
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank, UnsupportedGroupError
from tsrx.sense.curvature import calibrate_h_max
from tsrx.sense.saliency import ActivationStats, first_order_saliency
from tsrx.sense.topo import WindowedSignal, compute_uc_norms


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description="TSR-X Dynamic Architecture Training")
    ap.add_argument("--arch", choices=["resnet18", "vgg16_bn"], default="resnet18")
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar100")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--k", type=int, default=4, help="Candidate bank size per group")
    ap.add_argument("--update-interval", type=int, default=100, help="Steps between structural updates")
    ap.add_argument("--budget-ratio", type=float, default=0.85, help="Max param budget as ratio of baseline (e.g. 0.85)")
    ap.add_argument("--delta", type=float, default=1e-7, help="Exchange acceptance margin")
    ap.add_argument("--prune-tol", type=float, default=1e-3,
                    help="RELATIVE pure-prune tolerance (fraction of group median saliency)")
    ap.add_argument("--min-size", type=int, default=8, help="Min channels per group")
    ap.add_argument("--cifar-stem", dest="cifar_stem", action="store_true", default=True)
    ap.add_argument("--imagenet-stem", dest="cifar_stem", action="store_false")
    ap.add_argument("--H-max", type=float, default=1.0, help="curvature bound for Eq.14 removal score")
    ap.add_argument("--calibrate-hmax", action="store_true",
                    help="Replace --H-max with a data-driven estimate (curvature.calibrate_h_max), "
                         "refreshed every structural-update window")
    ap.add_argument("--budget-end-frac", type=float, default=0.5,
                    help="Fraction of total steps over which the budget anneals from baseline to target; "
                         "held at target after. Stretch toward 0.7 for aggressive (tight) budgets.")
    ap.add_argument("--max-prunes-per-update", type=int, default=4,
                    help="Feasibility-regime cap: at most this many groups pruned (one index each) per update")
    ap.add_argument("--verbose-decisions", action="store_true",
                    help="Print a compact line for every structural-update decision, including no-ops")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", default="results/tsrx/resnet18_cifar100.pt")
    ap.add_argument("--smoke-test", action="store_true", help="Quick 500-step test")
    args = ap.parse_args()

    if args.smoke_test:
        args.max_steps = 500
        args.update_interval = 50

    torch.manual_seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = 10 if args.dataset == "cifar10" else 100
    loader_fn = get_cifar10_loaders if args.dataset == "cifar10" else get_cifar100_loaders
    train_loader, val_loader = loader_fn(
        root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # 1. Build Reference Model
    model = build_model(args.arch, num_classes, cifar_stem=args.cifar_stem).to(args.device)

    baseline_params = count_params(model)
    if args.budget_ratio is None or args.budget_ratio <= 0 or args.budget_ratio >= 100.0:
        budget_params = None
        cap_str = "None (Unconstrained Capacity Discovery)"
    else:
        budget_params = int(baseline_params * args.budget_ratio)
        cap_str = f"{budget_params:,} ({args.budget_ratio*100:.0f}% of baseline)"

    dev_name = torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else args.device
    print(f"\n{'='*70}")
    print(f"  TSR-X Dynamic Plasticity Training: {args.arch.upper()} on {args.dataset.upper()}")
    print(f"  Compute Device               : {dev_name} ({args.device})")
    print(f"  Baseline Architecture Params : {baseline_params:,}")
    print(f"  Target Parameter Budget Cap  : {cap_str}")
    print(f"  Candidate Bank Size (k)      : {args.k}")
    print(f"  Structural Update Interval   : Every {args.update_interval} steps")
    print(f"{'='*70}\n")

    # 2. Graph Tracing & Coupling Group Discovery
    it = iter(train_loader)
    xb0, _ = next(it)
    traced = trace_model(model, (xb0[:2].to(args.device),))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, model)

    bank = CandidateBank(model, bundles, k=args.k)
    bank = bank.to(args.device)

    # 3. Optimizer & Scheduler
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay, nesterov=True)
    total_steps = args.max_steps or (args.epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    win = WindowedSignal(window=args.update_interval)
    act_stats = ActivationStats(model, bank)   # E[a_j^2] for Eq.(14)'s 2nd-order term
    saliency_sum: Dict[int, torch.Tensor] = {tap: None for tap in bank.handles}
    n_saliency_seen = 0
    h_max_value = args.H_max
    decisions_path = out_path.parent / f"{out_path.stem}_decisions.jsonl"
    decisions_file = open(decisions_path, "a")

    # Assert dormancy holds before training starts (should be exactly 0.0).
    assert bank.max_port_magnitude() == 0.0, "candidate ports must start dormant"

    global_step = 0
    best_acc = 0.0
    structural_events_count = 0
    events_log = []

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        total_loss, n_samples = 0.0, 0

        for xb, yb in pbar:
            if args.max_steps and global_step >= args.max_steps:
                break

            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad(set_to_none=True)

            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()

            # Sense candidate gradients u_c and compute removal saliency
            for tap, h in bank.handles.items():
                u = compute_uc_norms(bank, tap)
                win.record(tap, u)
                sal = first_order_saliency(model, h.bundle, h.base_size)
                saliency_sum[tap] = sal if saliency_sum[tap] is None else saliency_sum[tap] + sal

            n_saliency_seen += 1
            opt.step()
            # CRITICAL (Definition 3.2 / Lemma 2.2): candidate ports carry a
            # real gradient (u_c IS that gradient), so opt.step() trains them
            # away from zero unless we re-zero. Without this the "candidates"
            # become live channels, u_c stops being the topological derivative
            # at the silent point, and accuracy gets measured on a model that
            # deployed_params() does not count.
            bank.zero_ports()
            sched.step()

            loss_val = loss.item()
            total_loss += loss_val * xb.size(0)
            n_samples += xb.size(0)
            current_params = bank.deployed_params()

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{opt.param_groups[0]['lr']:.2e}",
                "params": f"{current_params:,}",
                "saved": f"{(1 - current_params/baseline_params)*100:.1f}%",
                "events": structural_events_count,
            })

            # Slow timescale: Two-regime structural update (feasibility vs
            # optimality — see tsrx/alloc/exchange.py module docstring)
            if global_step > 0 and global_step % args.update_interval == 0:
                t0_struct = time.time()
                deployed_before = bank.deployed_params()
                B_t = budget_at(global_step, total_steps, baseline_params, budget_params,
                                 end_frac=args.budget_end_frac)

                if args.calibrate_hmax:
                    def _loss_fn(m, b):
                        return F.cross_entropy(m(b[0]), b[1])
                    h_max_value = calibrate_h_max(bank, _loss_fn, (xb, yb))

                decisions = evaluate_structural_update(
                    bank=bank,
                    windowed_signal=win,
                    saliency_sum=saliency_sum,
                    n_seen=n_saliency_seen,
                    deployed_params=deployed_before,
                    budget_at_t=B_t,
                    delta=args.delta,
                    prune_tolerance=args.prune_tol,
                    min_size_per_group=args.min_size,
                    act_stats=act_stats,
                    H_max=h_max_value,
                    max_prunes_per_update=args.max_prunes_per_update,
                )

                applied = [d for d in decisions if d.action != "none"]
                for dec in applied:
                    apply_exchange(dec, bank, optimizer=opt, eps=1e-3)

                if applied:
                    # materialize_candidate writes a NONZERO port for the
                    # promoted unit (that is the point: v* = -eps*u_c/||u_c||),
                    # but every REMAINING candidate must stay dormant. One
                    # zero_ports() call after the WHOLE batch, not per-edit.
                    bank.zero_ports()
                    act_stats.remove()
                    act_stats = ActivationStats(model, bank)
                    structural_events_count += len(applied)

                overhead_ms = (time.time() - t0_struct) * 1000
                new_p = bank.deployed_params()

                for dec in decisions:
                    if args.verbose_decisions or dec.action != "none":
                        tag = {"exchange": "EXCHANGE", "pure_grow": "GROW",
                               "pure_prune": "PRUNE", "none": "NONE"}[dec.action]
                        name = ""
                        if dec.grow_tap is not None:
                            name += f" grow={bank.handles.get(dec.grow_tap, None) and bank.handles[dec.grow_tap].bundle.producer_slots[0].module_name}"
                        if dec.prune_tap is not None:
                            name += f" prune={bank.handles.get(dec.prune_tap, None) and bank.handles[dec.prune_tap].bundle.producer_slots[0].module_name}(idx {dec.prune_idx})"
                        tqdm.write(
                            f"  [{global_step:>6}] [{dec.regime:>11}:{tag:<8}]{name}  "
                            f"reason={dec.reason}  B_t={B_t}  params={new_p:,} ({overhead_ms:.0f}ms)"
                        )
                    decisions_file.write(json.dumps({
                        "step": global_step,
                        "regime": dec.regime,
                        "action": dec.action,
                        "reason": dec.reason,
                        "deployed_before": deployed_before,
                        "deployed_after": new_p,
                        "budget_at_t": B_t,
                        "H_max": h_max_value,
                        "grow_tap": dec.grow_tap,
                        "grow_cand_idx": dec.grow_cand_idx,
                        "grow_gamma": dec.grow_gamma,
                        "prune_tap": dec.prune_tap,
                        "prune_idx": dec.prune_idx,
                        "prune_rho": dec.prune_rho,
                        "delta_loss_bound": dec.delta_loss_bound,
                        "details": {k: v for k, v in dec.details.items()
                                    if not isinstance(v, (dict, list)) or k in ("grow", "prune")},
                    }, default=str) + "\n")
                decisions_file.flush()

                if applied:
                    events_log.append({
                        "step": global_step,
                        "actions": [d.action for d in applied],
                        "params": new_p,
                        "decisions": [d.details for d in applied],
                    })

                # Reset saliency + activation accumulators for next window
                saliency_sum = {tap: None for tap in bank.handles}
                n_saliency_seen = 0
                act_stats.reset()

            global_step += 1

        # Per-epoch evaluation
        val_acc = evaluate(model, val_loader, args.device)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        curr_p = bank.deployed_params()
        pct_saved = (1.0 - curr_p / baseline_params) * 100.0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  loss={total_loss/max(n_samples, 1):.4f}  "
            f"val_acc={val_acc:.4f}  best={best_acc:.4f}  "
            f"params={curr_p:,} ({pct_saved:+.1f}% saved)  events={structural_events_count}"
        )

        if is_best:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "best_val_acc": best_acc,
                "params": curr_p,
                "baseline_params": baseline_params,
                "param_saving_pct": pct_saved,
                "structural_events": events_log,
                "discovered_widths": {str(t): h.base_size for t, h in bank.handles.items()},
                "max_port_magnitude": bank.max_port_magnitude(),
                "args": vars(args),
            }, out_path)

        if args.max_steps and global_step >= args.max_steps:
            break

    decisions_file.close()

    final_p = bank.deployed_params()
    final_saved = (1.0 - final_p / baseline_params) * 100.0
    print(f"\n{'='*70}")
    print(f"  TSR-X Training Complete")
    print(f"  Baseline Reference Params : {baseline_params:,}")
    print(f"  Final Discovered Params   : {final_p:,} ({final_saved:+.1f}% parameter reduction)")
    print(f"  Best Validation Accuracy  : {best_acc:.4f}")
    print(f"  Total Structural Events   : {structural_events_count}")
    print(f"  Checkpoint Saved To       : {out_path}")
    print(f"  Decision Trace Saved To   : {decisions_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
