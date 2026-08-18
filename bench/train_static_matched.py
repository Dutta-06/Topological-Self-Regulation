"""C2 control: train TSR-X's DISCOVERED architecture from scratch.

This is the acceptance gate, not a nice-to-have. TSR-X beating the
reference proves only that the discovered SHAPE is better. The framework's
actual thesis (Section 8, Theorem 8.1) is that arriving at that shape by
growth beats being born in it — a claim about the trajectory, not the
shape. Only C2 separates the two:

    TSR-X > C2   => plasticity contributed something (thesis supported)
    TSR-X ~ C2   => the shape did all the work; plasticity is an expensive
                    architecture search and the thesis is NOT supported

Reads the group widths recorded in a TSR-X checkpoint, rebuilds the
reference architecture at exactly those widths with fresh random init,
and trains it under an identical schedule.

Usage:
    python -m bench.train_static_matched \
        --tsrx-checkpoint results/tsrx/resnet18_cifar10.pt \
        --epochs 100 --out results/static_matched/resnet18_cifar10.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from bench.models import build_model, describe
from data.cifar import get_cifar10_loaders, get_cifar100_loaders
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model


def resize_model_to_widths(model: nn.Module, widths: dict, example_input) -> nn.Module:
    """Rebuild `model` so every coupling group has the recorded width.

    Uses the same coupling engine as training, so the resize is applied
    consistently across every producer / affine / consumer slot of each
    group (Definition 5.3) rather than per-module.
    """
    traced = trace_model(model.eval(), (example_input,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, model)
    modules = dict(model.named_modules())

    for tap_str, target in widths.items():
        tap = int(tap_str)
        bd = bundles.get(tap)
        if bd is None or bd.size == target:
            continue

        seen = set()
        for slot in bd.producer_slots:
            if slot.module_name in seen:
                continue
            seen.add(slot.module_name)
            mod = modules[slot.module_name]
            w = mod.weight
            mod.weight = nn.Parameter(torch.empty(target, *w.shape[1:], device=w.device, dtype=w.dtype))
            nn.init.kaiming_uniform_(mod.weight.reshape(target, -1), a=5 ** 0.5)
            if getattr(mod, "bias", None) is not None:
                mod.bias = nn.Parameter(torch.zeros(target, device=w.device, dtype=w.dtype))
            for attr in ("out_channels", "out_features"):
                if hasattr(mod, attr):
                    setattr(mod, attr, target)

        seen = set()
        for slot in bd.affine_slots:
            if slot.module_name in seen:
                continue
            seen.add(slot.module_name)
            mod = modules[slot.module_name]
            dev = mod.weight.device if getattr(mod, "weight", None) is not None else "cpu"
            dt = mod.weight.dtype if getattr(mod, "weight", None) is not None else torch.float32
            if getattr(mod, "weight", None) is not None:
                mod.weight = nn.Parameter(torch.ones(target, device=dev, dtype=dt))
            if getattr(mod, "bias", None) is not None:
                mod.bias = nn.Parameter(torch.zeros(target, device=dev, dtype=dt))
            if getattr(mod, "running_mean", None) is not None:
                mod.running_mean = torch.zeros(target, device=dev, dtype=dt)
            if getattr(mod, "running_var", None) is not None:
                mod.running_var = torch.ones(target, device=dev, dtype=dt)
            for attr in ("num_features", "num_channels"):
                if hasattr(mod, attr):
                    setattr(mod, attr, target)

        seen = set()
        for slot in bd.consumer_slots:
            if slot.module_name in seen:
                continue
            seen.add(slot.module_name)
            mod = modules[slot.module_name]
            w = mod.weight
            new_in = target * slot.multiplicity
            mod.weight = nn.Parameter(torch.empty(w.shape[0], new_in, *w.shape[2:], device=w.device, dtype=w.dtype))
            nn.init.kaiming_uniform_(mod.weight.reshape(w.shape[0], -1), a=5 ** 0.5)
            for attr in ("in_channels", "in_features"):
                if hasattr(mod, attr):
                    setattr(mod, attr, new_in)

    return model


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsrx-checkpoint", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.tsrx_checkpoint, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {})
    arch = ck_args.get("arch", "resnet18")
    dataset = ck_args.get("dataset", "cifar10")
    cifar_stem = ck_args.get("cifar_stem", True)
    widths = ck.get("discovered_widths")
    if not widths:
        raise SystemExit(
            "checkpoint has no 'discovered_widths' — rerun train_tsrx.py after the "
            "fix that records them, or this control cannot be built."
        )

    torch.manual_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    num_classes = 10 if dataset == "cifar10" else 100
    loader_fn = get_cifar10_loaders if dataset == "cifar10" else get_cifar100_loaders
    train_loader, val_loader = loader_fn(root=args.data_root, batch_size=args.batch_size,
                                          num_workers=args.num_workers)

    model = build_model(arch, num_classes, cifar_stem=cifar_stem)
    ref_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = resize_model_to_widths(model, widths, torch.zeros(2, 3, 32, 32))
    model = model.to(args.device)
    matched_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*70}")
    print(f"  C2 CONTROL: discovered architecture trained FROM SCRATCH")
    print(f"  Arch / dataset            : {arch} / {dataset}  (cifar_stem={cifar_stem})")
    print(f"  Reference params          : {ref_params:,}")
    print(f"  Matched (discovered)      : {matched_params:,}")
    print(f"  TSR-X reported params     : {ck.get('params', 'n/a'):,}" if isinstance(ck.get('params'), int) else "")
    print(f"  TSR-X best val acc        : {ck.get('best_val_acc', 'n/a')}")
    print(f"  => TSR-X must BEAT this run for the plasticity thesis to hold.")
    print(f"{'='*70}\n")

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                           weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        acc = evaluate(model, val_loader, args.device)
        is_best = acc > best
        best = max(best, acc)
        print(f"epoch {epoch+1:3d}/{args.epochs}  loss={tot/n:.4f}  val_acc={acc:.4f}  best={best:.4f}  ({time.time()-t0:.1f}s)")
        if is_best:
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                         "val_acc": acc, "best_val_acc": best, "params": matched_params,
                         "control": "C2_static_matched", "args": vars(args)}, args.out)

    tsrx_acc = ck.get("best_val_acc")
    print(f"\n{'='*70}")
    print(f"  C2 control best  : {best:.4f}  @ {matched_params:,} params")
    if isinstance(tsrx_acc, float):
        print(f"  TSR-X best       : {tsrx_acc:.4f}")
        print(f"  PLASTICITY DELTA : {tsrx_acc - best:+.4f}")
        print(f"  => thesis {'SUPPORTED' if tsrx_acc > best else 'NOT supported'} on this seed "
              f"(needs >=5 seeds + Wilcoxon to be conclusive)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
