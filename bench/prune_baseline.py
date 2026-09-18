"""Structured-pruning baseline on the same plastic set as TSR-X (vision).

Takes the archived dense reference, removes whole channels from exactly the
coupling groups TSR-X was allowed to edit (a producer and a consumer, no
task-fixed axis) until the deployed count reaches the matching C2 model's, then
either fine-tunes the pruned network (inherited weights) or retrains the pruned
width vector from scratch under the C2 recipe (Liu et al., 2019).

The pruning itself is tsrx/edit/pruning.py: removals go through
`prune_group_index`, so residual trunks, depthwise pairs and SE branches are
edited by index exactly as TSR-X edits them, and the count is the real tensor
extent. Nothing here gives the baseline an action TSR-X lacked, and nothing
denies it one TSR-X had, except growth: pruning only removes.

Criteria: l1 (Li et al., 2017), bnscale (Liu et al., 2017), taylor (Molchanov
et al., 2017, via the paper's own first-order saliency).

Usage:
    python -m bench.prune_baseline --arch resnet18 --dataset cifar100 \
        --criterion l1 --mode finetune --out results/pruned/resnet18_cifar100_l1_finetune.pt
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from bench.models import build_model
from bench.train_static_matched import resize_model_to_widths
from tsrx.edit.pruning import (
    deployed_params, editable_bundles, importance_bnscale, importance_l1,
    make_importance_taylor, prune_to_target,
)

ROOT = Path(__file__).resolve().parent.parent

# arch name used by build_model -> key used in results/ filenames
ARCH_KEY = {"resnet18": "resnet18", "vgg16_bn": "vgg16bn",
            "mobilenet_v2": "mobilenetv2", "efficientnet_b0": "efficientnet_b0"}
CRITERIA = ("l1", "bnscale", "taylor")
MODES = ("finetune", "scratch")


# ---------------------------------------------------------------------------
# Data and evaluation (the shared recipe of the three arms)
# ---------------------------------------------------------------------------

def make_data(dataset: str, data_root: str, batch_size: int, num_workers: int, device: str):
    from data.cifar import get_cifar10_loaders, get_cifar100_loaders
    from data.tiny_imagenet import GPUBatchTransform, get_tiny_imagenet_loaders
    from data.imagenet100 import get_imagenet100_loaders

    train_tf = eval_tf = None
    if dataset in ("imagenet100", "imagenet-100"):
        num_classes, hw, loader_fn = 100, 224, get_imagenet100_loaders
    elif dataset == "cifar10":
        num_classes, hw, loader_fn = 10, 32, get_cifar10_loaders
    elif dataset == "cifar100":
        num_classes, hw, loader_fn = 100, 32, get_cifar100_loaders
    else:
        num_classes, hw, loader_fn = 200, 64, get_tiny_imagenet_loaders
        train_tf = GPUBatchTransform(is_train=True).to(device)
        eval_tf = GPUBatchTransform(is_train=False).to(device)
    train_loader, val_loader = loader_fn(root=data_root, batch_size=batch_size,
                                         num_workers=num_workers)
    return train_loader, val_loader, num_classes, hw, train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device, batch_tf=None, use_amp=False) -> float:
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if batch_tf is not None:
            x = batch_tf(x)
        with torch.amp.autocast("cuda", enabled=use_amp):
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# Locating the arms
# ---------------------------------------------------------------------------

def find_reference(arch: str, dataset: str, cifar_stem: bool) -> Path:
    """The dense reference for a cell, chosen by inspecting the checkpoint."""
    key = ARCH_KEY[arch]
    ds = dataset.replace("-", "_")
    candidates = [ROOT / "results" / "reference" / f"{key}_{ds}_stem.pt",
                  ROOT / "results" / "reference" / f"{key}_{ds}.pt"]
    checked = []
    for path in candidates:
        if not path.exists():
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["model_state_dict"]
        ok = (state["conv1.weight"].shape[2] == 3) == cifar_stem if arch == "resnet18" else True
        complete = ck.get("complete", True) and ck.get("epoch", 99) >= 10
        checked.append((path.name, ok, complete))
        if ok and complete:
            return path
    raise SystemExit(f"no usable dense reference for {arch}/{dataset} (cifar_stem={cifar_stem}); "
                     f"inspected {checked}")


def find_match(arch: str, dataset: str) -> Optional[Path]:
    path = ROOT / "results" / "static_matched" / f"{ARCH_KEY[arch]}_{dataset.replace('-', '_')}_two_regime.pt"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Training (the C2 recipe)
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, args, device, train_tf, eval_tf, epochs, lr,
          record: dict, out: Path) -> float:
    use_amp = args.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    best = 0.0
    for epoch in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train_tf is not None:
                x = train_tf(x)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = nn.functional.cross_entropy(model(x), y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
        acc = evaluate(model, val_loader, device, batch_tf=eval_tf, use_amp=use_amp)
        is_best = acc > best
        best = max(best, acc)
        print(f"epoch {epoch + 1:3d}/{epochs}  loss={tot / n:.4f}  val_acc={acc:.4f}  "
              f"best={best:.4f}  ({time.time() - t0:.1f}s)")
        if is_best:
            torch.save({**record, "model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_acc": acc, "best_val_acc": best, "complete": False}, out)
    ck = torch.load(out, map_location="cpu", weights_only=False)
    ck["complete"] = True
    ck["total_epochs"] = epochs
    torch.save(ck, out)
    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=sorted(ARCH_KEY), required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny_imagenet", "tiny-imagenet",
                                          "imagenet100", "imagenet-100"], required=True)
    ap.add_argument("--criterion", choices=CRITERIA, required=True)
    ap.add_argument("--mode", choices=MODES, default="finetune",
                    help="finetune: keep the reference weights; scratch: retrain the pruned width vector")
    ap.add_argument("--reference", default=None, help="dense reference checkpoint (default: auto)")
    ap.add_argument("--match", default=None, help="C2 checkpoint whose deployed count is the target (default: auto)")
    ap.add_argument("--target-params", type=int, default=None, help="explicit target instead of --match")
    ap.add_argument("--min-width", type=int, default=8, help="width floor, as in TSR-X")
    ap.add_argument("--taylor-batches", type=int, default=50)
    ap.add_argument("--round-fraction", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=None, help="default 40 (finetune) / 100 (scratch)")
    ap.add_argument("--lr", type=float, default=None, help="default 0.01 (finetune) / 0.1 (scratch)")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=None, help="default 128, 64 for ImageNet-100")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    imagenet = args.dataset in ("imagenet100", "imagenet-100")
    cifar_stem = not imagenet
    epochs = args.epochs or (40 if args.mode == "finetune" else 100)
    lr = args.lr or (0.01 if args.mode == "finetune" else 0.1)
    batch_size = args.batch_size or (64 if imagenet else 128)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes, hw, train_tf, eval_tf = make_data(
        args.dataset, args.data_root, batch_size, args.num_workers, args.device)
    example = torch.zeros(2, 3, hw, hw)

    # --- the dense reference and the target count ---------------------------
    ref_path = Path(args.reference) if args.reference else find_reference(args.arch, args.dataset, cifar_stem)
    ref_ck = torch.load(ref_path, map_location="cpu", weights_only=False)
    model = build_model(args.arch, num_classes, cifar_stem=cifar_stem)
    model.load_state_dict(ref_ck["model_state_dict"])
    ref_params = deployed_params(model)

    if args.target_params is not None:
        target, match_name = args.target_params, None
    else:
        match_path = Path(args.match) if args.match else find_match(args.arch, args.dataset)
        if match_path is None:
            raise SystemExit("no C2 checkpoint to match; pass --match or --target-params")
        match_ck = torch.load(match_path, map_location="cpu", weights_only=False)
        target, match_name = int(match_ck["params"]), match_path.name

    use_amp = args.amp and args.device.startswith("cuda")
    model = model.to(args.device)
    ref_top1 = evaluate(model, val_loader, args.device, batch_tf=eval_tf, use_amp=use_amp)
    print(f"\n{'=' * 70}\n  PRUNING BASELINE  {args.criterion} / {args.mode}\n"
          f"  Arch / dataset     : {args.arch} / {args.dataset}\n"
          f"  Reference          : {ref_path.name}  {ref_top1 * 100:.2f}% at {ref_params:,} params\n"
          f"  Target             : {target:,} params ({100 * (target / ref_params - 1):+.2f}%)"
          f"{'  = ' + match_name if match_name else ''}\n{'=' * 70}\n")

    # --- prune on the same plastic set ---------------------------------------
    bundles = editable_bundles(model.cpu(), example)
    reference_widths = {t: b.size for t, b in bundles.items()}
    model.to(args.device)
    if args.criterion == "taylor":
        def loss_of(m, batch):
            x, y = batch
            x, y = x.to(args.device), y.to(args.device)
            if train_tf is not None:
                x = train_tf(x)
            return nn.functional.cross_entropy(m(x), y)
        importance_fn = make_importance_taylor(lambda: iter(train_loader), loss_of, args.taylor_batches)
    else:
        importance_fn = {"l1": importance_l1, "bnscale": importance_bnscale}[args.criterion]
    t0 = time.time()
    events = prune_to_target(model, bundles, target, importance_fn, min_width=args.min_width,
                             round_fraction=args.round_fraction, log=print)
    pruned_params = deployed_params(model)
    widths = {str(t): b.size for t, b in bundles.items()}
    with torch.no_grad():
        model.cpu().eval()(example)                   # every shape still legal
    print(f"pruning took {time.time() - t0:.1f}s; {len(events)} channels removed from "
          f"{sum(1 for t in bundles if widths[str(t)] != reference_widths[t])} of {len(bundles)} groups")
    if pruned_params > target:
        raise SystemExit(f"pruned model has {pruned_params:,} > target {target:,}")
    if target - pruned_params > 0.01 * target:
        raise SystemExit(f"undershot the target by more than 1%: {pruned_params:,} vs {target:,}")

    model = model.to(args.device)
    pruned_top1 = evaluate(model, val_loader, args.device, batch_tf=eval_tf, use_amp=use_amp)
    print(f"pruned, before any training: {pruned_top1 * 100:.2f}%")

    if args.mode == "scratch":
        # The pruned width vector as a fresh architecture, trained exactly as C2 is.
        torch.manual_seed(args.seed)
        model = resize_model_to_widths(build_model(args.arch, num_classes, cifar_stem=cifar_stem),
                                       widths, example).to(args.device)
        assert deployed_params(model) == pruned_params, "rebuilt width vector has a different count"

    record = {
        "control": f"pruned_{args.criterion}_{args.mode}",
        "arch": args.arch, "dataset": args.dataset, "cifar_stem": cifar_stem,
        "criterion": args.criterion, "mode": args.mode,
        "reference": ref_path.name, "reference_params": ref_params, "reference_top1": ref_top1,
        "match": match_name, "target_params": target, "params": pruned_params,
        "pruned_top1_before_training": pruned_top1,
        "reference_widths": {str(t): w for t, w in reference_widths.items()},
        "discovered_widths": widths, "removed": len(events),
        "epochs": epochs, "lr": lr, "args": vars(args),
    }
    best = train(model, train_loader, val_loader, args, args.device, train_tf, eval_tf,
                 epochs, lr, record, out)
    out.with_suffix(".events.json").write_text(json.dumps(events), encoding="utf-8")
    print(f"\n{'=' * 70}\n  {record['control']}  best {best * 100:.2f}% at {pruned_params:,} params "
          f"(reference {ref_top1 * 100:.2f}% at {ref_params:,})\n{'=' * 70}")


if __name__ == "__main__":
    main()
