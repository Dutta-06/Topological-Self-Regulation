"""Structured-pruning baseline on the same plastic set as TSR-X (vision).

Takes a dense reference, removes whole channels from exactly the coupling groups
TSR-X was allowed to edit (a producer and a consumer, no task-fixed axis) until
the deployed count reaches the matching C2 model's, then either fine-tunes the
pruned network (inherited weights) or retrains the pruned width vector from
scratch under the C2 recipe (Liu et al., 2019).

Self-contained: the target count comes from bench/pruning_targets.json (the
exact deployed count of the archived C2 for that cell) unless a C2 checkpoint
is present, and the reference is whatever results/reference/ holds for the
cell -- the archived one, or one trained on this machine by
scripts/run_pruning_baselines.py with the shared recipe. Every run writes a
JSON record next to its checkpoint with everything the comparison needs, so
the .pt files never have to leave the machine.

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
import platform
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

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
TARGETS = ROOT / "bench" / "pruning_targets.json"

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
# Locating the reference and the target
# ---------------------------------------------------------------------------

def reference_name(arch: str, dataset: str) -> str:
    return f"{ARCH_KEY[arch]}_{dataset.replace('-', '_')}.pt"


def find_reference(arch: str, dataset: str, cifar_stem: bool) -> Path:
    """The dense reference for a cell, chosen by inspecting the checkpoint:
    the stem must match the cell, and stamped-incomplete or aborted runs are refused."""
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
        complete = bool(ck.get("complete", True)) and int(ck.get("epoch", 99)) >= 10
        checked.append((path.name, ok, complete))
        if ok and complete:
            return path
    raise SystemExit(f"no usable dense reference for {arch}/{dataset} (cifar_stem={cifar_stem}); "
                     f"inspected {checked}. Train one with bench/train_reference.py or let "
                     f"scripts/run_pruning_baselines.py do it.")


def load_targets() -> dict:
    if not TARGETS.exists():
        return {}
    return json.loads(TARGETS.read_text(encoding="utf-8")).get("cells", {})


def find_match(arch: str, dataset: str) -> Optional[Path]:
    path = ROOT / "results" / "static_matched" / f"{ARCH_KEY[arch]}_{dataset.replace('-', '_')}_two_regime.pt"
    return path if path.exists() else None


def find_target(arch: str, dataset: str) -> Tuple[int, str]:
    """(target deployed count, where it came from): the archived C2 checkpoint
    if present on this machine, else the committed table of exact C2 counts."""
    match = find_match(arch, dataset)
    if match is not None:
        ck = torch.load(match, map_location="cpu", weights_only=False)
        return int(ck["params"]), match.name
    cell = load_targets().get(f"{arch}/{dataset}")
    if cell is None:
        raise SystemExit(f"no target for {arch}/{dataset}: no C2 checkpoint and no entry in "
                         f"{TARGETS.relative_to(ROOT)}; pass --target-params")
    return int(cell["target_params"]), f"pruning_targets.json ({cell['target_params']:,})"


def environment() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    return {"git_commit": commit, "torch": torch.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "hostname": platform.node()}


# ---------------------------------------------------------------------------
# Training (the C2 recipe)
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, args, device, train_tf, eval_tf, epochs, lr,
          record: dict, out: Path) -> dict:
    use_amp = args.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    best, history, t_start = 0.0, [], time.time()
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
        history.append({"epoch": epoch, "loss": tot / n, "val_acc": acc, "seconds": time.time() - t0})
        print(f"epoch {epoch + 1:3d}/{epochs}  loss={tot / n:.4f}  val_acc={acc:.4f}  "
              f"best={best:.4f}  ({time.time() - t0:.1f}s)")
        if is_best:
            torch.save({**record, "model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_acc": acc, "best_val_acc": best, "complete": False}, out)
    ck = torch.load(out, map_location="cpu", weights_only=False)
    ck["complete"] = True
    ck["total_epochs"] = epochs
    torch.save(ck, out)
    return {"best_val_acc": best, "best_epoch": int(ck["epoch"]), "history": history,
            "train_seconds": time.time() - t_start}


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
    ap.add_argument("--target-params", type=int, default=None,
                    help="explicit target (default: the C2 checkpoint, else bench/pruning_targets.json)")
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
        target, target_source = args.target_params, "--target-params"
    else:
        target, target_source = find_target(args.arch, args.dataset)
    table = load_targets().get(f"{args.arch}/{args.dataset}", {})
    if table and int(table["reference_params"]) != ref_params:
        raise SystemExit(f"this reference has {ref_params:,} params but the archived one had "
                         f"{table['reference_params']:,}: not the same architecture / stem")

    use_amp = args.amp and args.device.startswith("cuda")
    model = model.to(args.device)
    ref_top1 = evaluate(model, val_loader, args.device, batch_tf=eval_tf, use_amp=use_amp)
    archived_note = f"  (archived reference: {table['reference_top1']:.2f}%)" if table else ""
    print(f"\n{'=' * 70}\n  PRUNING BASELINE  {args.criterion} / {args.mode}\n"
          f"  Arch / dataset     : {args.arch} / {args.dataset}\n"
          f"  Reference          : {ref_path.name}  {ref_top1 * 100:.2f}% at {ref_params:,} params{archived_note}\n"
          f"  Target             : {target:,} params ({100 * (target / ref_params - 1):+.2f}%)  from {target_source}"
          f"\n{'=' * 70}\n")

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
    prune_seconds = time.time() - t0
    pruned_params = deployed_params(model)
    widths = {str(t): b.size for t, b in bundles.items()}
    with torch.no_grad():
        model.cpu().eval()(example)                   # every shape still legal
    print(f"pruning took {prune_seconds:.1f}s; {len(events)} channels removed from "
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
        "reference_trained_here": bool(ref_ck.get("trained_for") == "pruning-baselines"),
        "target_source": target_source, "target_params": target, "params": pruned_params,
        "pruned_top1_before_training": pruned_top1,
        "reference_widths": {str(t): w for t, w in reference_widths.items()},
        "discovered_widths": widths, "removed": len(events), "prune_seconds": prune_seconds,
        "epochs": epochs, "lr": lr, "batch_size": batch_size, "args": vars(args),
        "environment": environment(),
    }
    outcome = train(model, train_loader, val_loader, args, args.device, train_tf, eval_tf,
                    epochs, lr, record, out)
    out.with_suffix(".events.json").write_text(json.dumps(events), encoding="utf-8")

    # The committable record: everything the comparison needs, no tensors.
    summary = {k: v for k, v in record.items() if k != "args"}
    summary.update({"best_top1": outcome["best_val_acc"], "best_epoch": outcome["best_epoch"],
                    "train_seconds": outcome["train_seconds"], "history": outcome["history"],
                    "archived": {k: table[k] for k in ("reference_top1", "c2_top1", "tsr_top1")} if table else None,
                    "checkpoint": out.name, "complete": True})
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{'=' * 70}\n  {record['control']}  best {outcome['best_val_acc'] * 100:.2f}% at {pruned_params:,} params "
          f"(reference here {ref_top1 * 100:.2f}% at {ref_params:,})\n  record: {out.with_suffix('.json')}\n{'=' * 70}")


if __name__ == "__main__":
    main()
